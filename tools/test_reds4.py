"""Evalute Space-Time Video Super-Resolution on Vimeo90k dataset.
"""
import os
import glob
import cv2
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import logging
from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2ycbcr

from models import create_model
from Tester import Tester
from utils import (mkdirs, parse_config, AverageMeter, structural_similarity, calculate_psnr,
                   read_seq_images, index_generation, setup_logger, get_model_total_params)

from tqdm import tqdm                   

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def check_refill(psnr_list):
    lens = len(psnr_list)
    sub_lens_list = [len(x) for x in psnr_list]
    max_lens = max(sub_lens_list)

    for i, v in enumerate(psnr_list):
        if len(v) < max_lens:
            dis = max_lens - len(v)
            v += dis * [0.]
            assert len(v) == max_lens, 'the value has not been refilled.'

    return psnr_list        

def select_topk(input, lr_paths, sub_lr_paths, k=10):
    res_loc = []
    seg_n, clip_n = input.shape
    v, index = torch.topk(input.flatten(), k) 
    index = index.tolist()

    for i, x in enumerate(index):
        seg_index = int(x // clip_n)
        clip_index = int(x % clip_n)
        res_loc.append((lr_paths[seg_index], sub_lr_paths[seg_index][clip_index]))
    assert len(res_loc) == k

    return res_loc    

def psnr_exceed_check(psnr):
    assert isinstance(psnr, float), 'Wrong type of PSNR.'
    if psnr >= float('inf'):
        # eps_t = np.finfo(np.float64).eps
        eps_t = 0.65025
        psnr = 10 * np.log10(255. ** 2 / eps_t)  
        return float(psnr)
    elif psnr < 0:
        raise Exception('Wrong way of calculating psnr.')    
    else:
        return psnr    

def main():
    parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution Evaluation on Vimeo90k dataset')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
 
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    config = parse_config(args.config, is_train=False)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        config['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        config['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        # print(f'world_szie is {world_size}')
        rank = torch.distributed.get_rank()

  
    save_path = config['path']['save_path'] 
    mkdirs(save_path)
    setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)

    torch.backends.cudnn.benckmark = True

   

    model = Tester(config)
    

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Data: {} - {} - {}'.format(config['dataset']['name'], config['dataset']['mode'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    # logger.info('Model parameters: {} M'.format(model_params))

    # if config['dataset']['name'] == 'Vimeo90k_septuplet':
    if 'Vimeo' in config['name']:
        LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], config['dataset']['mode'], '*')))
    else:    
        LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], '*')))  

    PSNR = []
    PSNR_Y = []
    SSIM = []
    SSIM_Y = []
    FPS = []

    LR_paths_ = []
    sub_top10, top10 = [], []

    for LR_path in LR_paths:
        sub_save_path = os.path.join(save_path, LR_path.split('/')[-1])
        mkdirs(sub_save_path)  # ./results/00000

        sub_LR_paths = sorted(glob.glob(os.path.join(LR_path, '*')))  # [X4/001, ...]

        sub_LR_paths_ = [x.split('/')[-1] for x in sub_LR_paths]
        LR_paths_.append(sub_LR_paths_)

        seq_PSNR = AverageMeter()
        seq_PSNR_Y = AverageMeter()
        seq_SSIM = AverageMeter()
        seq_SSIM_Y = AverageMeter()
        seq_FPS = AverageMeter()
        for sub_LR_path in sub_LR_paths:
            sub_sub_save_path = os.path.join(sub_save_path, sub_LR_path.split('/')[-1])  
            mkdirs(sub_sub_save_path)  # ./results/00000/0045    

            tested_index = []

            # if config['dataset']['name'] == 'Vimeo90k_septuplet':
            if 'Vimeo' in config['name']:
                sub_GT_path = sub_LR_path.replace('_LR', '')
            elif 'RED' in config['name']:    
                sub_GT_path = sub_LR_path.replace('_bicubic/X4', '')
            elif 'Vid' or 'Udm' in config['name']:
                sub_GT_path = sub_LR_path.replace('LR/X4', 'GT')    

            imgs_LR = read_seq_images(sub_LR_path)  
            # imgs_LR = imgs_LR.astype(np.float32) / 255.  
            # imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()  # T C H W  
            imgs_GT = read_seq_images(sub_GT_path)
        
            if 'Vid' in config['name']:
                indices_list = index_generation(imgs_LR.shape[0], imgs_LR.shape[0])
            else:    
                indices_list = index_generation(config['dataset']['num_out_frames'], imgs_LR.shape[0])  # [[0 1 2 3 4 5 6]]
        
            clips_PSNR = AverageMeter()
            clips_PSNR_Y = AverageMeter()
            clips_SSIM = AverageMeter()
            clips_SSIM_Y = AverageMeter()
            clips_FPS = AverageMeter()
            
            for indices in indices_list:
                inputs = imgs_LR[indices]  # shape : [T, H, W, C]
                
                # with torch.no_grad():
                #     outputs = model(inputs)
                # outputs = outputs.cpu().squeeze().clamp(0, 1).numpy()

                if 'Vid' in config['name']:
                    # print(f'the outfrmaenum is {model.model.module.num_out_frames}.')
                    if config['network']['num_frames'] is None:
                        model.model.module.num_out_frames = inputs.shape[0]  # module 
                    else:
                        model.model.module.num_out_frames = config['network']['num_frames']    
                    # print(f'the value is {id(model.model.num_out_frames)}.')
                    # logger.info(f'dict is {model.model.__dict__}.')
                
                if config['dataset']['name'] != 'REDS':
                    outputs, fps = model.evaluate_fps(inputs, num_frames=inputs.shape[0])  # numpy
                else:
                    outputs, fps = model.evaluate_fps(inputs, num_frames=inputs.shape[0], HR=imgs_GT[indices])     
                # print(f'{outputs.shape}')
                
                # PSNR, SSIM for each frame
                for idx, frame_idx in enumerate(indices):
                    if frame_idx in tested_index:
                        continue
                    tested_index.append(frame_idx)
                    
                    # output = (outputs[idx].squeeze().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
                    # target = imgs_GT[frame_idx]
                    # output_y = rgb2ycbcr(output[..., ::-1])[..., 0]
                    # target_y = rgb2ycbcr(target[..., ::-1])[..., 0]

                    # output = np.round(np.ascontiguousarray(outputs[idx].squeeze().transpose(1, 2, 0)) * 255.0).astype(np.uint8)
                    output = outputs[idx]  # (H, W, C)
                    target = imgs_GT[frame_idx]
                    # print(f'{output.shape}{target.shape}')
                    output_y = rgb2ycbcr(output)[..., 0]  
                    target_y = rgb2ycbcr(target)[..., 0]

                    psnr = calculate_psnr(output, target, border=0)
                    psnr_y = calculate_psnr(output_y, target_y, border=0)

                    # psnr = peak_signal_noise_ratio(output, target)
                    # psnr_y = peak_signal_noise_ratio(output_y, target_y, data_range=255)

                    ssim = structural_similarity(output, target)
                    ssim_y = structural_similarity(output_y, target_y)

                    # psnr = psnr_exceed_check(psnr)
                    # psnr_y = psnr_exceed_check(psnr_y)

                    cv2.imwrite(os.path.join(sub_sub_save_path, '{}-{:08d}.png'.format(config['name'], frame_idx+1)), output[..., ::-1])

                    if config['dataset']['eval_mid_clip']:
                        if idx == len(indices) // 2 and config['dataset']['use_mirrors'] is False:  # vimeo-7 frames
                            
                            clips_PSNR.update(psnr)
                            clips_PSNR_Y.update(psnr_y)
                            clips_SSIM.update(ssim)
                            clips_SSIM_Y.update(ssim_y)
                            logger.info(f'calculating on mid frame: PSNR: {psnr}, PSNR-Y: {psnr_y}, SSIM: {ssim}, SSIM-Y: {ssim_y}.')
                        elif config['dataset']['use_mirrors']:    
                            if idx == 3 or idx == 10:  # viemo-14  mirror
                                clips_PSNR.update(psnr)
                                clips_PSNR_Y.update(psnr_y)
                                clips_SSIM.update(ssim)
                                clips_SSIM_Y.update(ssim_y)
                                logger.info(f'calculating on mid-mirror frame: PSNR: {psnr}, PSNR-Y: {psnr_y}, SSIM: {ssim}, SSIM-Y: {ssim_y}.')
                    else:        
                        clips_PSNR.update(psnr)
                        clips_PSNR_Y.update(psnr_y)
                        clips_SSIM.update(ssim)
                        clips_SSIM_Y.update(ssim_y)

                    msg = '{:3d} - PSNR: {:.6f} dB  PSNR-Y: {:.6f} dB ' \
                        'SSIM: {:.6f} SSIM-Y: {:.6f}'.format(
                            frame_idx + 1, psnr, psnr_y, ssim, ssim_y
                        )
                    logger.info(msg)

            msg = 'Folder {}/{} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
                'Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} frames; '.format(
                    LR_path.split('/')[-1], sub_LR_path.split('/')[-1], 
                    clips_PSNR.average(), clips_PSNR_Y.average(), 
                    clips_SSIM.average(), clips_SSIM_Y.average(), 
                    clips_PSNR.count
                )
            logger.info(msg)
            
            seq_PSNR.update(clips_PSNR.average())
            seq_PSNR_Y.update(clips_PSNR_Y.average())
            seq_SSIM.update(clips_SSIM.average())
            seq_SSIM_Y.update(clips_SSIM_Y.average())
            seq_FPS.update(fps)
            sub_top10.append(clips_PSNR_Y.average())

        msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
              'Average SSIM: {:.6f} SSIM-Y: {:.6f} for {} clips; '.format(
                    LR_path.split('/')[-1], seq_PSNR.average(), 
                    seq_PSNR_Y.average(), seq_SSIM.average(), 
                    seq_SSIM_Y.average(), seq_PSNR.count
               )
        logger.info(msg)
        
        PSNR.append(seq_PSNR.average())
        PSNR_Y.append(seq_PSNR_Y.average())
        SSIM.append(seq_SSIM.average())
        SSIM_Y.append(seq_SSIM_Y.average())
        FPS.append(seq_FPS.average())
        top10.append(sub_top10)
        sub_top10 = []

    
    top10 = check_refill(top10)
    top10 = torch.tensor(top10, dtype=torch.float32)  # 96*1000
    res_loc = select_topk(top10, LR_paths, LR_paths_, k=config['dataset']['selected_topk'])

    logger.info('################ Tidy Outputs ################')
    for path, psnr, psnr_y, ssim, ssim_y, fps in zip(LR_paths, PSNR, PSNR_Y, SSIM, SSIM_Y, FPS):
        msg = 'Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB ' \
              'SSIM: {:.6f} dB SSIM-Y: {:.6f} dB FPS: {:.6f} fps. '.format(
                  path.split('/')[-1], psnr, psnr_y, ssim, ssim_y, fps
              )
        logger.info(msg)
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {} - {}'.format(config['dataset']['name'], config['dataset']['mode'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    msg = 'Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB SSIM: {:.6f} dB ' \
          'SSIM-Y: {:.6f} dB FPS: {:.6f} fps for {} clips.'.format(
              sum(PSNR) / len(PSNR), sum(PSNR_Y) / len(PSNR_Y), 
              sum(SSIM) / len(SSIM), sum(SSIM_Y) / len(SSIM_Y), 
              sum(FPS) / len(FPS), len(PSNR)
          )
    logger.info(msg)

  
    logger.info(f'topk is {res_loc}')

    logger.info(f'FLOPs is {model.FLOPs}.')

if __name__ == '__main__':
    main()
    