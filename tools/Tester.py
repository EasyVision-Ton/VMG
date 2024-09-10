import logging
from collections import OrderedDict

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from models import create_model
from utils import CharbonnierLoss, CosineAnnealingLR_Restart
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as gradscaler
from torch.nn.utils import clip_grad_norm_ as grad_clip

import random
import numpy as np
import time

from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
from skimage.metrics import peak_signal_noise_ratio

logger = logging.getLogger('base')

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

class Tester():
    def __init__(self, config):

        self.config = config
        self.device = torch.device('cuda' if config['gpu_ids'] is not None else 'cpu')  # current devices
        self.is_train = config['is_train']  # False
        self.hflip = config['dataset']['use_hflip']
        self.vflip = config['dataset']['use_vflip']
        self.rot = config['dataset']['use_rot']
        self.mirrors = config['dataset']['use_mirrors']
        self.data_enhance = config['dataset']['data_enhance']
        self.dataset_name = config['dataset']['name']
        self.scale = config['scale']

        self.test_num_frames = config['dataset']['num_frames']
        if config['dataset']['overlapped_mode'] == 'small':
            self.overlapped_num_frames = 2
        elif config['dataset']['overlapped_mode'] == 'mid':     
            self.overlapped_num_frames = config['dataset']['num_frames'] // 2
        elif config['dataset']['overlapped_mode'] == 'large':
            self.overlapped_num_frames = config['dataset']['num_frames'] - 1
        elif not isinstance(config['dataset']['overlapped_mode'], str):  
            self.overlapped_num_frames = int(config['dataset']['overlapped_mode'])    
        else:
            raise Exception('choose right mode of testing.')    
        
        self.test_spatial = config['dataset']['wins']
        self.overlapped_spatial_length = config['dataset']['overlapped_spatial_length']

        if config['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        # train_config = config['train']

        # define model and load pretrained model
        # raise Exception('!!!!!!!!!!!!!!')
        self.model = create_model(config)
        # raise Exception('!!!!!!!!!!!!!!')
       
        inputs = (torch.rand((1, config['dataset']['flops_num_frames'], 
                                 config['dataset']['image_shape'][0], 
                                 int(config['dataset']['image_shape'][1]//self.scale), 
                                 int(config['dataset']['image_shape'][2]//self.scale))))
        
        if config['dataset']['FLOPs']:
            self.FLOPs = self.get_FLOPs(inputs=inputs)  
            print('FLOPs=', str(self.FLOPs / 1e9) + '{}'.format('G'))
        else:
            self.FLOPs = None                       

        self.model = self.model.to(self.device)

        if config['dist']:
            self.model = DistributedDataParallel(self.model, find_unused_parameters=True, device_ids=[torch.cuda.current_device()])
        else:
            self.model = DataParallel(self.model)
        
        self.get_total_parameters(self.model)
        # raise Exception('!!!!!!!!!!!!!!')

        self.checkpoint_from = config['checkpoint_from']
        self.load()
        # raise Exception('!!!!!!!!!!!!!!')

        if not self.is_train:
            self.model.eval()
        else:
            raise Exception('while testing, keep training process close.')    

   
    @torch.no_grad()    
    def test_image(self, inputs):
        B, T, C, H, W = inputs.shape

        stride_h = self.test_spatial[0] - self.overlapped_spatial_length
        stride_w = self.test_spatial[1] - self.overlapped_spatial_length
        h_idx_list = list(range(0, H-self.test_spatial[0], stride_h)) + [max(0, H - self.test_spatial[0])]
        w_idx_list = list(range(0, W-self.test_spatial[1], stride_w)) + [max(0, W - self.test_spatial[1])]
        E = inputs.new_zeros(B, T, C, H*self.scale, W*self.scale)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = inputs[..., h_idx:h_idx+self.test_spatial[0], w_idx:w_idx+self.test_spatial[1]]
                out_patch = self.model(in_patch)

                out_patch_mask = torch.ones_like(out_patch)

            
                if h_idx < h_idx_list[-1]:
                    out_patch[..., -self.overlapped_spatial_length//2:, :] *= 0
                    out_patch_mask[..., -self.overlapped_spatial_length//2:, :] *= 0
                if w_idx < w_idx_list[-1]:
                    out_patch[..., :, -self.overlapped_spatial_length//2:] *= 0
                    out_patch_mask[..., :, -self.overlapped_spatial_length//2:] *= 0
                if h_idx > h_idx_list[0]:
                    out_patch[..., :self.overlapped_spatial_length//2, :] *= 0
                    out_patch_mask[..., :self.overlapped_spatial_length//2, :] *= 0
                if w_idx > w_idx_list[0]:
                    out_patch[..., :, :self.overlapped_spatial_length//2] *= 0
                    out_patch_mask[..., :, :self.overlapped_spatial_length//2] *= 0

                E[..., h_idx*self.scale:(h_idx+self.test_spatial[0])*self.scale, w_idx*self.scale:(w_idx+self.test_spatial[1])*self.scale].add_(out_patch)
                W[..., h_idx*self.scale:(h_idx+self.test_spatial[0])*self.scale, w_idx*self.scale:(w_idx+self.test_spatial[1])*self.scale].add_(out_patch_mask)

        outputs = E.div_(W)
        return outputs

   
    def test_clips(self, inputs):
        B, T, C, H, W = inputs.shape
        not_overlap_border = True if self.overlapped_num_frames > 0 else False
        E = inputs.new_zeros(B, T, C, H*self.scale, W*self.scale)
        N = inputs.new_zeros(B, T, 1, 1, 1)
        N_clips = inputs.new_ones(B, self.test_num_frames, 1, 1, 1)

        stride = self.test_num_frames - self.overlapped_num_frames  
        self.t_idx_list = list(range(0, T-self.test_num_frames, stride)) + [max(0, T-self.test_num_frames)]  # [0, 4, 8, 12, 16, 20, 24, 28, 32, 42-7=35]

        for t_idx in self.t_idx_list:
            inputs_clips = inputs[:, t_idx:t_idx+self.test_num_frames, ...]  # B, test_frames_num, C, H, W
            if self.overlapped_spatial_length is None:
                outputs_clips = self.model(inputs_clips)
            else:
                outputs_clips = self.test_image(inputs_clips)    
            N_clips = inputs.new_ones(B, self.test_num_frames, 1, 1, 1)

            if not_overlap_border:  
                if t_idx < self.t_idx_list[-1]:  
                    outputs_clips[:, -self.overlapped_num_frames//2:, ...] *= 0
                    N_clips[:, -self.overlapped_num_frames//2:, ...] *= 0
                if t_idx > self.t_idx_list[0]:  
                    outputs_clips[:, :self.overlapped_num_frames//2, ...] *= 0
                    N_clips[:, :self.overlapped_num_frames//2, ...] *= 0

            E[:, t_idx:t_idx+self.test_num_frames, ...].add_(outputs_clips)
            N[:, t_idx:t_idx+self.test_num_frames, ...].add_(N_clips)

        outputs = E.div_(N)

        return outputs    
    
    # add psnr
    def test_clips_max(self, inputs, HR):
        B, T, C, H, W = inputs.shape

        not_overlap_border = True if self.overlapped_num_frames > 0 else False
        stride = self.test_num_frames - self.overlapped_num_frames  
        self.t_idx_list = list(range(0, T-self.test_num_frames, stride)) + [max(0, T-self.test_num_frames)]  # [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 84]
        E = inputs.new_zeros(B, T, len(self.t_idx_list), C, H*self.scale, W*self.scale)
        N = inputs.new_zeros(B, T, 1, 1, 1)
        psnrs = inputs.new_zeros(B, T, len(self.t_idx_list))

        # begin calculating
        for idx, t_idx in enumerate(self.t_idx_list):
            inputs_clips = inputs[:, t_idx:t_idx+self.test_num_frames, ...]  # B, test_frames_num, C, H, W
            hr = HR[:, t_idx:t_idx+self.test_num_frames, ...]
            if self.overlapped_spatial_length is None:
                outputs_clips = self.model(inputs_clips)
            else:
                outputs_clips = self.test_image(inputs_clips)    
            # psnr statistics
            for i in range(self.test_num_frames):
               
                image_test = outputs_clips[:, i, ...].squeeze().permute(1, 2, 0).contiguous().cpu().clamp(0, 1).numpy()
                image_true = hr[:, i, ...].squeeze().permute(1, 2, 0).contiguous().cpu().clamp(0, 1).numpy()
                psnr = peak_signal_noise_ratio(image_test, image_true)
                psnr = psnr_exceed_check(psnr)
                psnrs[:, t_idx+i, idx] = psnr

            E[:, t_idx:t_idx+self.test_num_frames, idx, ...].add_(outputs_clips)

        _, max_idx = torch.max(psnrs, dim=-1)  # B, T    
        max_idx = max_idx.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand(-1, -1, -1, C, H*self.scale, W*self.scale)
        out = torch.gather(E, dim=2, index=max_idx).squeeze()  # B T C Hr Wr

        return out

    def evaluate(self, inputs, HR=None, extra_fps=False):
        
        imgs_LR = inputs.astype(np.float32) / 255. 
        imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()  # T C H W
        inputs = imgs_LR.unsqueeze(0)  # shape is 1 T C H W 
        if self.data_enhance:
            inputs = self.augment(inputs, self.hflip, self.vflip, self.rot)
        inputs = inputs.to(self.device)

        if HR is not None:
            HR = HR.astype(np.float32) / 255.  
            HR = torch.from_numpy(HR).permute(0, 3, 1, 2).contiguous()  # T C H W
            HR = HR.unsqueeze(0)  # shape is 1 T C H W 
            if self.data_enhance:
                HR = self.augment(HR, self.hflip, self.vflip, self.rot)
            HR = HR.to(self.device)
        
        # inference
        with torch.no_grad():
            if self.dataset_name == 'Vimeo90k_septuplet' and isinstance(self.dataset_name, str):
                if self.overlapped_spatial_length is None:
                    # print(f'value is {self.model.num_out_frames, self.model.__dict__}.')
                    outputs = self.model(inputs)
                else:
                    outputs = self.test_image(inputs)    
            elif self.dataset_name == 'REDS' and isinstance(self.dataset_name, str):
                outputs = self.test_clips_max(inputs, HR)
            else:
                outputs = self.test_clips(inputs)    
        
        if self.data_enhance:
            outputs = self.augment_inverse(outputs.cpu(), self.hflip, self.vflip, self.rot)

       
        outputs = outputs.cpu().squeeze().clamp(0, 1).numpy()
        outputs = np.round(np.ascontiguousarray(outputs.squeeze().transpose(0, 2, 3, 1)) * 255.0).astype(np.uint8)

        return outputs
    
    def evaluate_extra_fps(self, inputs, HR=None, extra_fps=False):
        
        imgs_LR = inputs.astype(np.float32) / 255.  
        imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()  # T C H W
        inputs = imgs_LR.unsqueeze(0)  # shape is 1 T C H W 
        if self.data_enhance:
            inputs = self.augment(inputs, self.hflip, self.vflip, self.rot)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            if self.overlapped_spatial_length is None:
                outputs = self.model(inputs)
            else:
                outputs = self.test_image(inputs)    
            # if self.dataset_name == 'Vimeo90k_septuplet' and isinstance(self.dataset_name, str):
                # outputs = self.model(inputs)
            # elif self.dataset_name == 'REDS' and isinstance(self.dataset_name, str):
                # outputs = self.test_clips_max(inputs)
            # else:
                # outputs = self.test_clips(inputs)    
        
        if self.data_enhance:
            outputs = self.augment_inverse(outputs.cpu(), self.hflip, self.vflip, self.rot)

      
        outputs = outputs.cpu().squeeze().clamp(0, 1).numpy()
        outputs = np.round(np.ascontiguousarray(outputs.squeeze().transpose(0, 2, 3, 1)) * 255.0).astype(np.uint8)

        return outputs

    def evaluate_fps(self, inputs, num_frames=7, HR=None):
       
        if self.dataset_name != 'REDS':
            torch.cuda.synchronize()
            begin_forward = time.time()
            outputs = self.evaluate(inputs)
            torch.cuda.synchronize()
            end_forward = time.time()
        else:  
            outputs = self.evaluate(inputs, HR)    
            torch.cuda.synchronize()
            
            begin_forward = time.time()
            outputs_fps = self.evaluate_extra_fps(inputs[:self.test_num_frames, ...])
            torch.cuda.synchronize()
            end_forward = time.time()
            fps = float(self.test_num_frames * 1. / (end_forward - begin_forward)) 

        if self.dataset_name == 'Vimeo90k_septuplet' and isinstance(self.dataset_name, str):
            fps = float(num_frames * 1. / (end_forward - begin_forward))
        elif self.dataset_name == 'REDS':
            fps = fps    
        else:
            fps = float(self.test_num_frames * len(self.t_idx_list) / (end_forward - begin_forward))    
        return outputs, fps

    def load(self):
        load_path = self.config['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model [{:s}] ...'.format(load_path))

            if self.checkpoint_from == 'mine':
                self.load_model(load_path, self.model, self.config['path']['strict_load'])
            elif self.checkpoint_from == 'ST':
                self.load_model_ST(load_path, self.model, self.config['path']['strict_load'])
            elif self.checkpoint_from == 'KAIR':
                self.load_model_KAIR(load_path, self.model, self.config['path']['strict_load'])        
            else:
                raise Exception('Please write a correct origin of checkpoint.')    

    def load_model(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            # model = model
            model = model.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=strict)

    
    def load_model_ST(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            # model = model
            model = model.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k == 'state_dict':
                for m, n in v.items():
                    if m.startswith('generator.'):
                        load_net_clean[m[10:]] = n
            # if k.startswith('module.'):
            #     load_net_clean[k[7:]] = v
            # else:
            #     load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=strict)            

   
    def load_model_KAIR(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            # model = model
            model = model.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k == 'params':
                for m, n in v.items():
                    load_net_clean[m] = n
                    # if m.startswith('generator.'):
                        # load_net_clean[m[10:]] = n
            # if k.startswith('module.'):
            #     load_net_clean[k[7:]] = v
            # else:
            #     load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=strict)

    def get_total_parameters(self, model):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        total_parameters = sum(map(lambda x: x.numel(), model.parameters()))

        # net_struc_str = '{}'.format(model.__class__.__name__)
        # if self.rank <= 0:
            # logger.info('Model structure: {}, with parameters: {:,d}'.format(net_struc_str, total_parameters))

        if self.rank <= 0:
            logger.info('With parameters: {:,d}'.format(total_parameters))    

    def augment(self, img_clip, hflip=True, vflip=True, rot90=True):
      
        B, D, C, H, W = img_clip.shape
        assert B == 1, 'testing.'

        img_list = list(torch.chunk(img_clip, D, dim=1))
        # print(f'res is {vflip}')
        # hflip = hflip and random.random() < 0.5
        # vflip = vflip and random.random() < 0.5
        # rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                # img = img[:, ::-1, :]
                img = img[..., ::-1]
            if vflip:
                # img = img[::-1, :, :]
                img = img[..., ::-1, :]
            if rot90:
                # img = img.transpose(1, 0, 2)
                # img = img.permute(0, 1, 2, 4, 3).contiguous()
                img = img.transpose(0, 1, 2, 4, 3)
            img = np.ascontiguousarray(img)    
            return torch.from_numpy(img).float()

        res = [_augment(img.numpy()) for img in img_list]
        out = torch.cat(res, 1)
        assert out.shape[1] == D
        return out

    def augment_inverse(self, img_clip, hflip=True, vflip=True, rot90=True):
      
        B, D, C, H, W = img_clip.shape
        assert B == 1, 'testing.'

        img_list = list(torch.chunk(img_clip, D, dim=1))
        # print(f'res is {vflip}')
        # hflip = hflip and random.random() < 0.5
        # vflip = vflip and random.random() < 0.5
        # rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                # img = img[:, ::-1, :]
                img = img[..., ::-1]
            if vflip:
                # img = img[::-1, :, :]
                img = img[..., ::-1, :]
            if rot90:
                # img = img.transpose(1, 0, 2)
                # img = img.permute(0, 1, 2, 4, 3).contiguous()
                img = img.transpose(0, 1, 2, 4, 3)
            img = np.ascontiguousarray(img)    
            return torch.from_numpy(img).float()

        res = [_augment(img.numpy()) for img in img_list]
        out = torch.cat(res, 1)
        assert out.shape[1] == D
        return out

    def get_FLOPs(self, inputs, mode='thop'):

        if mode == 'fvcore':
            FLOPs = FlopCountAnalysis(self.model, inputs)

            return FLOPs.total()
        elif mode == 'thop':
            print(f'the size of input is {inputs.shape}.')
            FLOPs, Params = profile(self.model, inputs=(inputs,))

            return FLOPs
