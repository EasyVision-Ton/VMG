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

logger = logging.getLogger('base')


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config['gpu_ids'] is not None else 'cpu')
        self.is_train = config['is_train']
        if config['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1
        train_config = config['train']
        self.train_configs = config['train']
        self.model = create_model(config).to(self.device)
        if config['dist']:
            self.model = DistributedDataParallel((self.model), find_unused_parameters=train_config['f_u_params'], device_ids=[torch.cuda.current_device()])
        else:
            self.model = DataParallel(self.model)
        self.get_total_parameters(self.model)
        self.pre_model = self.config['path']['pretrain_model']
        if self.pre_model is not None and self.config['network']['spynet'] is None:
            self.pretrained_p_name, pretrained_p = [], []
            for k, v in self.model.module.state_dict().items():
                k_d = k.split('.')
                if 'align_t_down' in k_d:  # traj_mixing
                    self.pretrained_p_name.append(k)
            for k, v in self.model.module.named_parameters():
                k_d = k.split('.')
                if 'align_t_down' in k_d:        
                    pretrained_p.append(v)  
        else:
            pretrained_p = None
        self.load(pretrain_path=self.pre_model)

        if self.is_train:
            self.model.train()
            # loss
            self.criterion = CharbonnierLoss(eps=float(config['train']['eps']), 
                                             if_aux_loss=config['train']['if_aux'],  # if add aux loss 
                                             aux_ratio=config['train']['aux_ratio']).to(self.device)  
            wd = train_config['weight_decay'] if train_config['weight_decay'] else 0
            logger.info(f"weight_decay is {wd}.")
            if not config['train']['pre_training']:
                optim_params = []
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            else:
                pre_train_param, original_param = [], []
                if self.pre_model is None or self.config['network']['spynet'] is not None:
                   
                    pretrained_param = list(self.model.module.spynet.parameters())
                else:
                    pretrained_param = pretrained_p    
                pre_train_param = list(map(id, pretrained_param))
                original_param = filter((lambda p: id(p) not in pre_train_param), self.model.module.parameters())
            if wd > 0:  # switch on the weight decay
                wd_param_id = map(id, self.model.module.mlp_wd_param)
                if train_config['pre_training']:
                    wd_param = filter((lambda p: id(p) in wd_param_id), original_param)
                    original_param = filter((lambda p: id(p) not in wd_param_id), original_param)
                else:
                    original_param = filter((lambda p: id(p) not in wd_param_id), optim_params)
                    wd_param = filter((lambda p: id(p) in wd_param_id), self.model.module.parameters())
            else:
                wd_param = None
            if not config['train']['pre_training']:
                if wd <= 0:
                    self.optimizer = torch.optim.AdamW(optim_params, lr=(train_config['lr']), weight_decay=wd,
                      betas=(
                     train_config['beta1'], train_config['beta2']))
                else:
                    self.optimizer = torch.optim.AdamW([{'params':wd_param,  'weight_decay':wd}, {'params': original_param}], lr=(train_config['lr']),
                      weight_decay=0.0,
                      betas=(
                     train_config['beta1'], train_config['beta2']))
            else:
                if wd <= 0:
                    self.optimizer = torch.optim.AdamW([{'params': pretrained_param,  'lr':0.}, {'params': original_param}], lr=(train_config['lr']),
                      weight_decay=wd,
                      betas=(
                     train_config['beta1'], train_config['beta2']))
                else:
                    self.optimizer = torch.optim.AdamW([{'params': pretrained_param,  'lr':0.}, {'params': original_param}, {'params':wd_param,  'weight_decay':wd}], lr=(train_config['lr']),
                      weight_decay=0.,  
                      betas=(
                     train_config['beta1'], train_config['beta2']))
            self.scheduler = CosineAnnealingLR_Restart((self.optimizer),
              (train_config['T_period']), eta_min=(train_config['eta_min']),
              restarts=(train_config['restarts']),
              weights=(train_config['restart_weights']))
            self.log_dict = OrderedDict()
        self.loss_ = []
        self.if_amp = config['train']['amp']
        self.if_clip = config['train']['if_grad_clip']
        self.grad_clip_v = config['train']['grad_clip_up']
        self.scaler = gradscaler()

        if train_config['reduced_iter'] is not None:
            self.recover_flag = False
            self.rd_iter = train_config['reduced_iter']
        else:
            self.recover_flag = None    

        print(f"the optimizer_lens is {len(self.optimizer.param_groups)}.")

    def train_one_sample(self, data, step, freq, flow_pretrained=None, grad_acc=1, revise_epoch=True, update_acc=False):
        self.inputs = data['LRs'].to(self.device)
        self.targets = data['HRs'].to(self.device)

        if not revise_epoch:
        # if grad_acc == 1:
            self.optimizer.zero_grad()
            if self.config['train']['amp']:
                with autocast():
                    self.outputs = self.model(self.inputs, flow_pretrained, self.if_amp)
                    loss = self.criterion(self.outputs, self.targets)
                # loss.backward()
                self.scaler.scale(loss).backward()
                if self.if_clip:
                    self.scaler.unscale_(self.optimizer)
                    grad_clip(self.model.module.parameters(), max_norm=self.grad_clip_v, norm_type=2)
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.outputs = self.model(self.inputs, flow_pretrained)
                loss = self.criterion(self.outputs, self.targets)
                loss.backward()
                if self.if_clip:
                    grad_clip(self.model.module.parameters(), max_norm=self.grad_clip_v, norm_type=2)
                self.optimizer.step()
           
            self.update_learning_rate(step, warmup_iter=self.config['train']['warmup_iter'])
            # set log
            if (step+1) % freq == 0:     
                # self.log_dict['loss'] = loss.item()
                self.log_dict['loss'] = torch.mean(torch.tensor(self.loss_)).item()
                self.loss_ = []
            else:
                self.loss_.append(loss.detach())    
        elif revise_epoch:
        # elif grad_acc > 1:
            if self.config['train']['amp']:
                with autocast():
                    self.outputs = self.model(self.inputs, flow_pretrained)
                    loss = self.criterion(self.outputs, self.targets) / grad_acc
                self.scaler.scale(loss).backward() 
                if self.if_clip:
                    self.scaler.unscale_(self.optimizer)
                    grad_clip(self.model.module.parameters(), max_norm=self.grad_clip_v, norm_type=2)   

                # if (step+1) % grad_acc == 0:
                if update_acc:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # self.update_learning_rate(step+1-grad_acc, warmup_iter=self.config['train']['warmup_iter'])
                    self.update_learning_rate(step, warmup_iter=self.config['train']['warmup_iter'])
            else:
                self.outputs = self.model(self.inputs, flow_pretrained)
                loss = self.criterion(self.outputs, self.targets) / grad_acc
                loss.backward()
                if self.if_clip:
                    grad_clip(self.model.module.parameters(), max_norm=self.grad_clip_v, norm_type=2)

                # if (step+1) % grad_acc == 0:
                if update_acc:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.update_learning_rate(step+1-grad_acc, warmup_iter=self.config['train']['warmup_iter'])
                    self.update_learning_rate(step, warmup_iter=self.config['train']['warmup_iter'])
            # set log
            if (step+2) % freq == 0 and update_acc is True:  
                # self.log_dict['loss'] = loss.item()
                self.log_dict['loss'] = torch.mean(torch.tensor(self.loss_)).item()
                self.loss_ = []
                # print(f"loss is {self.log_dict['loss']}")
            else:
                self.loss_.append(loss.detach())        
        else:
            raise Exception('please check up the grad_acc.')        

       

    def get_current_log(self):
        return self.log_dict

    def get_total_parameters(self, model):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        total_parameters = sum(map(lambda x: x.numel(), model.parameters()))

        net_struc_str = '{}'.format(model.__class__.__name__)

        if self.rank <= 0:
            logger.info('Model structure: {}, with parameters: {:,d}'.format(net_struc_str, total_parameters))

    def load(self, pretrain_path=None):
        if pretrain_path is None:
            load_path = self.config['path']['pretrain_model']
        else:
            load_path = pretrain_path    
        print(f'load_path is {load_path}.')
        if load_path is not None:
            logger.info('Loading model [{:s}] ...'.format(load_path))
            if self.config['path']['resume_state'] is not None:
                self.load_model(load_path, self.model, self.config['path']['strict_load'])
            else:
                self.load_model_with_pretraining(load_path, self.model, self.config['path']['strict_load'])    

    def save(self, iter_label):
        self.save_model(self.model, iter_label)

    def _set_lr(self, lr_groups):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups = [v['initial_lr'] for v in self.optimizer.param_groups]
        return init_lr_groups

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        if self.recover_flag:
            self.optimizer.param_groups[1]['lr'] = self.past_lr
            self.scheduler.step()
        else:    
            self.scheduler.step()

        if self.recover_flag is not None:
            if cur_iter >= self.rd_iter:
                self.past_lr = self.optimizer.param_groups[1]['lr']
                self.recover_flag = True
                self.optimizer.param_groups[1]['lr'] *= 0.5
            else:
                self.recover_flag = False        

        if self.train_configs['pre_training']:
            if cur_iter <= self.config['network']['flow_fix']:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['initial_lr']
            else:
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[1]['lr'] * self.train_configs['pre_lr_ratio']
                    
        # set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_groups = self._get_init_lr()
            # modify warming-up learning rates
            warmup_lr = [v / warmup_iter * cur_iter for v in init_lr_groups]
            # set learning rate
            self._set_lr(warmup_lr)

    def get_current_learning_rate(self):
        lr_l = []
        for param_group in self.optimizer.param_groups:
            lr_l.append(param_group['lr'])
        return lr_l


    def save_model(self, model, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(self.config['path']['models'], save_filename)
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_model(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        model.load_state_dict(load_net_clean, strict=strict)

    def load_model_with_pretraining_(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        load_net = torch.load(load_path)
        # load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        load_net_clean = model.state_dict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if k in self.pretrained_p_name:
                # print('tctctc')
                load_net_clean.update({k:v})     
        model.load_state_dict(load_net_clean, strict=strict)

    def load_model_with_pretraining(self, load_path, model, strict=True):
        if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
            model = model.module
        load_net = torch.load(load_path)['state_dict']
        # load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        load_net_clean = model.state_dict()
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if 'backbone' in k[:9]:
                k = k.replace('backbone', 'align_t_down', 1)
            if k in self.pretrained_p_name:
                # print('tctctc')
                k_l = k.split('.')
                if '2' in k_l[2]:
                    if '0' in k_l[4]:
                        # print(f'ttcc')
                        k_l[4] = '4'
                        kl = '.'.join(k_l).replace('align_t_down', 'backbone', 1)
                        v = load_net[kl]
                        k1 = k.replace('align_t_down', 'align_t_up', 1)
                        load_net_clean.update({k:v, k1:v})
                            
                    elif '1' in k_l[4]:
                        k_l[4] = '5'
                        kl = '.'.join(k_l).replace('align_t_down', 'backbone', 1)
                        v = load_net[kl]
                        k1 = k.replace('align_t_down', 'align_t_up', 1)
                        load_net_clean.update({k:v, k1:v})      
                else:
                    if '0' in k_l[2] or '1' in k_l[2]:
                        k1 = k.replace('align_t_down', 'align_t_up', 1)
                        load_net_clean.update({k:v, k1:v})
                    else:
                        load_net_clean.update({k:v})    
                    
        model.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {
            'epoch': epoch, 
            'iter': iter_step, 
            'scheduler': self.scheduler.state_dict(), 
            'optimizer': self.optimizer.state_dict()
        }
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.config['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizer = resume_state['optimizer']
        resume_scheduler = resume_state['scheduler']
        self.optimizer.load_state_dict(resume_optimizer)
        self.scheduler.load_state_dict(resume_scheduler)
