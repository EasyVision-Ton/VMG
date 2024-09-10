import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import repeat
from torch.cuda.amp import autocast as autocast
from timm.models.layers import DropPath
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30, r_scaling=1.):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN0, num_blocks, mid_channels=out_channels, res_scale=r_scaling))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow  
    
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)

    grid_flow = grid_flow.to(dtype=x.dtype)

    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


def Conv3x3ReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1ReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


def Conv1x1(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_channels)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=1, stride=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = int(in_channels * expansion_factor)

       
        self.bottleneck = nn.Sequential(
            Conv1x1ReLU(in_channels, mid_channels),
            Conv3x3ReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            Conv1x1(mid_channels, out_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(-1, C, H, W)
        assert self.in_channels == self.out_channels
        x = x + self.bottleneck(x)
        return x


class ResidualBlockNoBN0(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        
        # if res_scale == 1.0:
        #     self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale




class Trajectory_multi_head(nn.Module):
    def __init__(self, embed_dim=96, 
                 mode='max',
                 num_blocks=10, stride=4, frame_stride=3, traj_win=4, head=4, 
                 en_field=False, 
                 head_scale=False, 
                 feature_refine='d',
                 if_ia=False,
                 r_scaling=1.,
                 twins=[2, 2],
                 learnable=False,
                 if_win_par=False,
                 ltam=True):

        super().__init__()

        self.embed_dims = embed_dim
        self.keyframe_stride = frame_stride  # 3
        self.stride = stride  # 4
        self.if_win_par = if_win_par
        self.ltam = ltam

        self.learnable = learnable
        if learnable:
            # self.q_w = nn.Linear(embed_dim, embed_dim)
            self.q_w = nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)

        if ltam:
            self.LTAM = LTAM_multi_head(embed_dim=embed_dim, 
                                    stride=self.stride, 
                                    dim=embed_dim, 
                                    mode=mode, 
                                    keep_ratio=0.5, 
                                    head=head, 
                                    en_field=en_field, 
                                    if_scale=head_scale,
                                    if_ia=if_ia,
                                    twins=twins)
        else:
            self.LTAM = nn.Identity()

        # propagation branches
        self.resblocks = ResidualBlocksWithInputConv(2*embed_dim, embed_dim, num_blocks, r_scaling=r_scaling)    

        # upsample
        self.fusion = nn.Conv2d(3 * embed_dim, embed_dim, 1, 1, 0, bias=True)
       
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.traj_win = traj_win
        self.en_field = en_field  

    def windows_partition(self, lrs, flows_forward, flows_backward):
        n, t, c, h, w = lrs.size()

        self.s = t//self.traj_win
        lrs = lrs.view(n, self.s, self.traj_win, c, h, w).reshape(-1, self.traj_win, c, h, w)

        forward, backward = [], []
        for i in range(0, t, self.traj_win):
            forward.append(flows_forward[:, i:i+self.traj_win-1, ...])
            backward.append(flows_backward[:, i:i+self.traj_win-1, ...])

        flows_forward = torch.stack(forward, dim=1).reshape(-1, self.traj_win-1, 2, h, w)    
        flows_backward = torch.stack(backward, dim=1).reshape(-1, self.traj_win-1, 2, h, w)  

        return lrs, flows_forward, flows_backward
    
    def windows_merge(self, x):
        n, t, c, h, w = x.shape

        x = x.view(n//self.s, self.s, t, c, h, w).reshape(-1, self.s*t, c, h, w)
        return x

    def forward(self, lrs, flows_forward=None, flows_backward=None):

        if self.if_win_par:    
            lrs, flows_forward, flows_backward = self.windows_partition(lrs, flows_forward, flows_backward)

        n, t, c, h, w = lrs.size()

        outputs = torch.unbind(lrs, dim=1)  
        outputs = list(outputs) 
        keyframe_idx_forward = list(range(0, t, self.keyframe_stride))
        keyframe_idx_backward = list(range(t-1, -1, 0-self.keyframe_stride))


        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = lrs.new_zeros(n, self.embed_dims, h, w)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        location_update = torch.stack([grid_x, grid_y], dim=0).type_as(lrs).expand(n, -1, -1, -1)


        for i in range(t - 1, -1, -1):

            lr_curr_feat = outputs[i]  
      
            if i < t - 1:  
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1), padding_mode='border')
  
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),
                                            padding_mode='border', interpolation="nearest") 

                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                if self.en_field:
                    sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)  
                    sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1) 
                else:
                    sparse_feat_buffer_s2, sparse_feat_buffer_s3 = None, None
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)  # K

                if self.ltam:
                    feat_prop = self.LTAM(lr_curr_feat if not self.learnable else self.q_w(lr_curr_feat),  
                                          index_feat_buffer_s1,  
                                          feat_prop,              
                                          sparse_feat_buffer_s1,
                                          sparse_feat_buffer_s2, sparse_feat_buffer_s3,
                                          location_update,
                                          len(keyframe_idx_backward),
                                          ) 
                else:
                    feat_prop = self.LTAM(feat_prop)    
                    
                if i in keyframe_idx_backward:
                    location_update = torch.cat([location_update, torch.stack([grid_x, grid_y], dim=0).type_as(lrs).expand(n, -1, -1, -1)], dim=1)  # n , 2t , h , w
            
           
            feat_prop = torch.cat([lr_curr_feat, feat_prop], dim=1)      
            feat_prop = self.resblocks(feat_prop)

            if i in keyframe_idx_backward:
                sparse_feat_buffers_s1.append(feat_prop)
                index_feat_buffers_s1.append(lr_curr_feat)

                if self.en_field:
                    sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride), int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride)
                    sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h), int(1.5*w)), kernel_size=(int(1.5*self.stride), int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                    sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2, (h, w))
                    sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                    sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride), int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride)
                 
                    sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h), int(2*w)), kernel_size=(int(2*self.stride), int(2*self.stride)), padding=0, stride=int(2*self.stride))
                   
                    sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3, (h, w))
                    
                    sparse_feat_buffers_s3.append(sparse_feat_prop_s3)
            
            feat_buffers.append(feat_prop)
        
        outputs_back = feat_buffers[::-1]
        del location_update
        del feat_buffers
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1


        fina_out = []
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = torch.zeros_like(feat_prop)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        location_update = torch.stack([grid_x, grid_y], dim=0).type_as(lrs).expand(n, -1, -1, -1)

        for i in range(0, t):
            lr_curr_image = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]
     
            
            if i > 0: 
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')

                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),
                                            padding_mode='border', interpolation="nearest")  

                feat_buffer = torch.stack(feat_buffers, dim=1)
                sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                if self.en_field:
                    sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                    sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                else:
                    sparse_feat_buffer_s2, sparse_feat_buffer_s3 = None, None     
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)

                if self.ltam:
                    feat_prop = self.LTAM(lr_curr_feat, 
                                          index_feat_buffer_s1, 
                                          feat_prop,
                                          # sparse_feat_buffer_s1,
                                          sparse_feat_buffer_s1,
                                          sparse_feat_buffer_s2, sparse_feat_buffer_s3,
                                          location_update,
                                          len(keyframe_idx_forward),
                                          )
                else:
                    feat_prop = self.LTAM(feat_prop)    

                # init the location map
                if i in keyframe_idx_forward:
                    location_update = torch.cat([location_update, torch.stack([grid_x, grid_y], dim=0).type_as(lrs).expand(n, -1, -1, -1)], dim=1)
        
            feat_prop = torch.cat([lr_curr_feat, feat_prop], dim=1)
          

            feat_prop = self.resblocks(feat_prop)

            if i in keyframe_idx_forward:
                sparse_feat_buffers_s1.append(feat_prop)
                index_feat_buffers_s1.append(lr_curr_feat)

                if self.en_field:
                    
                    sparse_feat_prop_s2 = F.unfold(feat_prop, kernel_size=(int(1.5*self.stride), int(1.5*self.stride)), padding=int(0.25*self.stride), stride=self.stride)
                   
                    sparse_feat_prop_s2 = F.fold(sparse_feat_prop_s2, output_size=(int(1.5*h), int(1.5*w)), kernel_size=(int(1.5*self.stride), int(1.5*self.stride)), padding=0, stride=int(1.5*self.stride))
                   
                    sparse_feat_prop_s2 = F.adaptive_avg_pool2d(sparse_feat_prop_s2, (h, w))
                    
                    sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                
                    sparse_feat_prop_s3 = F.unfold(feat_prop, kernel_size=(int(2*self.stride), int(2*self.stride)), padding=int(0.5*self.stride), stride=self.stride)
                    
                    sparse_feat_prop_s3 = F.fold(sparse_feat_prop_s3, output_size=(int(2*h), int(2*w)), kernel_size=(int(2*self.stride), int(2*self.stride)), padding=0, stride=int(2*self.stride))
                    
                    sparse_feat_prop_s3 = F.adaptive_avg_pool2d(sparse_feat_prop_s3, (h, w))
                    
                    sparse_feat_buffers_s3.append(sparse_feat_prop_s3)
            
            feat_buffers.append(feat_prop)
                
            # upsampling given the backward and forward features
            out = torch.cat([outputs_back[i], lr_curr_feat, feat_prop], dim=1)
          
            out = self.lrelu(self.fusion(out))
            fina_out.append(out)
           
        del location_update
        del sparse_feat_buffers_s1
        del sparse_feat_buffers_s2
        del sparse_feat_buffers_s3
        del index_feat_buffers_s1
        
        if self.if_win_par:
            return self.windows_merge(torch.stack(fina_out, dim=1))
        else:
            return torch.stack(fina_out, dim=1)


class LTAM_multi_head(nn.Module):

    def __init__(self, embed_dim, stride=4, dim=96, mode='max', keep_ratio=0.5, head=4, 
                 en_field=True, 
                 if_scale=True,
                 if_ia=False,
                 twins=[2, 2]):
        super().__init__()
        self.stride = stride
        self.dim = dim
      

        self.mode = mode
        self.keep_ratio = keep_ratio

        self.embed_dim = embed_dim
        self.head = head

        head_dim = embed_dim // head
        if if_scale and mode=='wins':
            self.scale = head_dim ** -0.5
        else:
            self.scale = 1.    
        
        self.proj = nn.Linear(embed_dim, embed_dim)

        if en_field:
            self.fusion = nn.Conv2d(3 * embed_dim, embed_dim, 1, 1, 0, bias=True)

        self.ia = if_ia  
          
        if mode == 'wins':
            self.win_h = twins[0]
            self.win_w = twins[1]
            # self.mode = 'wins'
            self.relative_pos_encoding = nn.Parameter(torch.zeros(head, self.win_h*self.win_w, self.win_h*self.win_w))
            trunc_normal_(self.relative_pos_encoding, std=.02)
            decay_value = 1 - 2 ** (-5 - torch.arange(head-1, -1, -1, dtype=torch.float32))
            self.register_buffer('decay_v', decay_value)    

              
    def cal_pe(self, attn_shape):
        b, n, h, q_n, kv_n = attn_shape 

        t = kv_n // q_n

        pe = self.decay_v.unsqueeze(-1)  # head, 1
        pe = torch.repeat_interleave(pe, t, -1)  # head, t
        for i in range(1, t):  
            pe[:, i] *= pe[:, i-1]
        pe = torch.flip(pe, dims=[-1])  
        pe = pe.unsqueeze(-1).expand(-1, -1, q_n).contiguous().view(h, kv_n)     
        pe = pe.unsqueeze(1).expand(-1, q_n, -1) * self.relative_pos_encoding.unsqueeze(2).expand(-1, -1, t, -1).contiguous().view(h, q_n, kv_n)  
        pe = pe.unsqueeze(0).expand(n, -1, -1, -1).unsqueeze(0).expand(b, -1, -1, -1, -1)
        return pe            
    
    
    def forward_max(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=None, sparse_feat_set_s3=None, location_feat=None, video_lens=None):

        n, c, h, w = anchor_feat.size()
       
        t = sparse_feat_set_s1.size(1)  
       
        feat_len = c 
        feat_num = int(h*w)  

     
        grid_flow = location_feat.contiguous().view(n, t, 2, h, w).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)
       
        if not self.ia:
            output_s1 = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s1.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)
            if sparse_feat_set_s2 is not None:
                output_s2 = F.grid_sample(sparse_feat_set_s2.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s2.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)
            if sparse_feat_set_s3 is not None:
                output_s3 = F.grid_sample(sparse_feat_set_s3.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s3.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)    

            # (nt) * (c*4*4) * (h//4) * (w//4)
            index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1, c, h, w),
                                        grid_flow.contiguous().view(-1, h, w, 2).to(dtype=index_feat_set_s1.dtype),
                                        mode='nearest', padding_mode='zeros', align_corners=True)
        else:
            s1, s2, s3, ind1 = [], [], [], []
            for i in range(t):
                output_s1 = self.resampling_ia(sparse_feat_set_s1[:, i, ...].contiguous().view(-1, c, h, w),
                                               curr_feat,
                                               grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s1.dtype),
                                               if_specific=True)    
                if sparse_feat_set_s2 is not None:
                    output_s2 = self.resampling_ia(sparse_feat_set_s2[:, i, ...].contiguous().view(-1, c, h, w),
                                                   curr_feat,
                                                   grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s2.dtype),
                                                   if_specific=True)
                    s2.append(output_s2)
                if sparse_feat_set_s3 is not None:
                    output_s3 = self.resampling_ia(sparse_feat_set_s3[:, i, ...].contiguous().view(-1, c, h, w),
                                                   curr_feat,
                                                   grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s3.dtype),
                                                   if_specific=True)
                    s3.append(output_s3)    
                index_output_s1 = self.resampling_ia(index_feat_set_s1[:, i, ...].contiguous().view(-1, c, h, w),
                                                     curr_feat,
                                                     grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=index_feat_set_s1.dtype),
                                                     if_specific=True)   
                s1.append(output_s1)  
                ind1.append(index_output_s1)
            # (nt, c, h, w)    
            output_s1 = torch.cat(s1, dim=0)
            if sparse_feat_set_s2 is not None:
                output_s2 = torch.cat(s2, dim=0)
                del s2
            if sparse_feat_set_s3 is not None:
                output_s3 = torch.cat(s3, dim=0)
                del s3
            index_output_s1 = torch.cat(ind1, dim=0)
            del s1        
            del ind1

     
        curr_feat = curr_feat.reshape(-1, c, h*w).permute(0, 2, 1).contiguous()
        curr_feat = F.normalize(curr_feat, dim=-1).unsqueeze(3)  # n * (h//4*w//4) * (c*4*4) * 1
        curr_feat = curr_feat.reshape(-1, h*w, self.head, c//self.head, 1)

      
        index_output_s1 = index_output_s1.reshape(-1, t, c, h*w).permute(0, 3, 1, 2).contiguous()
        index_output_s1 = F.normalize(index_output_s1, dim=-1)  # n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.reshape(-1, h*w, t, self.head, c//self.head).permute(0, 1, 3, 2, 4).contiguous()

        ##############################################################################################################

        matrix_index = torch.matmul(index_output_s1, curr_feat*self.scale).squeeze(4)  # n * (h*w) * head * t
        matrix_index = matrix_index.view(n, feat_num, self.head, t)  # n * (h//4*w//4) * head * t
      
        corr_soft, corr_index = torch.max(matrix_index, dim=3)  # n * (h//4*w//4) * head
       
        corr_soft = corr_soft.unsqueeze(3).expand(-1, -1, -1, feat_len//self.head).reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
       

        #############################################################################################################

        # Aggr
        
        output_s1 = output_s1.contiguous().view(n*t, c, h, w)
       
        mid_r = corr_index.view(n, 1, feat_num, self.head, 1).expand(-1, -1, -1, self.head, c//self.head).reshape(-1, 1, feat_num, feat_len).permute(0, 1, 3, 2).contiguous()
        
        out = torch.gather(output_s1.contiguous().view(n, t, feat_len, feat_num), 1, mid_r)
        # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
        out = out.squeeze(1).reshape(n, c, h, w)

        # x6x8
        if sparse_feat_set_s2 is not None and sparse_feat_set_s3 is not None:
            output_s2 = output_s2.contiguous().view(n*t, c, h, w)
            # mid_r = corr_index.view(n, 1, feat_num, self.head, 1).expand(-1, -1, -1, self.head, c//self.head).reshape(-1, 1, feat_num, feat_len).permute(0, 1, 3, 2).contiguous()
            out_2 = torch.gather(output_s2.contiguous().view(n, t, feat_len, feat_num), 1, mid_r)
            out_2 = out_2.squeeze(1).reshape(n, c, h, w)

            output_s3 = output_s3.contiguous().view(n*t, c, h, w)
            # mid_r = corr_index.view(n, 1, feat_num, self.head, 1).expand(-1, -1, -1, self.head, c//self.head).reshape(-1, 1, feat_num, feat_len).permute(0, 1, 3, 2).contiguous()
            out_3 = torch.gather(output_s3.contiguous().view(n, t, feat_len, feat_num), 1, mid_r)
            out_3 = out_3.squeeze(1).reshape(n, c, h, w)

            out = self.fusion(torch.cat([out, out_2, out_3], dim=1))
        
        # print(f'out1 is {out.shape}')  # out1 is torch.Size([2, 64, 64, 64])
        out = out * corr_soft
        out = self.proj(out.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        # print(out.shape)
        out += anchor_feat  
        return out
    
    def forward_wins(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=None, sparse_feat_set_s3=None, location_feat=None, video_lens=None):
      
        n, c, h, w = anchor_feat.size()
      
        t = sparse_feat_set_s1.size(1)  

        
        feat_len = c 
        feat_num = int(h*w)  

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n, t, 2, h, w).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)
       
        if not self.ia:
            output_s1 = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s1.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)
            if sparse_feat_set_s2 is not None:
                output_s2 = F.grid_sample(sparse_feat_set_s2.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s2.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)
            if sparse_feat_set_s3 is not None:
                output_s3 = F.grid_sample(sparse_feat_set_s3.contiguous().view(-1, c, h, w),
                                  grid_flow.contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s3.dtype),
                                  mode='nearest', padding_mode='zeros', align_corners=True)    

            # (nt) * (c*4*4) * (h//4) * (w//4)
            index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1, c, h, w),
                                        grid_flow.contiguous().view(-1, h, w, 2).to(dtype=index_feat_set_s1.dtype),
                                        mode='nearest', padding_mode='zeros', align_corners=True)
        else:
            s1, s2, s3, ind1 = [], [], [], []
            for i in range(t):
                output_s1 = self.resampling_ia(sparse_feat_set_s1[:, i, ...].contiguous().view(-1, c, h, w),
                                               curr_feat,
                                               grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s1.dtype),
                                               if_specific=True)    
                if sparse_feat_set_s2 is not None:
                    output_s2 = self.resampling_ia(sparse_feat_set_s2[:, i, ...].contiguous().view(-1, c, h, w),
                                                   curr_feat,
                                                   grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s2.dtype),
                                                   if_specific=True)
                    s2.append(output_s2)
                if sparse_feat_set_s3 is not None:
                    output_s3 = self.resampling_ia(sparse_feat_set_s3[:, i, ...].contiguous().view(-1, c, h, w),
                                                   curr_feat,
                                                   grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=sparse_feat_set_s3.dtype),
                                                   if_specific=True)
                    s3.append(output_s3)    
                index_output_s1 = self.resampling_ia(index_feat_set_s1[:, i, ...].contiguous().view(-1, c, h, w),
                                                     curr_feat,
                                                     grid_flow[:, i, ...].contiguous().view(-1, h, w, 2).to(dtype=index_feat_set_s1.dtype),
                                                     if_specific=True)   
                s1.append(output_s1)  
                ind1.append(index_output_s1)
            # (nt, c, h, w)    
            output_s1 = torch.cat(s1, dim=0)
            if sparse_feat_set_s2 is not None:
                output_s2 = torch.cat(s2, dim=0)
                del s2
            if sparse_feat_set_s3 is not None:
                output_s3 = torch.cat(s3, dim=0)
                del s3
            index_output_s1 = torch.cat(ind1, dim=0)
            del s1        
            del ind1

    
        curr_feat = curr_feat.reshape(-1, c, h*w).permute(0, 2, 1).contiguous()
        curr_feat = F.normalize(curr_feat, dim=-1).unsqueeze(3)  # n * (h//4*w//4) * (c*4*4) * 1
        curr_feat = curr_feat.reshape(-1, h*w, self.head, c//self.head, 1)  # b, hw, head, d, 1

        
        index_output_s1 = index_output_s1.reshape(-1, t, c, h*w).permute(0, 3, 1, 2).contiguous()
        index_output_s1 = F.normalize(index_output_s1, dim=-1)  # n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.reshape(-1, h*w, t, self.head, c//self.head).permute(0, 1, 3, 2, 4).contiguous()  # b, hw, head, t, d


        ##############################################################################################################
        # wins
        curr_feat = curr_feat.view(-1, h, w, self.head, c//self.head, 1)
        index_output_s1 = index_output_s1.view(-1, h, w, self.head, t, c//self.head)
        curr_feat = rearrange(curr_feat, 'b (h wh) (w ww) he d o -> b (h w) he d (o wh ww)', wh=self.win_h, ww=self.win_w).contiguous()
        index_output_s1 = rearrange(index_output_s1, 'b (h wh) (w ww) he t d -> b (h w) he (t wh ww) d', wh=self.win_h, ww=self.win_w).contiguous()
        output_s1 = rearrange(output_s1, '(b t) (he d) (h wh) (w ww) -> b (h w) he (t wh ww) d', t=t, he=self.head, wh=self.win_h, ww=self.win_w).contiguous()

       
        matrix_index = torch.matmul(index_output_s1, curr_feat*self.scale)  # n * (h*w) * head * (t*wh*ww) * (wh*ww)
        # matrix_index = matrix_index.view(n, feat_num, self.head, t)  # n * (h//4*w//4) * head * t
        matrix_index = rearrange(matrix_index, 'b hw he t1 t2 -> b hw he t2 t1').contiguous()
       
        rpe = self.cal_pe(attn_shape=matrix_index.shape)
        matrix_index = (matrix_index + rpe).softmax(dim=-1)
    

        #############################################################################################################


        out = torch.einsum('b f h m n, b f h n d -> b f h m d', matrix_index, output_s1)
        out = rearrange(out, 'b (h w) he (wh ww) d -> b (h wh) (w ww) (he d)', h=h//self.win_h, wh=self.win_h, ww=self.win_w).contiguous()  # b h w c

        # x6x8
        if sparse_feat_set_s2 is not None and sparse_feat_set_s3 is not None:
            output_s2 = output_s2.contiguous().view(n*t, c, h, w)
            # mid_r = corr_index.view(n, 1, feat_num, self.head, 1).expand(-1, -1, -1, self.head, c//self.head).reshape(-1, 1, feat_num, feat_len).permute(0, 1, 3, 2).contiguous()
            out_2 = torch.gather(output_s2.contiguous().view(n, t, feat_len, feat_num), 1, mid_r)
            out_2 = out_2.squeeze(1).reshape(n, c, h, w)

            output_s3 = output_s3.contiguous().view(n*t, c, h, w)
            # mid_r = corr_index.view(n, 1, feat_num, self.head, 1).expand(-1, -1, -1, self.head, c//self.head).reshape(-1, 1, feat_num, feat_len).permute(0, 1, 3, 2).contiguous()
            out_3 = torch.gather(output_s3.contiguous().view(n, t, feat_len, feat_num), 1, mid_r)
            out_3 = out_3.squeeze(1).reshape(n, c, h, w)

            out = self.fusion(torch.cat([out, out_2, out_3], dim=1))
        
        # print(f'out1 is {out.shape}')  # out1 is torch.Size([2, 64, 64, 64])
        # out = out * corr_soft
        out = self.proj(out).permute(0, 3, 1, 2).contiguous()
        # print(out.shape)
        out += anchor_feat  
        return out


    def forward(self, curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=None, sparse_feat_set_s3=None, location_feat=None, video_lens=None, config_amp=False):
        if self.mode == 'max':  
            res = self.forward_max(curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=sparse_feat_set_s2, sparse_feat_set_s3=sparse_feat_set_s3, location_feat=location_feat)
        elif self.mode == 'wins': 
            res = self.forward_wins(curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=sparse_feat_set_s2, sparse_feat_set_s3=sparse_feat_set_s3, location_feat=location_feat)
        elif self.mode == 'average':  
            res = self.forward_average(curr_feat, index_feat_set_s1, anchor_feat, sparse_feat_set_s1, sparse_feat_set_s2=sparse_feat_set_s2, sparse_feat_set_s3=sparse_feat_set_s3, location_feat=location_feat, video_lens=None, config_amp=None)
        else:
            raise Exception('invalid mode')        
        return res
    