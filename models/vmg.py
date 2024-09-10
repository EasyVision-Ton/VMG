import os
import numpy as np
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from models.layers import (make_layer, ResidualBlock_noBN, InputProj, UpdownkeepSampling)
from timm.models.layers import trunc_normal_
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger
from mmcv.cnn import ConvModule
from torch.cuda.amp import autocast as autocast
from models.function import *
from fractions import Fraction
from thop import profile


class SPyNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class VMG(nn.Module):
    def __init__(self, in_chans=3, embed_dim=[112, 224, 224, 448, 448, 224, 224, 112], 
                 depths=[8, 8, 8, 8, 8, 8, 8, 8], 
                 num_heads=[2, 4, 8, 16, 16, 8, 4, 2], 
                 num_frames=7,
                 window_sizes=[(4,4), (4,4), (4,4), (4,4), (4,4), (4,4), (4,4), (4,4)], 
                 mdsc=True, if_concat=False,
                 mlp_ratio=4., n_groups=1,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 back_RBs=0,
                 spynet_pretrained=None,
                 image_size=[64, 112],
                 is_train=True,
                 if_print=False,
                 ltam=True,
                 traj_win=[16, None, None, None], traj_keyframes_n=[3, None, None, None], traj_heads=[4, None, None, None],
                 temporal_type=[False, None, None, None], temporal_empty=True,
                 traj_res_n=[1, 0, 0, 0, 0, 0, 1],
                 deform_groups=[8, 16, 16, 32], max_residual_scale=[1, 2, 2, 4],
                 spatial_type=[False, False, False, False],
                 flow_smooth=True, smooth_region_range=4,
                 retention_decay=True,
                 non_linear=True,
                 gating=True, symm=True, symm_act=nn.Tanh,
                 relu_scale=True, relu_scale_norm=False,
                 ffn_type='vanilla',
                 mixer_type=['mbconv','mbconv', 'mlps', 'mlps'], mixer_n=[2, 3, None, None],
                 r_scaling=1.,
                 chunk_ratios=[1/4, 1/4, 3/16, 1/8],
                 traj_mode='wins', twins=[2, 2], traj_scale=True, traj_refine=None, 
                 m_scaling=1.,
                 if_local_fuse=False,
                 channel_mixer='vanilla'):
        super().__init__()

        self.num_layers = len(depths)  # 7
        self.num_enc_layers = self.num_layers // 2 + 1  # 4
        self.num_dec_layers = self.num_layers // 2  # 3
        self.scale = 2 ** (self.num_enc_layers - 1)  
        dec_depths = depths[self.num_enc_layers:]
        self.embed_dim = embed_dim  
        self.patch_norm = patch_norm
        self.num_in_frames = num_frames
        self.num_out_frames = num_frames
        self.is_train = is_train
        self.if_print = if_print

        self.init_H, self.init_W = image_size

        self.pos_drop = nn.Dropout(p=drop_rate)

        if spynet_pretrained is not None:
            self.spynet = SPyNet(spynet_pretrained) 
        else:
            self.spynet = None    

        # Stochastic depth 
        enc_dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        # dec_dpr = enc_dpr[::-1]
        dec_dpr_ = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[self.num_enc_layers:]))] 
        dec_dpr = dec_dpr_[::-1]
        if not is_train:
            enc_dpr = [0.] * len(enc_dpr)
            dec_dpr = [0.] * len(dec_dpr)

        self.chunk_ratio = [float(Fraction(r)) for r in chunk_ratios]    
        self.chunk_h = [int(self.init_H * x) for x in self.chunk_ratio]
        self.chunk_w = [int(self.init_W * x) for x in self.chunk_ratio]    

        self.tps_size = 4
        self.n_nonkeyframes = traj_keyframes_n 
        self.scale_r = max_residual_scale
        self.deform_groups = deform_groups 
        self.traj_win = traj_win  
        # control temporal
        self.aligned = temporal_type  
        self.empty_aligned = temporal_empty 
        self.traj_resblock_num = traj_res_n
        self.if_traj = spatial_type  

        self.local_fuse = if_local_fuse
        if if_local_fuse:
            self.local_cnn = nn.Conv2d(embed_dim[0], embed_dim[0], 3, 1, 1)

        # TAU
        self.k_s = [21, 15, 15, 9]

        if symm_act == 'tanh':
            symm_act = nn.Tanh
        elif symm_act == 'sigmoid':
            symm_act = sigmoid_symm
        elif symm_act == 'relu':
            symm_act = nn.ReLU
        elif symm_act == 'gelu':
            symm_act = nn.GELU  
        elif symm_act == 'swish':
            symm_act = nn.SiLU               

        self.input_proj = InputProj(in_channels=in_chans, embed_dim=embed_dim[0],
                                    kernel_size=3, stride=1, act_layer=nn.LeakyReLU)

        self.encoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_enc_layers):
            encoder_layer = Mlp_encoder(
                    embed_dim=embed_dim[i_layer], 
                    depth=depths[i_layer],   
                    segm=num_heads[i_layer],
                    num_frames=num_frames,
                    chunk_dim=num_frames, block_len=4, grid_len=4,
                    chunk_dim_h=self.chunk_h[i_layer], chunk_dim_w=self.chunk_w[i_layer],
                    mlp_ratio=mlp_ratio, n_groups=n_groups,
                    qkv_bias=qkv_bias, 
                    attn_drop=attn_drop_rate, proj_drop=0.,
                    drop_path=enc_dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    win_size=self.tps_size,
                    window_size=window_sizes[i_layer],
                    if_traj=self.if_traj[i_layer],
                    n_nonkeyframes=self.n_nonkeyframes[i_layer],
                    aligned=self.aligned[i_layer], empty_aligned=self.empty_aligned, traj_r_n=self.traj_resblock_num[i_layer],   
                    deformable_groups=self.deform_groups[i_layer],
                    max_residue_magnitude=10. / self.scale_r[i_layer],
                    is_train=self.is_train,
                    unfold_stride=0 if i_layer==0 else 0,
                    unfold_conv=False,
                    ltam=ltam,
                    traj_win=self.traj_win[i_layer], traj_heads=traj_heads[i_layer],
                    k_s=self.k_s[i_layer],
                    if_smooth=flow_smooth, region_range=smooth_region_range,
                    if_decay=retention_decay,
                    non_linear=non_linear,
                    gating=gating, symm=symm, symm_act=symm_act,
                    relu_scale=relu_scale, relu_scale_norm=relu_scale_norm,
                    ffn_type=ffn_type,
                    mixer_type=mixer_type[i_layer], mixer_n=mixer_n[i_layer],
                    r_scaling=r_scaling, traj_mode=traj_mode, twins=twins, traj_scale=traj_scale, traj_refine=traj_refine,
                    m_scaling=m_scaling,
                    if_local_fuse=if_local_fuse,
                    channel_mixer=channel_mixer
            )
            self.encoder_layers.append(encoder_layer)
            if i_layer != self.num_enc_layers - 1:
                downsample = UpdownkeepSampling(embed_dim[i_layer], embed_dim[i_layer + 1], mode='down')
                self.downsample.append(downsample)
            else:
                upsample = UpdownkeepSampling(embed_dim[i_layer], embed_dim[i_layer + 1], mode='up')
                self.upsample.append(upsample)    

        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = Mlp_encoder(
                    embed_dim=embed_dim[i_layer + self.num_enc_layers], 
                    depth=depths[i_layer + self.num_enc_layers], 
                    segm=num_heads[i_layer + self.num_enc_layers],
                    num_frames=num_frames,
                    chunk_dim=num_frames, block_len=4, grid_len=4,
                    chunk_dim_h=self.chunk_h[-i_layer-2], chunk_dim_w=self.chunk_w[-i_layer-2],
                    mlp_ratio=mlp_ratio, n_groups=n_groups,
                    qkv_bias=qkv_bias, 
                    attn_drop=attn_drop_rate, proj_drop=0.,
                    drop_path=dec_dpr[sum(dec_depths[:i_layer]):sum(dec_depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    win_size=self.tps_size,
                    window_size=window_sizes[i_layer + self.num_enc_layers],
                    if_traj=self.if_traj[-i_layer-2],
                    n_nonkeyframes=self.n_nonkeyframes[-i_layer-2],
                    aligned=self.aligned[-i_layer-2], empty_aligned=self.empty_aligned, traj_r_n=self.traj_resblock_num[i_layer + self.num_enc_layers],
                    deformable_groups=self.deform_groups[-i_layer-2],
                    max_residue_magnitude=10. / self.scale_r[-i_layer-2],
                    is_train=self.is_train,
                    unfold_stride=0 if i_layer==self.num_dec_layers-1 else 0,
                    unfold_conv=False,
                    ltam=ltam,
                    traj_win=self.traj_win[-i_layer-2], traj_heads=traj_heads[-i_layer-2],
                    k_s=self.k_s[-i_layer-2],
                    if_smooth=flow_smooth, region_range=smooth_region_range,
                    if_decay=retention_decay,
                    non_linear=non_linear,
                    gating=gating, symm=symm, symm_act=symm_act,
                    relu_scale=relu_scale, relu_scale_norm=relu_scale_norm,
                    ffn_type=ffn_type,
                    mixer_type=mixer_type[-i_layer-2], mixer_n=mixer_n[-i_layer-2],
                    r_scaling=r_scaling, traj_mode=traj_mode, twins=twins, traj_scale=traj_scale, traj_refine=traj_refine,
                    m_scaling=m_scaling,
                    if_local_fuse=if_local_fuse,
                    channel_mixer=channel_mixer
            )
            self.decoder_layers.append(decoder_layer)
            if i_layer != self.num_dec_layers - 1:
                upsample = UpdownkeepSampling(embed_dim[i_layer + self.num_enc_layers], embed_dim[i_layer + self.num_enc_layers + 1], mode='up')
                self.upsample.append(upsample)

        # Reconstruction block
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=embed_dim[-1])  
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        self.back_rbs = back_RBs
        # Upsampling
        self.upconv1 = nn.Conv2d(embed_dim[-1], embed_dim[-1] * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(embed_dim[-1], 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.mdsc = mdsc  # Multi-downsampling short-connection
        if mdsc:
            self.sc_64_16 = nn.Sequential(
                # nn.AdaptiveMaxPool2d((16, 16)),
                nn.Conv2d(embed_dim[0], embed_dim[2], 1, 1, 0),
                nn.GroupNorm(1, embed_dim[2]),  
                nn.ReLU()  
                )
            self.sc_32_8 = nn.Sequential(
                # nn.AdaptiveMaxPool2d((8, 8)),
                nn.Conv2d(embed_dim[1], embed_dim[3], 1, 1, 0),
                nn.GroupNorm(1, embed_dim[3]),
                nn.ReLU()
                )

        self.if_concat = if_concat
        if if_concat:
            self.reduce0 = nn.Conv2d(2*embed_dim[-1], embed_dim[-1], 1, 1, 0)
            self.reduce1 = nn.Conv2d(2*embed_dim[-2], embed_dim[-2], 1, 1, 0)
            self.reduce2 = nn.Conv2d(2*embed_dim[-3], embed_dim[-3], 1, 1, 0)            
        
        self.mlp_wd_param = []
        for name, p in self.named_parameters():
            if '.mlp_blocks.' in name:  
                self.mlp_wd_param.append(p)
        print(f'The length of wd is {len(self.mlp_wd_param)}.')  
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # print(f'm is {m}')
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # self.mlp_wd_param.append(m)  # weight decay    
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_frames_mirror(self, lrs):

        self.frames_mirror = False
        if lrs.size(1) % 2 == 0:
            lrs_0, lrs_1 = torch.chunk(lrs, 2, dim=1)
            if torch.linalg.norm(lrs_0 - lrs_1.flip(1)) == 0:
                self.frames_mirror = True
        

    def compute_flow(self, lrs):
        B, T, C, H, W = lrs.shape

        lrs_list = [F.adaptive_avg_pool2d(lrs.reshape(B*T, C, H, W), (H//(2**i), W//(2**i))) for i in range(self.num_enc_layers)]
        resolution = [(H//(2**i), W//(2**i)) for i in range(self.num_enc_layers)]
        forward_list, backward_list = [], []

        for i in range(len(lrs_list)):
            H, W = resolution[i]
            src_forward = lrs_list[i].reshape(B, T, C, H, W)[:, :-1, ...].contiguous()
            src_backward = lrs_list[i].reshape(B, T, C, H, W)[:, 1:, ...].contiguous()

            flow_forward, flow_backward = self.compute_flow_forward(src_forward, src_backward)
            forward_list.append(flow_forward)
            backward_list.append(flow_backward)

        assert len(forward_list) == self.num_enc_layers
        # assert len(forward_list) == 2 
        return forward_list, backward_list

    def compute_flow_forward(self, src_forward, src_backward):
        B, T, C, H, W = src_forward.shape

        forward_flow = self.spynet(src_backward.view(-1, C, H, W), src_forward.view(-1, C, H, W)).view(B, T, 2, H, W)
        if not self.frames_mirror:
            backward_flow = self.spynet(src_forward.view(-1, C, H, W), src_backward.view(-1, C, H, W)).view(B, T, 2, H, W)
        else:
            backward_flow = forward_flow.flip(1)  # reverse the order.    

        return forward_flow, backward_flow
    
    def forward_features_multi_stages(self, x, flow_forwards, flow_backwards):

        if not self.mdsc:
            x1 = self.encoder_layers[0](x, flow_forwards[0], flow_backwards[0])  # B C T H W
            # x1 = self.align_t_down.layers[0](x1)
            x1_ = self.downsample[0](x1)

            x2 = self.encoder_layers[1](x1_, flow_forwards[1], flow_backwards[1])
            # x2 = self.align_t_down.layers[1](x2)
            x2_ = self.downsample[1](x2)
        
            x3 = self.encoder_layers[2](x2_, flow_forwards[2], flow_backwards[2])
            # x3 = self.align_t_down.layers[2](x3)
            x3_ = self.downsample[2](x3)

            x4 = self.encoder_layers[3](x3_, flow_forwards[3], flow_backwards[3])
            # x4 = self.align_t_down.layers[3](x4)
            x4_ = self.upsample[0](x4)

            if not self.if_concat:
                x5 = self.decoder_layers[0](x4_, flow_forwards[2], flow_backwards[2])
                # x5 = self.align_t_up.layers[2](x5)
                x5_ = self.upsample[1](x5 + x3)

                x6 = self.decoder_layers[1](x5_, flow_forwards[1], flow_backwards[1])
                # x6 = self.align_t_up.layers[1](x6)
                x6_ = self.upsample[2](x6 + x2)

                x7 = self.decoder_layers[2](x6_, flow_forwards[0], flow_backwards[0])
                # x7 = self.align_t_up.layers[0](x7)
                y = x7 + x1
            else:
                x5 = self.decoder_layers[0](x4_, flow_forwards[2], flow_backwards[2])
                # x5 = self.align_t_up.layers[2](x5)
                _, _, C_c, H_c, W_c = x5.shape
                x5_x3 = self.reduce2(torch.cat([x3, x5], 2).view(-1, 2*C_c, H_c, W_c)).view(B, T, C_c, H_c, W_c)
                x5_ = self.upsample[1](x5_x3)

                x6 = self.decoder_layers[1](x5_, flow_forwards[1], flow_backwards[1])
                # x6 = self.align_t_up.layers[1](x6)
                _, _, C_b, H_b, W_b = x6.shape
                x6_x2 = self.reduce1(torch.cat([x2, x6], 2).view(-1, 2*C_b, H_b, W_b)).view(B, T, C_b, H_b, W_b)
                x6_ = self.upsample[2](x6_x2)

                x7 = self.decoder_layers[2](x6_, flow_forwards[0], flow_backwards[0])
                # x7 = self.align_t_up.layers[0](x7)
                _, _, C_a, H_a, W_a = x7.shape
                x7_x1 = self.reduce0(torch.cat([x1, x7], 2).view(-1, 2*C_a, H_a, W_a)).view(B, T, C_a, H_a, W_a)
                y = x7_x1  
        else:
            x1 = self.encoder_layers[0](x, flow_forwards[0], flow_backwards[0])  # B C T H W
            # x1 = self.align_t_down.layers[0](x1)
            B, T, C1, H1, W1 = x1.shape
            x1_3 = self.sc_64_16(F.adaptive_max_pool2d(x1.view(-1, C1, H1, W1), (H1//4, W1//4))).view(B, T, self.embed_dim[2], H1//4, W1//4)
            x1_ = self.downsample[0](x1)

            x2 = self.encoder_layers[1](x1_, flow_forwards[1], flow_backwards[1])
            # x2 = self.align_t_down.layers[1](x2)
            _, _, C2, H2, W2 = x2.shape
            x2_4 = self.sc_32_8(F.adaptive_max_pool2d(x2.view(-1, C2, H2, W2), (H2//4, W2//4))).view(B, T, self.embed_dim[3], H2//4, W2//4)
            x2_ = self.downsample[1](x2)
        
            x3 = self.encoder_layers[2](x2_, flow_forwards[2], flow_backwards[2])
            # x3 = self.align_t_down.layers[2](x3)
            x3_ = self.downsample[2](x3 + x1_3)

            x4 = self.encoder_layers[3](x3_, flow_forwards[3], flow_backwards[3])
            # x4 = self.align_t_down.layers[3](x4)
            x4_ = self.upsample[0](x4 + x2_4)

            if not self.if_concat:
                x5 = self.decoder_layers[0](x4_, flow_forwards[2], flow_backwards[2])
                # x5 = self.align_t_up.layers[2](x5)
                x5_ = self.upsample[1](x5 + x3)

                x6 = self.decoder_layers[1](x5_, flow_forwards[1], flow_backwards[1])
                # x6 = self.align_t_up.layers[1](x6)
                x6_ = self.upsample[2](x6 + x2)

                x7 = self.decoder_layers[2](x6_, flow_forwards[0], flow_backwards[0])
                # x7 = self.align_t_up.layers[0](x7)
                y = x7 + x1    
            else:
                x5 = self.decoder_layers[0](x4_, flow_forwards[2], flow_backwards[2])
                # x5 = self.align_t_up.layers[2](x5)
                _, _, C_c, H_c, W_c = x5.shape
                x5_x3 = self.reduce2(torch.cat([x3, x5], 2).view(-1, 2*C_c, H_c, W_c)).view(B, T, C_c, H_c, W_c)
                x5_ = self.upsample[1](x5_x3)

                x6 = self.decoder_layers[1](x5_, flow_forwards[1], flow_backwards[1])
                # x6 = self.align_t_up.layers[1](x6)
                _, _, C_b, H_b, W_b = x6.shape
                x6_x2 = self.reduce1(torch.cat([x2, x6], 2).view(-1, 2*C_b, H_b, W_b)).view(B, T, C_b, H_b, W_b)
                x6_ = self.upsample[2](x6_x2)

                x7 = self.decoder_layers[2](x6_, flow_forwards[0], flow_backwards[0])
                # x7 = self.align_t_up.layers[0](x7)
                _, _, C_a, H_a, W_a = x7.shape
                x7_x1 = self.reduce0(torch.cat([x1, x7], 2).view(-1, 2*C_a, H_a, W_a)).view(B, T, C_a, H_a, W_a)
                y = x7_x1

        return y

    def forward_features_few_stages(self, x, flow_forwards, flow_backwards):

        # Encoder
        x1 = self.encoder_layers[0](x, flow_forwards[0], flow_backwards[0])
        x1_ = self.downsample[0](x1)

        # Bottleneck
        x2 = self.encoder_layers[1](x1_, flow_forwards[1], flow_backwards[1])
        x2_ = self.upsample[0](x2)

        # Decoder
        x3 = self.decoder_layers[0](x2_, flow_forwards[0], flow_backwards[0])

        return x3 + x1


    def forward(self, x, flow_pretrained=None, config_amp=None):
        B, D, C, H, W = x.size()  # B D 3 H W       D input video frames

        assert H>=64 and W>=64, "The height and width must larger than 64."

        self.check_frames_mirror(lrs=x)

        x = x.permute(0, 2, 1, 3, 4).contiguous()  # B C D H W
        upsample_x = F.interpolate(x, (D, H*4, W*4), mode='trilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # B D C H W

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        # Dp = int(np.ceil(D / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H, 0, 0), mode='replicate')

        if self.spynet is not None:
            flow_forwards, flow_backwards = self.compute_flow(x)    
        else:
            flow_forwards = [None] * self.num_enc_layers
            flow_backwards = [None] * self.num_enc_layers    

        x = self.input_proj(x) # B, D, 3, H, W -> B, D, C, H, W

        if self.num_layers > 3:
            y = self.forward_features_multi_stages(x, flow_forwards, flow_backwards)
            # print(f'Performing multi stages for {self.num_layers} stages.')
        else:
            y = self.forward_features_few_stages(x, flow_forwards, flow_backwards)
            # print(f'Performing few stages for {self.num_layers} stages.')

        if self.local_fuse:
            y = x + self.local_cnn(y.view(B*D, -1, Hp, Wp)).view(B, D, -1, Hp, Wp)

        y = y[:, :, :, :H, :W].contiguous()
        # Super-resolution
        B, D, C, H, W = y.size()
        y = y.view(B*D, C, H, W)

        if self.back_rbs > 0:
            out = self.recon_trunk(y)
        else:
            out = y    
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, H, W = out.size()  # B*D, 3, H, W

        outs = out.view(B, self.num_out_frames, -1, H, W)
        outs = outs + upsample_x.permute(0, 2, 1, 3, 4).contiguous()
        return outs


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


class sigmoid_symm(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()

    def forward(self, x):  # to be symmetric
        x = self.act(x) - 0.5

        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    upscale = 4
    num_frames = 16

    inputs = torch.randn((1, num_frames, 3, 180, 320)).to(device)
    model = VMG(embed_dim=[144, 224, 144],  # [144, 144, 144]
                 depths=[4, 4, 4],
                 num_frames=num_frames,
                 mdsc=False,
                 if_concat=False,
                 mlp_ratio=2, n_groups=1,
                 spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth',
                 image_size=[64, 64],
                 is_train=False,
                 traj_win=[num_frames, None],
                 traj_keyframes_n=[3, None],
                 traj_heads=[4, None],
                 temporal_type=[False, None],
                 temporal_empty=True,
                 traj_res_n=[15, 0, 15],
                 spatial_type=[False, False],
                 flow_smooth=True,
                 smooth_region_range=4,
                 retention_decay=True,
                 non_linear=True,
                 gating=True, symm=True, symm_act='tanh',
                 relu_scale=True, relu_scale_norm=False,
                 ffn_type='ffn_cnn',
                 mixer_type=['mlps', 'mlps'],
                 mixer_n=[None, None],
                 r_scaling=0.1,
                 chunk_ratios=[1/8, 1/4],
                 traj_mode='wins', twins=[2, 2],
                 traj_scale=True,
                 traj_refine=None,
                 m_scaling=1.,
                 if_local_fuse=True
                 ).to(device)

    print('{:>16s} : {:<.4f} [M]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 6))


    flops, params = profile(model, inputs=(inputs, ))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))
