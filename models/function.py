import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from itertools import repeat
from functools import reduce
from timm.models.layers import DropPath, trunc_normal_
from operator import mul
import torchvision, math
from torch.nn.modules.utils import _pair, _single
from models.CNNs import *
from models.trajectory import Trajectory_multi_head, ResidualBlocksWithInputConv
from models.swin_3d import DecoderLayer
from models.layers import DecoderLayer as DL
from models.norm_store import RMSNorm as rnorm


# symmetric Sigmoid class
class sigmoid_symm(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sigmoid()

    def forward(self, x):  # to be symmetric
        x = self.act(x) - 0.5

        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_cnn(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=4, n_groups=1):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_channels=in_features,
                                         out_channels=self.hidden_features,
                                         kernel_size=3,
                                         stride=1, 
                                         padding=1,
                                         groups=n_groups)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_cnn_gating(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=2):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_channels=in_features,
                                         out_channels=self.hidden_features,
                                         kernel_size=3,
                                         stride=1, padding=1)
        # self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.sym_act = nn.Tanh()

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x0 = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)

        # main_path = x0
        seco_path = self.sym_act(x0)
        x = x0 * seco_path

        x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_cnn_gating_v2(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=2):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_channels=in_features,
                                         out_channels=self.hidden_features,
                                         kernel_size=3,
                                         stride=1, padding=1)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

        self.sym_act = nn.Tanh()

    def forward(self, x):
        B, T, H, W, C = x.shape
        x_short = x.clone()
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)

        # main_path = x0
        # seco_path = self.sym_act(x0)
        # x = x0 * seco_path

        x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc2(x)
        # x = self.drop(x)

        seco_path = self.sym_act(x)
        x = x + x_short
        x = x * seco_path

        return x


class Mlp_cnn_gating_v3(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=2):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_channels=in_features,
                                         out_channels=self.hidden_features,
                                         kernel_size=3,
                                         stride=1, padding=1)
        self.act = act_layer()  # add gelu---20231208
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.sym_act = nn.Tanh()

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x0 = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)

        # main_path = x0
        seco_path = self.sym_act(x0)
        x0 = self.act(x0)
        x = x0 * seco_path

        x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchShift2D(nn.Module):

    def __init__(self, inv=False, win=3):
        super().__init__()
        self.inv = inv

        self.windows = tuple(repeat(win, 2))
        # self.small_windows = tuple(repeat(wins, 2))
        self.num_elements_win = reduce(mul, self.windows)
        # self.num_elements_small_win = reduce(mul, self.small_windows)
        self.shift_content = torch.tensor([[(1, 1), (1, 0), (1, -1)], [(0, 1), (0, 0), (0, -1)], [(-1, 1), (-1, 0), (-1, -1)]], dtype=torch.long)
        # self.center_element_loc = [1, 1]

    def forward(self, x):
        B, T, H, W, C = x.shape

        Ch = int(np.ceil(C/self.num_elements_win)*self.num_elements_win)
        x = F.pad(x, (0, Ch-C, 1, 1, 1, 1))

        C_chunk = C // self.num_elements_win

        if self.inv:
            multiply = -1
        else:
            multiply = 1

        xs = torch.chunk(x, self.num_elements_win, dim=-1)
        x_shift = []
        for h in range(self.windows[0]):
            for w in range(self.windows[1]):
                i = h * self.windows[1] + w  
                # shift_t = self.shift_content[(h, w)] * multiply
                loc = self.shift_content[h, w]
                shift_h = (loc[0]) * multiply
                shift_w = (loc[1]) * multiply
                x_s = torch.roll(xs[i], shifts=(shift_h, shift_w), dims=(-3, -2))
                x_shift.append(x_s)

        out = torch.cat(x_shift, dim=-1)[..., 1:H+1, 1:W+1, :C].contiguous()
        return out


class Mlp_cnn_shift(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=2):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        self.fc = nn.Linear(in_features, self.hidden_features)
        # self.fc1 = nn.Conv2d(in_channels=in_features,
        #                                  out_channels=self.hidden_features,
        #                                  kernel_size=3,
        #                                  stride=1, padding=1)
        self.shift = PatchShift2D(inv=False)
        self.fc1 = nn.Linear(self.hidden_features, out_features)
        self.shift_inv = PatchShift2D(inv=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)

        dim = out_features
        self.reweight = Mlp(dim, dim // 4, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        # x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x = self.fc(x)
        x = self.act(x)
        # x = self.drop(x)

        # x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        h = self.shift_inv(self.act(self.fc1(self.shift(x))))
        w = self.act(self.fc2(x))

        a = (h + w).permute(0, 4, 1, 2, 3).contiguous().flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 2).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1]
        x = self.proj(x)
        x = self.drop(x)
        return x


class Mlp_ir(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, exp_r=4):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(in_channels=self.hidden_features,
                                         out_channels=self.hidden_features,
                                         kernel_size=3,
                                         stride=1, padding=1,
                                         groups=self.hidden_features)
        self.act2 = act_layer()
        self.fc3 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        # x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, self.hidden_features, H, W)
        x = x + self.act2(self.fc2(x))

        x = x.view(B, T, self.hidden_features, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc3(x)
        x = self.drop(x)
        return x


class Mlp_ir_multi(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, stage_n=[1, 3, 5, 7], exp_r=4):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or int(in_features * exp_r)
        self.scale_n = len(stage_n)
        self.ratio = exp_r

        k = stage_n
        s = [1, 1, 1, 1]
        p = [int(x/2) for x in stage_n]

        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act1 = act_layer()
        self.fc2_multi = nn.ModuleList([
                                nn.Conv2d(in_channels=self.hidden_features//len(stage_n),
                                         out_channels=self.hidden_features//len(stage_n),
                                         kernel_size=k[i],
                                         stride=s[i], padding=p[i],
                                         groups=self.hidden_features//len(stage_n)) for i in range(len(stage_n))])
        self.act2 = act_layer()
        self.fc3 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        feat_ = []
        expansion_c = int(self.ratio*C)
        assert expansion_c == self.hidden_features
        # x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, expansion_c, H, W)
        feat = torch.chunk(x, self.scale_n, 1)
        for i, blk in enumerate(self.fc2_multi):
            mid = blk(feat[i])
            feat_.append(mid)
        x = torch.cat(feat_, dim=1)    
        x = x + self.act2(x)

        x = x.view(B, T, expansion_c, H, W).permute(0, 1, 3, 4, 2).contiguous()

        x = self.fc3(x)
        x = self.drop(x)
        return x   






    def forward(self, x, flow_forward=None, flow_backward=None) -> torch.Tensor:
        B, T, H, W, C = x.shape
        self.temporal_dim = T
        output = []
        feat_prop = x.new_zeros(B, H, W, C)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        location = torch.stack([grid_x, grid_y], 0).type_as(x).expand(B, -1, -1, -1)
        backward_features_buffer = []
        for i in range(T - 1, -1, -1):
            x_i = x[:, i, ...]
            if i == T - 1:
                feat_prop = x_i
                backward_features_buffer.append(feat_prop)
            elif i < T - 1:
                flow = flow_backward[:, i, ...]
                guided_feature = flow_warp((feat_prop.permute(0, 3, 1, 2).contiguous()), (flow.permute(0, 2, 3, 1).contiguous()), padding_mode='border').permute(0, 2, 3, 1).contiguous()
                location = flow_warp(location, (flow.permute(0, 2, 3, 1).contiguous()), interpolation='nearest', padding_mode='border')
                backward_features = torch.stack(backward_features_buffer, 1)
                feat_prop = self.mixing(x_i, backward_features, location, guided_feature)
                backward_features_buffer.append(feat_prop)
                location = torch.cat([location, torch.stack([grid_x, grid_y], 0).type_as(x).expand(B, -1, -1, -1)], 1)

        output_backward = backward_features_buffer[::-1]
        del backward_features_buffer
        feat_prop = torch.zeros_like(feat_prop)
        location = torch.stack([grid_x, grid_y], 0).type_as(x).expand(B, -1, -1, -1)
        forward_features_buffer = []
        for i in range(0, T):
            x_i = x[:, i, ...]
            if i == 0:
                feat_prop = x_i
                forward_features_buffer.append(feat_prop)
                out = self.fusion(torch.cat([output_backward[i], feat_prop, x_i], -1))
                output.append(out)
            elif i > 0:
                flow = flow_forward[:, i - 1, ...]
                guided_feature = flow_warp((feat_prop.permute(0, 3, 1, 2).contiguous()), (flow.permute(0, 2, 3, 1).contiguous()), padding_mode='border').permute(0, 2, 3, 1).contiguous()
                location = flow_warp(location, (flow.permute(0, 2, 3, 1).contiguous()), interpolation='nearest', padding_mode='border')
                forward_features = torch.stack(forward_features_buffer, 1)
                feat_prop = self.mixing(x_i, forward_features, location, guided_feature)
                forward_features_buffer.append(feat_prop)
                location = torch.cat([location, torch.stack([grid_x, grid_y], 0).type_as(x).expand(B, -1, -1, -1)], 1)
                out = self.fusion(torch.cat([output_backward[i], feat_prop, x_i], -1))
                output.append(out)

        res = torch.stack(output, 1)
        return res

# MAXIM
class Double_axis_spatialmixing_block(nn.Module):

    def __init__(self, embed_dim=96, block_len=(4, 4), grid_len=(4, 4)):
        super().__init__()
        self.grid = grid_len
        self.block = block_len
        self.chunk_grid = grid_len[0] * grid_len[1]
        self.chunk_block = block_len[0] * block_len[1]
        self.grid_fc = nn.Linear(embed_dim, embed_dim)
        self.block_fc = nn.Linear(embed_dim, embed_dim)
        self.channel_fc = nn.Linear(embed_dim, embed_dim)
        self.reweight = Mlp(embed_dim, embed_dim // 4, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def grid_partition(self, x):
        B, T, H, W, C = x.shape
        S = C // self.chunk_grid
        G = self.chunk_grid
        f = self.ggh * self.ggw
        grid = rearrange(x, 'B T (gh fh) (gw fw) C -> B T (gh gw) (fh fw) C', gh=(self.grid[0]), gw=(self.grid[1])).contiguous()
        grid = grid.reshape(B, T, G, f, self.chunk_grid, S).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(B, T, self.chunk_grid, f, C)
        return grid

    def block_partition(self, x):
        B, T, H, W, C = x.shape
        S = C // self.chunk_block
        N = self.chunk_block
        g = self.ffh * self.ffw
        block = rearrange(x, 'B T (gh fh) (gw fw) C -> B T (gh gw) (fh fw) C', fh=(self.block[0]), fw=(self.block[1])).contiguous()
        block = block.reshape(B, T, g, N, self.chunk_block, S).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, g, self.chunk_block, C)
        return block

    def grid_unpartition(self, grid):
        B, T, ghw, bhw, C = grid.shape
        S = C // self.chunk_grid
        G = self.chunk_grid
        f = self.ggh * self.ggw
        grid = grid.reshape(B, T, self.chunk_grid, f, G, S).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(B, T, G, f, C)
        x = rearrange(grid, 'B T (gh gw) (fh fw) C -> B T (gh fh) (gw fw) C', gh=(self.grid[0]), gw=(self.grid[1]), fh=(self.ggh), fw=(self.ggw)).contiguous().reshape(B, T, self.H, self.W, C)
        return x

    def block_unpartition(self, block):
        B, T, ghw, bhw, C = block.shape
        S = C // self.chunk_block
        N = self.chunk_block
        g = self.ffh * self.ffw
        block = block.reshape(B, T, g, self.chunk_block, N, S).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, g, N, C)
        x = rearrange(block, 'B T (gh gw) (fh fw) C -> B T (gh fh) (gw fw) C', fh=(self.block[0]), fw=(self.block[1]), gh=(self.ffh), gw=(self.ffw)).contiguous().reshape(B, T, self.H, self.W, C)
        return x

    def forward(self, x):
        B, T, H, W, C = x.shape
        assert self.grid[0] == self.block[0], 'for convenient, both of them keep the same value.'
        Hp = int(np.ceil(H / self.grid[0])) * self.grid[0]
        Wp = int(np.ceil(W / self.grid[1])) * self.grid[1]
        x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
        self.H, self.W = Hp, Wp
        self.ggh, self.ggw = Hp // self.grid[0], Wp // self.grid[1]
        self.ffh, self.ffw = Hp // self.block[0], Wp // self.block[1]
        grid = self.grid_fc(self.grid_partition(x))
        block = self.block_fc(self.block_partition(x))
        x_grid = self.grid_unpartition(grid)[:, :, :H, :W, :].contiguous()
        x_block = self.block_unpartition(block)[:, :, :H, :W, :].contiguous()
        x_c = self.channel_fc(x)[:, :, :H, :W, :]
        a = (x_block + x_grid + x_c).permute(0, 4, 1, 2, 3).contiguous().flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x = x_block * a[0] + x_grid * a[1] + x_c * a[2]
        x = self.proj(x)
        return x


class Enhanced_MorphFCs(nn.Module):
    def __init__(self, dim, chunk_h=8, chunk_w=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w

        self.Ch = int(np.ceil(dim / self.chunk_h)) * self.chunk_h
        self.mlp_h = nn.Linear((self.Ch), (self.Ch), bias=qkv_bias)
        self.Cw = int(np.ceil(dim / self.chunk_w)) * self.chunk_w
        self.mlp_w = nn.Linear((self.Cw), (self.Cw), bias=qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, T, H, W, C = x.shape
        Hp = int(np.ceil(H / self.chunk_h)) * self.chunk_h
        Wp = int(np.ceil(W / self.chunk_w)) * self.chunk_w

        S_h = self.Ch // self.chunk_h
        tmp_h = self.chunk_h
        S_w = self.Cw // self.chunk_w
        tmp_w = self.chunk_w

        if mask is not None:
            pass
        else:
            
            x_h = F.pad(x, (0, self.Ch - C, 0, 0, 0, Hp - H))
            h = x_h.transpose(3, 2).contiguous().reshape(B, T, Hp * W // tmp_h, tmp_h, self.chunk_h, S_h).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, Hp * W // tmp_h, self.chunk_h, tmp_h * S_h)
            h = self.mlp_h(h).reshape(B, T, Hp * W // tmp_h, self.chunk_h, tmp_h, S_h).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, W, Hp, self.Ch).transpose(3, 2).contiguous()[..., 0:H, :, :C]
          
            x_w = F.pad(x, (0, self.Cw - C, 0, Wp - W))
            w = x_w.reshape(B, T, H * Wp // tmp_w, tmp_w, self.chunk_w, S_w).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, H * Wp // tmp_w, self.chunk_w, tmp_w * S_w)
            w = self.mlp_w(w).reshape(B, T, H * Wp // tmp_w, self.chunk_w, tmp_w, S_w).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, H, Wp, self.Cw)[..., 0:W, :C]

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 4, 1, 2, 3).contiguous().flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction=8, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if bn: 
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: 
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        res = res.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        return res    


class scale_func(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.s = scale
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x) / self.s 


class Enhanced_MorphFCs_decay(nn.Module):
    def __init__(self, 
                 dim, 
                 chunk_h=8, chunk_w=8, 
                 qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 # decay=True,
                 non_linear=False,
                 gating=True, symm=True, symm_act=nn.Tanh,
                 relu_scale=False, relu_scale_norm=False,
                 channel_mixer='vanilla'):
        super().__init__()
        self.chunk_h = chunk_h
        self.chunk_w = chunk_w
        # self.decay = decay
        
        self.non_linear = non_linear
        self.if_gating = gating  
        self.symmetric_act = symm  
        self.channel_mixer = channel_mixer

        # Relu-T from the Google
        self.relu_scale = relu_scale
        self.relu_scale_norm = relu_scale_norm

        self.Ch = int(np.ceil(dim / self.chunk_h)) * self.chunk_h
        self.Cw = int(np.ceil(dim / self.chunk_w)) * self.chunk_w
        if not non_linear:
            self.mlp_h = nn.Linear((self.Ch), (self.Ch), bias=qkv_bias)
            self.mlp_w = nn.Linear((self.Cw), (self.Cw), bias=qkv_bias)
            if channel_mixer == 'vanilla':
                self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
            elif channel_mixer == 'rcab':
                self.mlp_c = RCAB(n_feat=dim)    
        else:
            self.mlp_h = nn.Sequential(
                nn.Linear((self.Ch), (self.Ch), bias=qkv_bias),
                nn.ReLU()
            )    
            self.mlp_w = nn.Sequential(
                nn.Linear((self.Cw), (self.Cw), bias=qkv_bias),
                nn.ReLU()
            )
            if channel_mixer == 'vanilla':
                self.mlp_c = nn.Sequential(
                    nn.Linear(dim, dim, bias=qkv_bias),
                    nn.ReLU()
            )
            elif channel_mixer == 'rcab':    
                self.mlp_c = RCAB(n_feat=dim)  # use SE module
        
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # if decay:
        self.decay_h = torch.log(1 - 2 ** (-5 - torch.arange(chunk_h-1, -1, -1, dtype=torch.float))).exp()
        self.decay_w = torch.log(1 - 2 ** (-5 - torch.arange(chunk_w-1, -1, -1, dtype=torch.float))).exp()
        # self.register_buffer("decay_h", gamma_h)
        # self.register_buffer('decay_w', gamma_w)
        self.form_decay(sh=self.Ch//self.chunk_h, sw=self.Cw//self.chunk_w)
        # else:
            # raise Exception('Decay should be True, but get False.')

        if gating:  # if use gate
            if not symm:
                self.gating_fc = nn.Linear(dim, dim)
            else:
                self.gating_fc = symm_act()  # symmetric act    

       
        if relu_scale_norm:
            self.scale_h = rnorm(dim=self.Ch)
            self.scale_w = rnorm(dim=self.Cw)
            self.scale_c = rnorm(dim=dim)
        elif relu_scale:
            self.scale_h = scale_func(scale=self.Ch)
            self.scale_w = scale_func(scale=self.Cw)
            self.scale_c = scale_func(scale=dim)
        else:
            self.scale_h = nn.Identity()    
            self.scale_w = nn.Identity() 
            self.scale_c = nn.Identity()         

    def get_gamma(self, heads, device=None):
        
        gamma = torch.log(1 - 2 ** (-5 - torch.arange(heads, dtype=torch.float, device=device)))
        return gamma
    
    @torch.no_grad()
    def form_decay(self, sh, sw):
        gamma_h = self.decay_h.unsqueeze(-1)
        gamma_w = self.decay_w.unsqueeze(-1)
      
        gamma_h = torch.repeat_interleave(gamma_h, self.chunk_h, -1)  # 16, 16
        gamma_w = torch.repeat_interleave(gamma_w, self.chunk_w, -1)

        for i in range(1, gamma_h.shape[-1]):
            gamma_h[:, i] *= gamma_h[:, i-1]
        for i in range(1, gamma_w.shape[-1]):
            gamma_w[:, i] *= gamma_w[:, i-1]   

        
        gamma_h = gamma_h.unsqueeze(-1).expand(-1, -1, sh).flatten(-2).unsqueeze(-1).expand(-1, -1, sh)  # 16, 112, 7
        gamma_w = gamma_w.unsqueeze(-1).expand(-1, -1, sw).flatten(-2).unsqueeze(-1).expand(-1, -1, sw)  

        gamma_h = gamma_h.flatten(-2).view(self.chunk_h, self.chunk_h, sh*sh)  # 16, 16, 49
        gamma_w = gamma_w.flatten(-2).view(self.chunk_w, self.chunk_w, sw*sw) 

        shift_h, shift_w = [], []
        shift_h.append(gamma_h)
        shift_w.append(gamma_w)    
        for i in range(1, self.chunk_h):
            if i == 1:
                mid = torch.roll(gamma_h, 1, dims=-2)
                mid[:, 0, :] = gamma_h[:, i, :]
            else:
                mid = torch.roll(mid, 1, dims=-2)
                mid[:, 0, :] = gamma_h[:, i, :]   
            shift_h.append(mid)    

        for i in range(1, self.chunk_w):
            if i == 1:
                mid = torch.roll(gamma_w, 1, dims=-2)
                mid[:, 0, :] = gamma_w[:, i, :]
            else:
                mid = torch.roll(mid, 1, dims=-2)
                mid[:, 0, :] = gamma_w[:, i, :]   
            shift_w.append(mid)    

        # 16, 112, 112
        gamma_h = torch.stack(shift_h, -1).view(self.chunk_h, self.chunk_h, sh, sh, -1).transpose(-2, -1).contiguous().reshape(self.chunk_h, self.chunk_h*sh, self.chunk_h*sh)     
        gamma_w = torch.stack(shift_w, -1).view(self.chunk_w, self.chunk_w, sw, sw, -1).transpose(-2, -1).contiguous().reshape(self.chunk_w, self.chunk_w*sw, self.chunk_w*sw)
        
        gamma_h = torch.mean(gamma_h, 0)  # Ch, Ch
        gamma_w = torch.mean(gamma_w, 0)
        self.register_buffer('gamma_h', gamma_h)
        self.register_buffer('gamma_w', gamma_w)
        # return gamma_h, gamma_w

    def get_activation_fn(self, activation='swish'):
        if activation == "swish":
            return F.silu
        elif activation == "gelu":
            return F.gelu
        else:
            raise NotImplementedError

    def forward(self, x, mask=None):
        B, T, H, W, C = x.shape

        if self.if_gating:
            x_short = x.clone()

        Hp = int(np.ceil(H / self.chunk_h)) * self.chunk_h
        Wp = int(np.ceil(W / self.chunk_w)) * self.chunk_w

        S_h = self.Ch // self.chunk_h
        tmp_h = self.chunk_h
        S_w = self.Cw // self.chunk_w
        tmp_w = self.chunk_w

        

        if mask is not None:
            pass
        else:
          
            x_h = F.pad(x, (0, self.Ch - C, 0, 0, 0, Hp - H))
            h = x_h.transpose(3, 2).contiguous().reshape(B, T, Hp * W // tmp_h, tmp_h, self.chunk_h, S_h).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, Hp * W // tmp_h, self.chunk_h, tmp_h * S_h)
            if not self.non_linear:
                self.mlp_h.weight.data.mul_(self.gamma_h.detach()) 
            else:    
                self.mlp_h[0].weight.data.mul_(self.gamma_h.detach())  
            # mlp
            h = self.mlp_h(h)
           
            h = self.scale_h(h).reshape(B, T, Hp * W // tmp_h, self.chunk_h, tmp_h, S_h).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, W, Hp, self.Ch).transpose(3, 2).contiguous()[..., 0:H, :, :C]
            # h = self.mlp_h(h).reshape(B, T, Hp * W // tmp_h, self.chunk_h, tmp_h, S_h).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, W, Hp, self.Ch).transpose(3, 2).contiguous()[..., 0:H, :, :C]
            
           
            x_w = F.pad(x, (0, self.Cw - C, 0, Wp - W))
            w = x_w.reshape(B, T, H * Wp // tmp_w, tmp_w, self.chunk_w, S_w).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, H * Wp // tmp_w, self.chunk_w, tmp_w * S_w)
            if not self.non_linear:
                self.mlp_w.weight.data.mul_(self.gamma_w.detach())  
            else:
                self.mlp_w[0].weight.data.mul_(self.gamma_w.detach())    
            # mlp
            w = self.mlp_w(w)
            # scale
            w = self.scale_w(w).reshape(B, T, H * Wp // tmp_w, self.chunk_w, tmp_w, S_w).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, H, Wp, self.Cw)[..., 0:W, :C]
            # w = self.mlp_w(w).reshape(B, T, H * Wp // tmp_w, self.chunk_w, tmp_w, S_w).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(B, T, H, Wp, self.Cw)[..., 0:W, :C]
           
        c = self.mlp_c(x)
        c = self.scale_c(c)

        a = (h + w + c).permute(0, 4, 1, 2, 3).contiguous().flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)

        if self.if_gating:
            if not self.symmetric_act:
                x_short = self.get_activation_fn(activation='swish')(self.gating_fc(x_short)) 
                x = x_short * self.get_activation_fn(activation='gelu')(x)
            else:
                seco_path = self.gating_fc(x)
                x = (x_short + x) * seco_path  

        x = self.proj_drop(x)
        return x


class TimemixingFC(nn.Module):

    def __init__(self, embed_dim, head=6, frame_lens=16, segment_lens=2, chunk_lens=7, win_size=4):
        super().__init__()
        self.S = segment_lens
        self.shift_windows = segment_lens // 2
        self.lr_fc = nn.Linear(embed_dim, embed_dim)
        self.sw_fc = nn.Linear(embed_dim, embed_dim)
        self.num_frames_morph = False
        self.chunk_lens = segment_lens

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_s = nn.Linear(embed_dim, embed_dim)

    def window_fc(self, x):
        B, D, H, W, C = x.size()
        if D % self.S != 0:
            x = torch.cat([x, x[:, 0:1, ...]], 1)
            D += 1
            self.num_frames_morph = True
        
        x = x.reshape(B, D // self.S, self.S, H, W, self.chunk_lens, C // self.chunk_lens).permute(0, 1, 5, 3, 4, 2, 6).contiguous().reshape(B, D, H, W, -1)
        x = self.lr_fc(x)
        x = x.reshape(B, D // self.S, self.chunk_lens, H, W, self.S, C // self.chunk_lens).permute(0, 1, 5, 3, 4, 2, 6).contiguous().reshape(B, D, H, W, C)
        x = self.proj(x)
        if self.num_frames_morph:
            y = x[:, 0:D - 1, ...]
            self.num_frames_morph = False
        else:
            y = x
        return y

    def shift_fc(self, x):
        B, D, H, W, C = x.size()
        x = torch.roll(x, self.shift_windows, 1)
        if D % self.S != 0:
            x = torch.cat([x, x[:, 0:1, ...]], 1)
            D += 1
            self.num_frames_morph = True
        
        x = x.reshape(B, D // self.S, self.S, H, W, self.chunk_lens, C // self.chunk_lens).permute(0, 1, 5, 3, 4, 2, 6).contiguous().reshape(B, D, H, W, -1)
        x = self.sw_fc(x)
        x = x.reshape(B, D // self.S, self.chunk_lens, H, W, self.S, C // self.chunk_lens).permute(0, 1, 5, 3, 4, 2, 6).contiguous().reshape(B, D, H, W, C)
        x = self.proj_s(x)
        if self.num_frames_morph:
            x = x[:, 0:D - 1, ...]
            self.num_frames_morph = False
        else:
            x = x
        y = torch.roll(x, -self.shift_windows, 1)
        return y

    def forward(self, x):
        B, D, H, W, C = x.size()
        x = self.window_fc(x)
        out = self.shift_fc(x)
        return out


class TimemixingFC_channel(nn.Module):

    def __init__(self, embed_dim, segment_lens=2, win=2, shift=False):
        super().__init__()
        self.S = segment_lens
        self.W = win
        self.shift_windows = win // 2
        self.shift = shift
        self.lr_fc = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        if shift:
            self.sw_fc = nn.Linear(embed_dim, embed_dim*3)
            self.proj_s = nn.Linear(embed_dim, embed_dim)
        self.flag = False
        
    def rearrange_shape(self, x):
        B, D, H, W, C = x.shape
        self.flag = True
        delta_T = int(np.ceil(D / self.window_size[0])) * self.window_size[0] - D  # 18-16=2
        delta_ = list(range(-1, -(delta_T+1), -1))  # [-1, -2]

        new_seq_start = list(range(0, int(D//self.window_size[0])*self.window_size[0]))  # [0, ..., 11]
        new_seq_end = list(range(int(D//self.window_size[0])*self.window_size[0], D))  # [12, 13, 14, 15]
        new_seq = new_seq_start + delta_ + new_seq_end
        seq_back = new_seq_start + list(range(-1, -(len(new_seq_end)+1), -1))[::-1]

        repeat_index = [new_seq_start[i] for i in delta_]  # [11, 10]
        # x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, C, -1, D)
        x = torch.cat([x, x[:, repeat_index, ...]], dim=1)[:, new_seq, ...]
        # x = x.reshape(B, C, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        return x, seq_back

    def window_fc(self, x):
        B, D, H, W, C = x.size()

        if D % self.S != 0:
            x, seq_back = self.rearrange_shape(x)
            B, D, H, W, C = x.shape
        
        x = self.lr_fc(x)
        x = x.view(B, D//self.W, self.W, H, W, 3, C//self.S, self.S).reshape(-1, self.W, H, W, 3, C//self.S, self.S)
        x = x.permute(4, 0, 2, 3, 1, 5, 6).contiguous().reshape(3, -1, self.W*(C//self.S), self.S)
        q, k, v = x[0], x[1], x[2]

        attn = torch.einsum('b m c, b n c -> b m n', q, k)  # b n n
        x = torch.einsum('b m n, b n c -> b m c', attn, v)  # b n c

        x = x.reshape(B, D//self.W, H, W, self.W, C).permute(0, 1, 4, 2, 3, 5).contiguous().view(B, D, H, W, C)

        x = self.proj(x)

        if self.flag:
            x = x[:, seq_back, ...]
            self.flag = False    
            B, D, H, W, C = x.shape

        return x

    def shift_fc(self, x):
        B, D, H, W, C = x.size()

        if D % self.S != 0:
            x, seq_back = self.rearrange_shape(x)
            B, D, H, W, C = x.shape
        
        x = self.sw_fc(x)
        x = x.view(B, D//self.W, self.W, H, W, 3, C//self.S, self.S).reshape(-1, self.W, H, W, 3, C//self.S, self.S)
        x = x.permute(4, 0, 2, 3, 1, 5, 6).contiguous().reshape(3, -1, self.W*(C//self.S), self.S)
        q, k, v = x[0], x[1], x[2]

        attn = torch.einsum('b m c, b n c -> b m n', q, k)  # b n n
        x = torch.einsum('b m n, b n c -> b m c', attn, v)  # b n c

        x = x.reshape(B, D//self.W, H, W, self.W, C).permute(0, 1, 4, 2, 3, 5).contiguous().view(B, D, H, W, C)

        x = self.proj_s(x)

        if self.flag:
            x = x[:, seq_back, ...]
            self.flag = False    
            B, D, H, W, C = x.shape

        return x
        
    def forward(self, x):
        B, D, H, W, C = x.size()
        x = self.window_fc(x)
        if self.shift:
            x = torch.roll(x, self.shift_windows, 1)
            out = self.shift_fc(x)
            x = torch.roll(out, -self.shift_windows, 1)
        return x


class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConvPack(ModulatedDeformConv):

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

# DCN
class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)


class Mlp_GEGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DWConv(nn.Module):
    def __init__(self, 
                 embed_dim=112,
                 stride=1) -> None:
        super().__init__()
        self.dwconv = Conv3x3ReLU(embed_dim, embed_dim, stride, groups=embed_dim)

    def forward(self, x):
        B, T, H, W, C = x.shape

        x = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()

        return x  # B T H W C


class TAB(nn.Module):

    def __init__(self, 
                 embed_dim=96, 
                 head=6, 
                 num_frames=7, 
                 chunk_dim=7, 
                 block_len=4, grid_len=4, 
                 chunk_h=8, chunk_w=8, 
                 mlp_ratio=2, n_groups=1,
                 qkv_bias=False, attn_drop=0.0, proj_drop=0.0, 
                 drop_path=0, 
                 norm_layer=nn.LayerNorm, 
                 win_size=4,
                 shift=False,
                 if_decay=False,
                 non_linear=True,
                 gating=True, symm=True, symm_act=nn.Tanh,
                 relu_scale=True, relu_scale_norm=False,
                 ffn='vanilla',
                 mixer_type='mlps', mixer_n=10,
                 mixer_scaling=1.,
                 channel_mixer='vanilla'):
        super().__init__()
        self.spatial_scale = mixer_scaling
        self.norm2 = norm_layer(embed_dim)
        if mixer_type == 'mlps':

            if not if_decay:
                self.spatial_mixing = Enhanced_MorphFCs(dim=embed_dim,
                                                    chunk_h=chunk_h,
                                                    chunk_w=chunk_w,
                                                    qkv_bias=qkv_bias,
                                                    qk_scale=None,
                                                    attn_drop=attn_drop,
                                                    proj_drop=proj_drop)
            else:
                self.spatial_mixing = Enhanced_MorphFCs_decay(dim=embed_dim,
                                                              chunk_h=chunk_h, chunk_w=chunk_w,
                                                              qkv_bias=qkv_bias,
                                                              qk_scale=None,
                                                              attn_drop=attn_drop,
                                                              proj_drop=proj_drop,
                                                              # decay=True,
                                                              non_linear=non_linear,
                                                              gating=gating, symm=symm, symm_act=symm_act,
                                                              relu_scale=relu_scale, relu_scale_norm=relu_scale_norm,
                                                              channel_mixer=channel_mixer)    
        elif mixer_type == 'mbconv':
            self.spatial_mixing = Multi_MBConv(embed_dim=embed_dim,  
                                               expansion_factor=4, 
                                               stride=1,
                                               num_blocks=mixer_n)     
        else:
            raise Exception('please input correct mixer type.')   
        
        self.norm3 = norm_layer(embed_dim)
        if ffn == 'vanilla':
            self.channel_mixing = Mlp(in_features=embed_dim, hidden_features=(int(embed_dim * mlp_ratio)), act_layer=(nn.GELU), drop=0.0)
        elif ffn == 'ffn_cnn':
            self.channel_mixing = Mlp_cnn(in_features=embed_dim, act_layer=nn.GELU, drop=0.0, exp_r=mlp_ratio, n_groups=n_groups)
        elif ffn == 'ffn_cnn_shift':
            self.channel_mixing = Mlp_cnn_shift(in_features=embed_dim, act_layer=nn.GELU, drop=0., exp_r=mlp_ratio)    
        elif ffn == 'irffn_single':
           
            self.channel_mixing = Mlp_ir(in_features=embed_dim, act_layer=nn.GELU, drop=0.0, exp_r=mlp_ratio)
        elif ffn == 'irffn_multi':
         
            self.channel_mixing = Mlp_ir_multi(in_features=embed_dim, act_layer=nn.GELU, drop=0.0, exp_r=mlp_ratio)
        else:
            raise Exception('please input correct type of ffn.')    
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, flow_forward=None, flow_backward=None):
        B, T, H, W, C = x.shape
      
        x = x + self.drop_path(self.spatial_mixing(self.norm2(x))) * self.spatial_scale
        x = x + self.drop_path(self.channel_mixing(self.norm3(x))) * self.spatial_scale
        return x


class TABTAB(nn.Module):
    def __init__(self, 
                 embed_dim=96, 
                 head=6, 
                 num_frames=7, 
                 chunk_dim=7, 
                 block_len=4, grid_len=4, 
                 chunk_h=8, chunk_w=8, 
                 mlp_ratio=2, 
                 qkv_bias=False, attn_drop=0.0, proj_drop=0.0, 
                 drop_path=0, 
                 norm_layer=nn.LayerNorm, 
                 win_size=4,
                 shift=False):
        super().__init__()
        self.spatial_mixing_0 = TAB(embed_dim=embed_dim, 
                                             head=head, 
                                             num_frames=num_frames, 
                                             chunk_dim=chunk_dim, 
                                             block_len=block_len, grid_len=grid_len, 
                                             chunk_h=chunk_h, chunk_w=chunk_w, 
                                             mlp_ratio=mlp_ratio, 
                                             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, 
                                             drop_path=drop_path, 
                                             norm_layer=norm_layer, 
                                             win_size=win_size) 
        # Second
        self.spatial_mixing_1 = TAB(embed_dim=embed_dim, 
                                             head=head, 
                                             num_frames=num_frames, 
                                             chunk_dim=chunk_dim, 
                                             block_len=block_len, grid_len=grid_len, 
                                             chunk_h=chunk_h, chunk_w=chunk_w, 
                                             mlp_ratio=mlp_ratio, 
                                             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, 
                                             drop_path=drop_path, 
                                             norm_layer=norm_layer, 
                                             win_size=win_size)

    def forward(self, x):
        B, T, H, W, C = x.shape
       
        y = self.spatial_mixing_0(x)
        y = self.spatial_mixing_1(y)
        return y    


class Mlp_encoder(nn.Module):
    def __init__(self, 
                 embed_dim=96, 
                 depth=4, 
                 segm=6, 
                 num_frames=7, 
                 chunk_dim=7, 
                 block_len=4, grid_len=4, 
                 chunk_dim_h=8, chunk_dim_w=8, 
                 mlp_ratio=2.0, n_groups=1,
                 qkv_bias=False, attn_drop=0.0, proj_drop=0.0, drop_path=0.0, 
                 norm_layer=nn.LayerNorm, 
                 win_size=4, window_size=[2, 8, 8],
                 if_traj=True, 
                 n_nonkeyframes=0, 
                 aligned=False, empty_aligned=False, traj_r_n=5, 
                 deformable_groups=16, max_residue_magnitude=10, pa_frames=2,
                 is_train=True,
                 unfold_stride=0, unfold_conv=False,
                 ltam=True,
                 traj_win=4, traj_heads=4,
                 k_s=21,
                 if_smooth=True, region_range=4,
                 if_decay=False,
                 non_linear=True,
                 gating=True, symm=True, symm_act=nn.Tanh,
                 relu_scale=True, relu_scale_norm=False,
                 ffn_type='vanilla',
                 mixer_type='mlps', mixer_n=10,
                 r_scaling=1., traj_mode='wins', twins=[2, 2], traj_scale=True, traj_refine=None,
                 m_scaling=1.,
                 if_local_fuse=False,
                 channel_mixer='vanilla'):
        super().__init__()
        self.traj = if_traj
        self.aligned = aligned
        self.empty = empty_aligned
        self.pa_frames = pa_frames
        self.is_train = is_train
        self.head = segm
        self.unfold_stride = unfold_stride
        self.unfold_conv = unfold_conv
        self.if_smooth = if_smooth
        self.region_range = region_range

        self.local_fuse = if_local_fuse
        if if_local_fuse:
            self.local_cnn = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if unfold_stride > 0:
            self.stride = unfold_stride
            self.stride_2 = unfold_stride ** 2
            self.stride_w = tuple(repeat(unfold_stride, 2))
   
        self.mlp_blocks = nn.ModuleList([TAB(embed_dim=embed_dim, 
                                             head=segm, 
                                             num_frames=num_frames, 
                                             chunk_dim=chunk_dim, 
                                             block_len=block_len, grid_len=grid_len, 
                                             chunk_h=chunk_dim_h, chunk_w=chunk_dim_w, 
                                             mlp_ratio=mlp_ratio, n_groups=n_groups,
                                             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, 
                                             drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                                             norm_layer=norm_layer, 
                                             win_size=win_size,
                                             if_decay=if_decay,
                                             non_linear=non_linear,
                                             gating=gating, symm=symm, symm_act=symm_act,
                                             relu_scale=relu_scale, relu_scale_norm=relu_scale_norm,
                                             ffn=ffn_type,
                                             mixer_type=mixer_type, mixer_n=mixer_n,
                                             mixer_scaling=m_scaling,
                                             channel_mixer=channel_mixer) for i in range(depth)])

        if aligned:
            self.pa_deform = DCNv2PackFlowGuided(embed_dim, embed_dim, 3, padding=1, deformable_groups=deformable_groups, max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
            self.pa_fuse = ResidualBlocksWithInputConv(embed_dim*3, embed_dim, traj_r_n)
        elif aligned is None:
            if self.empty:
                self.traj_mixing = nn.Identity()
            else:
                if unfold_conv:
                    dims = embed_dim * self.stride
                elif unfold_stride > 0:
                    dims = embed_dim * self.stride_2    
                else:
                    dims = embed_dim  
                self.traj_mixing = DecoderLayer(dim=dims, 
                                            input_resolution=segm, 
                                            depth=2,  
                                            num_heads=segm, 
                                            window_size=window_size, shift_size=None, 
                                            mlp_ratio=mlp_ratio, 
                                            qkv_bias=qkv_bias, qk_scale=None, 
                                            drop=0.0, attn_drop=attn_drop, drop_path=0., 
                                            norm_layer=norm_layer,
                                            is_train=self.is_train,
                                            # if_unfold=True if window_size[0] == 2 or window_size[0] == 4 else False)
                                            if_unfold=False)
        else:
            self.traj_mixing = Trajectory_multi_head(embed_dim=embed_dim,
                                                     mode=traj_mode,
                                                     num_blocks=traj_r_n,  
                                                     frame_stride=n_nonkeyframes,
                                                     traj_win=traj_win,  
                                                     head=traj_heads, 
                                                     en_field=False,
                                                     head_scale=traj_scale, 
                                                     feature_refine=traj_refine,  
                                                     r_scaling=r_scaling,  
                                                     twins=twins,  
                                                     ltam=ltam
                                                     )

        if unfold_conv and unfold_stride > 0:
            self.unfold_c_down = nn.Conv2d(in_channels=embed_dim*self.stride_2, 
                                       out_channels=embed_dim*self.stride, 
                                       kernel_size=1, 
                                       stride=1, padding=0)
            self.unfold_c_up = nn.Conv2d(in_channels=embed_dim*self.stride, 
                                       out_channels=embed_dim*self.stride_2, 
                                       kernel_size=1, 
                                       stride=1, padding=0)
            
    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""

        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 4 frames."""
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_backward.insert(0, self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        x_forward = [
         torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]
            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def video_unfold(self, x):
        B, T, C, H, W = x.shape
        x = F.unfold(x.view(-1, C, H, W), kernel_size=self.stride_w, padding=0, stride=self.stride//2)  # B*T, C*2*2, (H-1)*(W-1)
        x = F.fold(x, output_size=(2*(H-1), 2*(W-1)), kernel_size=self.stride_w, padding=0, stride=self.stride)  # B*T, C, 2(H-1), 2(W-1)
        x = F.adaptive_avg_pool2d(x, (H, W))  # B*T, C, H, W
        x = F.unfold(x, kernel_size=self.stride_w, padding=0, stride=self.stride)  # B*T, C*2*2, H//2*W//2
        x = F.fold(x, output_size=(H//self.stride, W//self.stride), kernel_size=(1, 1), padding=0, stride=1)  # B*T, C*2*2, H//2, W//2
        if self.unfold_conv:
            x = self.unfold_c_down(x)  # B*T, C*2, H//2, W//2
            return x.view(B, T, C*self.stride, H//self.stride, W//self.stride)
        else:    
            return x.view(B, T, C*self.stride_2, H//self.stride, W//self.stride)
    
    def video_fold(self, x):
        B, T, C, H, W = x.shape
        if self.unfold_conv:
            x = self.unfold_c_up(x.view(-1, C, H, W)).reshape(B, T, -1, H, W)  
            _, T, C, H, W = x.shape
        x = F.fold(x.view(-1, C, H*W), output_size=(H*self.stride, W*self.stride), kernel_size=self.stride_w, padding=0, stride=self.stride)  # B*T, C//(4*4), H*4, W*4
        return x.view(B, T, C//self.stride_2, H*self.stride, W*self.stride)

    def flow_smoothing(self, flow, region_range=4):
        B, T, C, H, W = flow.shape

        flow = flow.view(-1, C, H, W)

        hf = int(np.ceil(H/region_range)) * region_range
        wf = int(np.ceil(W/region_range)) * region_range
        flow = F.pad(flow, (0, wf-W, 0, hf-H), mode='reflect')

        flow = F.adaptive_avg_pool2d(flow, (hf//region_range, wf//region_range))
        flow = F.interpolate(flow, scale_factor=region_range, mode='nearest')[..., :H, :W].contiguous()

        return flow.view(B, T, C, H, W)

    def forward(self, x, flow_forward=None, flow_backward=None):
        B, T, C, H, W = x.shape

        # if self.local_fuse:
            # shortcut_input = x.clone()

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.local_fuse:
            shortcut_input = x.clone()

        if flow_forward is not None:
            flow_forward = flow_forward.reshape(B, T - 1, 2, H, W)
            flow_backward = flow_backward.reshape(B, T - 1, 2, H, W)
        
            if self.if_smooth:
                flow_backward = self.flow_smoothing(flow_backward, self.region_range)
                flow_forward = self.flow_smoothing(flow_forward, self.region_range)

 
        for blk in self.mlp_blocks:  # self.blocks
            x = blk(x)
       

        if self.local_fuse:
            x = rearrange(x, 'b t h w c -> (b t) c h w').contiguous()
            x = shortcut_input + rearrange(self.local_cnn(x), '(b t) c h w -> b t h w c', b=B).contiguous()


        # Alignment
        if self.aligned:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
          
            x_backward, x_forward = getattr(self, f"get_aligned_feature_{self.pa_frames}frames")(x, flow_backward, flow_forward)
            x_backward = x_backward.view(-1, C, H, W)
            x_forward = x_forward.view(-1, C, H, W)
            x = x.view(-1, C, H, W)
            # x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2).contiguous())
            x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 1)).view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        elif self.aligned is None:
            if self.empty: 
                x = self.traj_mixing(x)
            else:
                x = x.permute(0, 1, 4, 2, 3).contiguous()
                x = self.traj_mixing(x)
                x = x.permute(0, 1, 3, 4, 2).contiguous()
        else:
            x = x.permute(0, 1, 4, 2, 3).contiguous()


            if self.unfold_stride > 0:
                x = self.video_unfold(x)
            # x = self.traj_mixing(x, flow_forward, flow_backward)

            if flow_forward is not None:
                # x = self.traj_mixing(x, flow_f=flow_forward, flow_b=flow_backward)
                x = self.traj_mixing(x, flow_forward, flow_backward)
            else:
                x = self.traj_mixing(x)
            if self.unfold_stride > 0:
                x = self.video_fold(x)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            # x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x.permute(0, 1, 4, 2, 3).contiguous()
    

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
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
        raise ValueError(f"The spatial sizes of input ({x.size()[-2:]}) and flow ({flow.size()[1:3]}) are not the same.")
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)
    grid.requires_grad = False

  
    grid_flow = grid + flow  

    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.to(dtype=(x.dtype))
    output = F.grid_sample(x,
      grid_flow,
      mode=interpolation,
      padding_mode=padding_mode,
      align_corners=align_corners)
    return output
