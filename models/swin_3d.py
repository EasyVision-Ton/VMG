import os, warnings, math, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair, _single
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from itertools import repeat


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
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


class Mlp_GEGLU(nn.Module):

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


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    return (tuple(use_window_size), tuple(use_shift_size))


def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
        for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
            for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (B*num_windows, window_size*window_size, C)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class rWindowAttention(nn.Module):
    def __init__(self, 
                 dim, 
                 window_size, 
                 num_heads, 
                 qkv_bias=False, qk_scale=None, 
                 attn_drop=0.0, proj_drop=0.0,
                 only_one=False,
                 selected_n=1,
                 align_mask=False,
                 is_train=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.temporal_length = window_size[0]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.oo = only_one
        self.is_train = is_train

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        self.register_buffer('relative_position_index', self.get_position_index(window_size))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)

        if not only_one:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=(-1))
        trunc_normal_((self.relative_position_bias_table), std=0.02)

        self.selected_token_nums = self.temporal_length - 1
        self.align_mask = align_mask
        # if align_mask:
        #     self.reweight = Mlp_GEGLU(dim, dim//4, dim*2)

        self.interval = window_size[1] * window_size[2]
        self.time_index = list(range(0, reduce(mul, window_size), self.interval))  # [0, 16]
        self.total_seq = list(range(0, reduce(mul, window_size)))

        # self.norm = norm_layer(dim)

    def forward(self, x, kv=None, mask=None, flow_f=None, flow_b=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        kv = x if kv is None else kv
        B_, N, C = x.shape
        B_, N_, C = kv.shape
        # print(f'********{N}')

        # x_v = x
        # x = self.norm(x)

        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        kv = self.kv(kv).reshape(B_, N_, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = self.v(x_v).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # B_, h, N, d

        token_map = []
        for i, value in enumerate(self.time_index):
            id_begin = i*self.interval  
            id_end = self.total_seq[-1]+1 if i == len(self.time_index)-1 else (i+1)*self.interval
        
            q_ = q[..., id_begin:id_end, :]
            other_id = list(filter(lambda p: p not in self.total_seq[id_begin:id_end], self.total_seq))
            k_, v_ = k[..., other_id, :], v[..., other_id, :]

            if not self.oo:
                if self.is_train:
                    x_out = self.attention(q_, k_, v_, mask, (B_, N, C), relative_position_encoding=True, query=self.total_seq[id_begin:id_end], key=other_id)  # B_, N_q, C
                else:
                    x_out, attn = self.attention(q_, k_, v_, mask, (B_, N, C), relative_position_encoding=True, query=self.total_seq[id_begin:id_end], key=other_id)  # B_, N_q, C
            else:
                if self.is_train:
                    x_out = self.attention_oo(q_, k_, v_, mask, (B_, N, C), relative_position_encoding=True, align_mask=self.align_mask, query=self.total_seq[id_begin:id_end], key=other_id)   
                else:
                    x_out, attn = self.attention_oo(q_, k_, v_, mask, (B_, N, C), relative_position_encoding=True, align_mask=self.align_mask, query=self.total_seq[id_begin:id_end], key=other_id)

            token_map.append(x_out)

        x_out = torch.cat(token_map, dim=1)  # B_, N, C
        x = self.proj(x_out)
        x = self.proj_drop(x)    

        if self.is_train:
            return x  # B_, N, C   
        else:
            return x, attn 
    
    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True, query=None, key=None):
        B_, N, C = x_shape
        N_q, N_k = len(query), len(key)
        # print(f'*******is{N_q, N_k}')
        q_id_begin, q_id_end = query[0], query[-1]+1
        # k_id_begin, k_id_end = key[0], key[-1]+1
        # print(f'******is{q_id_begin, q_id_end, k_id_begin, k_id_end}')
        assert N_k == N - N_q, "Check the relation of query and key."
        # print(query, key)

        attn = q * self.scale @ k.transpose(-2, -1)  # B_, h, Nq Nk
        # print(f'attn is {q.shape}')

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[q_id_begin:q_id_end, key].reshape(-1)].reshape(N_q, N_k, -1)
            position_enc = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
            # print(f'******************: {attn.shape, position_enc.shape}')
            attn = attn + position_enc
            
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_k) + mask[:, q_id_begin:q_id_end, key].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_k)
            attn = self.softmax(attn)    

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)

        if self.is_train:
            return x
        else:
            return x, attn

    def attention_oo(self, q, k, v, mask, x_shape, relative_position_encoding=True, align_mask=False, query=None, key=None):
        B_, N, C = x_shape
        N_q, N_k = len(query), len(key)
        q_id_begin, q_id_end = query[0], query[-1]+1
        assert N_k == N - N_q, "Check the relation of query and key."

        _, h, _, C_ = q.shape   # B_ h N C_
        out_list = []
        attn = q * self.scale @ k.transpose(-2, -1)  # B_ h Nq Nk
        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)
        # attn = q @ k.transpose(-2, -1)  # B_ h Nq Nk

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[q_id_begin:q_id_end, key].reshape(-1)].reshape(N_q, N_k, -1)
            position_enc = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
            # print(f'******************: {attn.shape, position_enc.shape}')
            attn = attn + position_enc
            
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_k) + mask[:, q_id_begin:q_id_end, key].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_k)
            attn = self.softmax(attn)   

        if not align_mask:
            soft_v, soft_index = torch.topk(attn, k=self.selected_token_nums, dim=-1)  # B_ h N 2
            soft_v = soft_v.unsqueeze(-2).expand(-1, -1, -1, C_, -1)  # B_ h N C_ 2
            # soft_v = F.softmax(soft_v, dim=-1)
            soft_index = soft_index.unsqueeze(-2).expand(-1, -1, -1, C_, -1)  # B_ h N C_ 2
            for i in range(soft_v.shape[-1]):
                out = torch.gather(v, 2, soft_index[..., i])  # B_ h N C_
                out_list.append(out*soft_v[..., i])
            x = sum(out_list).permute(0, 2, 1, 3).contiguous().reshape(B_, N_q, C)    
        else:

            soft_v, soft_index = torch.max(attn, dim=-1)  # B_ h Nq
            soft_v = soft_v.unsqueeze(-1).expand(-1, -1, -1, C_)  # B_ h Nq C_
            soft_index = soft_index.view(B_, -1, N_q, 1).expand(-1, -1, -1, C_)  # B_ h Nq C_
            output = torch.gather(v, 2, soft_index)  # B_ h Nq C_
            x = (output*soft_v).permute(0, 2, 1, 3).contiguous().reshape(B_, N_q, C)

        if self.is_train:
            return x
        else:
            return x, attn.detach().cpu().numpy()

    def get_position_index(self, window_size):
        """ Get pair-wise relative position index for each token inside the window. """
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """ Get sine position encoding """
        if scale is not None:
            if normalize is False:
                raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=(torch.float32))
        x_embed = not_mask.cumsum(2, dtype=(torch.float32))
        if normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_t = torch.arange(num_pos_feats, dtype=(torch.float32))
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class WindowAttention(nn.Module):

    def __init__(self, 
                 dim, 
                 window_size, 
                 num_heads, 
                 qkv_bias=False, qk_scale=None, 
                 attn_drop=0.0, proj_drop=0.0,
                 only_one=False,
                 selected_n=1,
                 align_mask=False,
                 is_train=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.temporal_length = window_size[0]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.oo = only_one
        self.is_train = is_train

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        self.register_buffer('relative_position_index', self.get_position_index(window_size))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, (dim * 2), bias=qkv_bias)
        if not only_one:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=(-1))
        trunc_normal_((self.relative_position_bias_table), std=0.02)

        self.selected_token_nums = selected_n + 1
        self.align_mask = align_mask
        if align_mask:
            self.reweight = Mlp_GEGLU(dim, dim//4, dim*2)

    def forward(self, x, kv=None, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        kv = x if kv is None else kv
        B_, N, C = x.shape
        B_, N_, C = kv.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B_, N_, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]

        if not self.oo:
            x_out = self.attention(x, q, k, v, mask, (B_, N, C), relative_position_encoding=True)
        else:
            if self.is_train:
                x_out = self.attention_oo(x, q, k, v, mask, (B_, N, C), relative_position_encoding=True, align_mask=self.align_mask)   
                x = self.proj(x_out)
                x = self.proj_drop(x)
                return x 
            else:
                x_out, attn = self.attention_oo(x, q, k, v, mask, (B_, N, C), relative_position_encoding=True, align_mask=self.align_mask)
                x = self.proj(x_out)
                x = self.proj_drop(x)
                return x, attn

    def attention(self, x, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape
        attn = q * self.scale @ k.transpose(-2, -1)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
            
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)    

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return x
    
    def attention_oo(self, x, q, k, v, mask, x_shape, relative_position_encoding=True, align_mask=False):
        B_, N, C = x_shape
        _, h, _, C_ = q.shape   # B_ h N C_
        out_list = []
        attn = q * self.scale @ k.transpose(-2, -1)  # B_ h N N

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
            attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
            
        if mask is None:
            attn = self.softmax(attn)
        else:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)    

        if not align_mask:
            soft_v, soft_index = torch.topk(attn, k=self.selected_token_nums, dim=-1)  # B_ h N 2
            soft_v = soft_v.unsqueeze(-2).expand(-1, -1, -1, C_, -1)  # B_ h N C_ 2
            # soft_v = F.softmax(soft_v, dim=-1)
            soft_index = soft_index.unsqueeze(-2).expand(-1, -1, -1, C_, -1)  # B_ h N C_ 2
            for i in range(soft_v.shape[-1]):
                out = torch.gather(v, 2, soft_index[..., i])  # B_ h N C_
                out_list.append(out*soft_v[..., i])
            x = sum(out_list).permute(0, 2, 1, 3).contiguous().reshape(B_, N, C)    
        else:
            mask_vector = q.new_ones(N) * float(-100.) 
            mask_matrix = torch.diag_embed(mask_vector).unsqueeze(0).unsqueeze(0).expand(B_, h, -1, -1)  # B_ h N N  
            attn_ = attn + mask_matrix

            s_v, s_i = torch.topk(attn_, k=self.temporal_length, dim=-1)  # B_ h N tl
            mask = attn_.new_ones(attn_.size()).bool()  # B_ h N N 
            mask.scatter_(-1, s_i, False)
            attn_.masked_fill_(mask, float(-100.))  

            attn_ = self.softmax(attn_) 

            soft_v, soft_index = torch.max(attn_, dim=-1)  # B_ h N
            soft_v = soft_v.unsqueeze(-1).expand(-1, -1, -1, C_)  # B_ h N C_
            soft_index = soft_index.view(B_, -1, N, 1).expand(-1, -1, -1, C_)  # B_ h N C_
            output = torch.gather(v, 2, soft_index)  # B_ h N C_
            x_ = (output*soft_v).permute(0, 2, 1, 3).contiguous().reshape(B_, N, C)
            # x = self.fusion(torch.cat([x_, x], dim=-1))
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C) 

            # fusion
            a = (x + x_).permute(0, 2, 1).contiguous().mean(2)
            a = self.reweight(a).reshape(B_, C, 2).permute(2, 0, 1).contiguous().softmax(dim=0).unsqueeze(2)  # 2 B_ 1 C
            x = x * a[0] + x_ * a[1]

            # x = x_
        
        if self.is_train:
            return x
        else:
            return x, attn_.detach().cpu().numpy()

    def get_position_index(self, window_size):
        """ Get pair-wise relative position index for each token inside the window. """
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """ Get sine position encoding """
        if scale is not None:
            if normalize is False:
                raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=(torch.float32))
        x_embed = not_mask.cumsum(2, dtype=(torch.float32))
        if normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_t = torch.arange(num_pos_feats, dtype=(torch.float32))
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class VSTSRDecoderTransformerBlock(nn.Module):

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=(6, 8, 8), shift_size=(0, 0, 0), 
                 mlp_ratio=2.0, 
                 qkv_bias=True, qk_scale=None, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_checkpoint_attn=False, use_checkpoint_ffn=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn
        assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[2], 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, 
                                    window_size=(self.window_size), 
                                    num_heads=num_heads, 
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn_ = WindowAttention(dim, 
                                     window_size=(self.window_size), 
                                     num_heads=num_heads, 
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop,
                                     proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=(int(dim * mlp_ratio)), act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        if any((i > 0 for i in shift_size)):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = (attn_windows.view)(*(-1, ), *window_size + (C,))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any((i > 0 for i in shift_size)):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)
        return x

    def forward_part2(self, x, attn_kv, mask_matrix):
        B, D, H, W, C = x.shape
        B, D_, H_, W_, C = attn_kv.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        window_size_, shift_size_ = get_window_size((D_, H_, W_), self.window_size, self.shift_size)

        x = self.norm2(x)
        attn_kv = self.norm_kv(attn_kv)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        pad_d1_ = (window_size_[0] - D_ % window_size_[0]) % window_size_[0]
        pad_b_ = (window_size_[1] - H_ % window_size_[1]) % window_size_[1]
        pad_r_ = (window_size_[2] - W_ % window_size_[2]) % window_size_[2]
        attn_kv = F.pad(attn_kv, (0, 0, pad_l, pad_r_, pad_t, pad_b_, pad_d0, pad_d1_), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        if any((i > 0 for i in shift_size)):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        _, Dp_, Hp_, Wp_, _ = attn_kv.shape
        if any((i > 0 for i in shift_size_)):
            shifted_x_ = torch.roll(attn_kv, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask_ = mask_matrix
        else:
            shifted_x_ = attn_kv
            attn_mask_ = None

        x_windows = window_partition(shifted_x, window_size)
        x_windows_ = window_partition(shifted_x_, window_size_)
        attn_windows = self.attn_(x_windows, x_windows_, mask=attn_mask_)
        attn_windows = (attn_windows.view)(*(-1, ), *window_size + (C,))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any((i > 0 for i in shift_size)):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)
        return x

    def forward_part3(self, x):
        return self.drop_path(self.mlp(self.norm3(x)))

    def forward(self, x, attn_kv, mask_matrix, mask_matrix_):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part2, x, attn_kv, mask_matrix_)
        else:
            x = x + self.forward_part2(x, attn_kv, mask_matrix_)

        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part3, x)
        else:
            x = x + self.forward_part3(x)
        return x


class EncoderBlockOnOnetoken(nn.Module):

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=(3, 8, 8), shift_size=(0, 0, 0), 
                 mlp_ratio=2.0, 
                 qkv_bias=True, qk_scale=None, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=None, use_checkpoint_ffn=None,
                 is_train=True,
                 if_unfold=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.is_train = is_train

        self.unfold = if_unfold  
        self.stride_h, self.stride_w = window_size[1], window_size[2]
        self.stride_2 = self.stride_h * self.stride_w
        self.stride = (self.stride_h, self.stride_w)
        
        assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[2], 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)  
        self.attn = rWindowAttention(dim, 
                                    window_size=(self.window_size), 
                                    num_heads=num_heads, 
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop, 
                                    only_one=False,
                                    align_mask=True,
                                    is_train=self.is_train,
                                    norm_layer=norm_layer)
        if if_unfold:
            self.attn1 = rWindowAttention(dim, 
                                    window_size=(self.window_size), 
                                    num_heads=num_heads, 
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop, 
                                    only_one=True,
                                    align_mask=True,
                                    is_train=self.is_train)
            self.proj = nn.Linear(2*dim, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if num_heads > 0:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=(int(dim * mlp_ratio)), act_layer=act_layer, drop=drop)

    def unfold_feature(self, x):
        B, D, H, W, C = x.shape
        x = F.unfold(x.reshape(B*D, H, W, C).permute(0, 3, 1, 2).contiguous(), 
                                kernel_size=(int(2*self.stride_h), int(2*self.stride_w)), 
                                padding=(int(0.5*self.stride_h), int(0.5*self.stride_w)), stride=self.stride)  # _, C*2h*2w, (H/h*W/h)
        x = F.fold(x, output_size=(int(2*H), int(2*W)), 
                                kernel_size=(int(2*self.stride_h), int(2*self.stride_w)), 
                                padding=0, stride=(int(2*self.stride_h), int(2*self.stride_w)))  # _, C, 2H, 2W
        x = F.adaptive_avg_pool2d(x, (H, W))  # _, C, H, W
        return x.reshape(B, D, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # B D H W C

    def flow_alignment(self, x, flow_f=None, flow_b=None):
        B, D, H, W, C = x.shape
        _, F, _, _, _ = flow_f.shape
        assert D ==2*F, "check the value between D and F."
        f1 = x[:, ::2, ...].contiguous()
        f2 = x[:, 1::2, ...].contiguous()

        feat_right = flow_warp(f1.permute(0, 1, 4, 2, 3).reshape(-1, C, H, W), flow_f.permute(0, 1, 3, 4, 2).reshape(-1, H, W, 2), padding_mode='border').reshape(B, D//2, C, H, W)
        feat_left = flow_warp(f2.permute(0, 1, 4, 2, 3).reshape(-1, C, H, W), flow_b.permute(0, 1, 3, 4, 2).reshape(-1, H, W, 2), padding_mode='border').reshape(B, D//2, C, H, W)

        x = torch.cat([feat_left, feat_right], dim=1).view(B, 2, D//2, C, H, W).permute(0, 2, 1, 4, 5, 3).reshape(B, D, H, W, C)

        return x

    def forward_part1(self, x, mask_matrix, flow_f=None, flow_b=None):
        B, D, H, W, C = x.shape
        
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)  

        if flow_f is not None:
            flow_x = self.flow_alignment(x, flow_f, flow_b)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        if any((i > 0 for i in shift_size)):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        if not self.unfold:
            x_windows = window_partition(shifted_x, window_size)  # (B*num_windows, wt*wh*ww, C)
            if self.is_train:
                attn_windows = self.attn(x_windows, mask=attn_mask)
            else:
                attn_windows, attn = self.attn(x_windows, mask=attn_mask)  
        else:  
            x_windows = window_partition(shifted_x, window_size)
            kv_windows = window_partition(self.unfold_feature(shifted_x), window_size)
            if self.is_train:
                attn_windows = self.attn(x_windows, mask=attn_mask)
                attn_windows1 = self.attn1(x_windows, kv_windows, mask=attn_mask)
            else:
                attn_windows, attn = self.attn(x_windows, mask=attn_mask)
                attn_windows1, attn1 = self.attn1(x_windows, kv_windows, mask=attn_mask)      
            attn_windows = self.proj(torch.cat([attn_windows, attn_windows1], dim=-1))        
        attn_windows_ = (attn_windows.view)(*(-1, ), *window_size + (C,))
        shifted_x = window_reverse(attn_windows_, window_size, B, Dp, Hp, Wp)

        if any((i > 0 for i in shift_size)):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        if flow_f is not None:    
            x = x + flow_x

        x = self.drop_path(x)

        if self.is_train:
            return x
        else:
            return x, attn

    def forward_part2(self, x):
        if self.num_heads > 0:
            return self.drop_path(self.mlp(self.norm2(x)))
        else:
            return x

    def forward(self, x, attn_kv, mask_matrix, mask_matrix_, flow_f=None, flow_b=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        if self.is_train:
            x = x + self.forward_part1(x, mask_matrix, flow_f=flow_f, flow_b=flow_b)
            x = x + self.forward_part2(x)
            return x
        else:
            res, attn = self.forward_part1(x, mask_matrix, flow_f=flow_f, flow_b=flow_b)    
            x = x + res
            x = x + self.forward_part2(x)
            return x, attn


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


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


class ResidualBlockNoBN(nn.Module):
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
            # self.init_weights()

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


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

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


class AlignmentTransformer(nn.Module):
    def __init__(self, 
                 dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=(3, 8, 8), shift_size=(0, 0, 0), 
                 mlp_ratio=2.0, 
                 qkv_bias=True, qk_scale=None, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=None, use_checkpoint_ffn=None,
                 is_train=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.is_train = is_train
        
        assert 0 <= self.shift_size[0] < self.window_size[0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[2], 'shift_size must in 0-window_size'

        # self.norm1 = norm_layer(dim)
        self.attn = rWindowAttention(dim, 
                                    window_size=(self.window_size), 
                                    num_heads=num_heads, 
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop, 
                                    only_one=True,
                                    align_mask=True,
                                    is_train=self.is_train)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if num_heads > 0:
            # self.norm2 = norm_layer(dim)
            # self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=(int(dim * mlp_ratio)), act_layer=act_layer, drop=drop)
            # self.resblocks = nn.Linear(2*dim, dim)
            self.resblocks = ResidualBlocksWithInputConv(2*dim, dim, num_blocks=1)
            
    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        if any((i > 0 for i in shift_size)):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)  # (B*num_windows, wt*wh*ww, C)
        if self.is_train:
            attn_windows = self.attn(x_windows, mask=attn_mask)
        else:
            attn_windows, attn = self.attn(x_windows, mask=attn_mask)    
        attn_windows_ = (attn_windows.view)(*(-1, ), *window_size + (C,))
        shifted_x = window_reverse(attn_windows_, window_size, B, Dp, Hp, Wp)

        if any((i > 0 for i in shift_size)):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)

        if self.is_train:
            return x
        else:
            return x, attn   

    # def forward_part2(self, x):
    #     if self.num_heads > 0:
    #         return self.drop_path(self.mlp(self.norm2(x)))
    #     else:
    #         return x

    def forward(self, x, attn_kv, mask_matrix, mask_matrix_):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        B, D, H, W, C = x.shape
        if self.is_train:
            x_align = self.forward_part1(x, mask_matrix)
            x = torch.cat([x, x_align], dim=-1).permute(0, 1, 4, 2, 3).contiguous().reshape(-1, 2*C, H, W)
            x = self.drop_path(self.resblocks(x)).reshape(B, D, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
            # FFN
            # x = self.resblocks(torch.cat([x, x_align], dim=-1))
            # x = x + self.forward_part2(x)
            return x
        else:
            x_align, attn = self.forward_part1(x, mask_matrix)    
            x = torch.cat([x, x_align], dim=-1).permute(0, 1, 4, 2, 3).contiguous().reshape(-1, 2*C, H, W)
            x = self.drop_path(self.resblocks(x)).reshape(B, D, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
            # x = self.resblocks(torch.cat([x, x_align], dim=-1))
            # x = x + self.drop_path(self.forward_part2(x))
            return x, attn


class DecoderLayer(nn.Module):

    def __init__(self, 
                 dim, 
                 input_resolution, 
                 depth, 
                 num_heads, window_size=[2, 4, 4], shift_size=None, 
                 mlp_ratio=2.0, 
                 qkv_bias=False, qk_scale=None, 
                 drop=0.0, attn_drop=0.0, drop_path=0.0, 
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint_attn=False, use_checkpoint_ffn=False,
                 is_train=True,
                 if_unfold=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list((i // 2 for i in window_size)) if shift_size is None else shift_size
        self.is_train = is_train

        self.blocks = nn.ModuleList([EncoderBlockOnOnetoken(dim=dim, 
                                                            input_resolution=input_resolution, 
                                                            num_heads=num_heads, 
                                                            window_size=window_size, shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size, 
                                                            mlp_ratio=mlp_ratio, 
                                                            qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                                            drop=drop, attn_drop=attn_drop, drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path), 
                                                            norm_layer=norm_layer, 
                                                            use_checkpoint_attn=use_checkpoint_attn, use_checkpoint_ffn=use_checkpoint_ffn,
                                                            is_train=self.is_train,
                                                            if_unfold=if_unfold) for i in range(depth)])
        self.flag = False

    def rearrange_shape(self, x):
        B, D, C, H, W = x.shape
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

    def forward(self, x, attn_kv=None, flow_f=None, flow_b=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        B, D, C, H, W = x.shape
        if D % self.window_size[0] != 0:
            x, seq_back = self.rearrange_shape(x)
            B, D, C, H, W = x.shape

        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b d c h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        if attn_kv is not None:
            B, D_, C, H_, W_ = attn_kv.shape
            window_size_, shift_size_ = get_window_size((D_, H_, W_), self.window_size, self.shift_size)
            attn_kv = rearrange(attn_kv, 'b d c h w -> b d h w c')
            Dp_ = int(np.ceil(D_ / window_size_[0])) * window_size_[0]
            Hp_ = int(np.ceil(H_ / window_size_[1])) * window_size_[1]
            Wp_ = int(np.ceil(W_ / window_size_[2])) * window_size_[2]
            attn_mask_ = compute_mask(Dp_, Hp_, Wp_, window_size_, shift_size_, attn_kv.device)
        else:
            attn_mask_ = None     

        for blk in self.blocks:
            if self.is_train:
                x = blk(x, attn_kv, attn_mask, attn_mask_, flow_f=flow_f, flow_b=flow_b)
            else:
                x, _ = blk(x, attn_kv, attn_mask, attn_mask_, flow_f=flow_f, flow_b=flow_b)


        if self.flag:
            x = x[:, seq_back, ...]
            self.flag = False    
            B, D, H, W, C = x.shape

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b d c h w')
        return x


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
