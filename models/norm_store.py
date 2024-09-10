import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, if_mean=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.cal_mean = if_mean
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def pipeline(self, x):
        if self.cal_mean:
            x -= torch.mean(x, dim=-1, keepdim=True)
        output = self._norm(x).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output
    
    def forward(self, x):
        if self.dim == x.shape[-1]:
            return self.pipeline(x)
        elif self.dim == x.shape[2] and len(x.shape) == 5:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            return self.pipeline(x).permute(0, 1, 4, 2, 3).contiguous()
        elif self.dim == x.shape[1] and len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            return self.pipeline(x).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception('Please input correct choice of appropriate Tensor.')
    

class LayerNormTranspose(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, if_mean=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.cal_mean = if_mean
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def pipeline(self, x):
        if self.cal_mean:
            x -= torch.mean(x, dim=-1, keepdim=True)
        output = self._norm(x).type_as(x)
        if self.weight is not None:
            output = output * self.weight + self.bias
        return output
    
    def forward(self, x):
        if self.dim == x.shape[-1]:
            return self.pipeline(x)
        elif self.dim == x.shape[2] and len(x.shape) == 5:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            return self.pipeline(x).permute(0, 1, 4, 2, 3).contiguous()
        elif self.dim == x.shape[1] and len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            return self.pipeline(x).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception('Please input correct choice of appropriate Tensor.')
   