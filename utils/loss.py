import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss_original(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss_original, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-6, if_aux_loss=False, aux_ratio=0.005):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.if_aux_l = if_aux_loss
        if if_aux_loss:
            self.aux_loss = EdgeLoss(eps=eps)
            self.aux_ratio = aux_ratio  # 20231231:0.05 is hard to converage

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
       
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))

        if self.if_aux_l:
            aux = self.aux_loss(x, y)
            loss = loss + self.aux_ratio * aux
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        # if torch.cuda.is_available():
            # self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss(eps=eps, if_aux_loss=False)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel.to(img.device), groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # x = torch.clamp(x + 0.5, min = 0,max = 1)
        # y = torch.clamp(y + 0.5, min = 0,max = 1)
        B, T, C, H, W = x.shape
        loss_l = []
        x_tuple = torch.chunk(x, T, 1)
        y_tuple = torch.chunk(y, T, 1)
        for x_i, y_i in zip(x_tuple, y_tuple):
            loss = self.loss(self.laplacian_kernel(x_i.squeeze(1)), self.laplacian_kernel(y_i.squeeze(1)))
            loss_l.append(loss)
        loss = sum(loss_l) / T    
        return loss    
