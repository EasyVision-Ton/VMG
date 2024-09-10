import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init


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
        if res_scale == 1.0:
            self.init_weights()

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

# DWconv
def Conv3x3ReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
           
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
          
            nn.ReLU6(inplace=True)
        )

# PW
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
  
    def __init__(self, in_channels=64, out_channels=None, expansion_factor=4, stride=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # in_channels = in_channels
        out_channels = out_channels or in_channels
        mid_channels = int(in_channels * expansion_factor)

   
        self.bottleneck = nn.Sequential(
           
            Conv1x1ReLU(in_channels, mid_channels),
          
            Conv3x3ReLU(mid_channels, mid_channels, stride, groups=mid_channels),
          
            Conv1x1(mid_channels, out_channels)
        )

    def forward(self, x):
        B, D, H, W, C = x.shape
        x_short = x.clone()
        x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(-1, C, H, W)
        x = self.bottleneck(x)
      
        x = x.reshape(B, D, C, H, W).permute(0, 1, 3, 4, 2).contiguous()
        return x + x_short

class Multi_MBConv(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, stride=1, num_blocks=10):
        super().__init__()

        main = []
        main.append(
            make_layer(InvertedResidual, num_blocks, in_channels=embed_dim, expansion_factor=expansion_factor, stride=stride)
        )

        self.main = nn.Sequential(*main)

    def forward(self, x):
        B, T, H, W, C = x.shape
        res = self.main(x)
        return res    
