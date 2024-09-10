import torch.nn as nn
import torch.nn.functional as F

from .sr_backbone_utils import default_init_weights


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for PixelShufflePack.
        """
        default_init_weights(self, 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # print(f'x is {x.shape}')  # x is torch.Size([2, 64, 128, 128])
        x = self.upsample_conv(x)
        # print(f'x1 is {x.shape}')  # x1 is torch.Size([2, 256, 128, 128])
        x = F.pixel_shuffle(x, self.scale_factor)
        # print(f'ox is {x.shape}')  # ox is torch.Size([2, 64, 256, 256])
        return x
