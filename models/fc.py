import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        in_channels = 3
        out_channels = 1
        kernel_size = 3
        stride = 1
        padding = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)
