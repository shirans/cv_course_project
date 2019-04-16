import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        in_channels = 3 # Number of channels in the input image
        out_channels = 1 # Number of channels produced by the convolution
        kernel_size = 3 # Size of the convolving kernel
        stride = 1 # Stride of the convolution. Default: 1
        padding = 1 # Zero-padding added to both sides of the input. Default: 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(1, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
