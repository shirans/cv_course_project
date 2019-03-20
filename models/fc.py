import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.conv = nn.Conv2d(3, 1, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)
