import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        # in_channels = 3 # Number of channels in the input image
        # out_channels = 1 # Number of channels produced by the convolution
        channels1 = 16
        channels2 = 32
        # kernel_size = 3 # Size of the convolving kernel
        stride = 1 # Stride of the convolution. Default: 1
        padding = 2 # Zero-padding added to both sides of the input. Default: 0

        #layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels1, kernel_size=3,
                               stride=stride, padding=padding)
        nn.init.xavier_uniform(self.conv1.weight)  # Xaviers Initialisation

        # max pooling 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)


        # layer 2
        self.conv2 = nn.Conv2d(in_channels=channels1, out_channels=channels2, kernel_size=3,
                               stride=stride, padding=1)
        nn.init.xavier_uniform(self.conv2.weight)  # Xaviers Initialisation

        # max pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)


        # layer 3
        self.deconv1=nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=3,padding=1) ##
        nn.init.xavier_uniform(self.deconv1.weight)

        # max unpooling
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # layer 4
        self.deconv2=nn.ConvTranspose2d(in_channels=48,out_channels=1,kernel_size=3,padding=2)

        nn.init.xavier_uniform(self.deconv2.weight)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2)




    def forward(self, x):
        out1 = F.relu(self.conv1(x))




        x = F.relu(self.conv1(x))
        return self.conv2(x)