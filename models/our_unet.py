import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.comp1_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.comp1_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.comp1_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.comp2_conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.comp2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.comp2_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.comp3_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.comp3_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.comp3_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.comp4_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.comp4_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.comp4_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.comp5_conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2)
        self.comp5_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)

        self.expan0_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.expan1_conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=2)
        self.expan1_conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.expan1_upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.expan2_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.expan2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.expan2_upsample = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        self.expan3_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.expan3_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.expan3_upsample = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)

        self.expan4_conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.expan4_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)

        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        # contraction path 1
        compaction_1_input = x                                          # input is 128*128*3
        compaction1_out = F.relu(self.comp1_conv1(compaction_1_input))
        compaction1_out = F.relu(self.comp1_conv2(compaction1_out))
        compaction1_size = compaction1_out.size()                       # should be 128*128*16

        compaction1_pooled_out, indices1 = self.comp1_pool(compaction1_out)  # should be 64*64*16
        # contraction path 2

        compaction2_input = compaction1_pooled_out                      # should be 64*64*16
        compaction2_out = F.relu(self.comp2_conv1(compaction2_input))
        compaction2_out = F.relu(self.comp2_conv2(compaction2_out))
        compaction2_size = compaction2_out.size()                       # should be 64*64*32

        compaction2_pooled_out, indices2 = self.comp2_pool(compaction2_out)  # should be 32*32*32

        # contraction path 3

        compaction3_input = compaction2_pooled_out                      # should be 32*32*32
        compaction3_out = F.relu(self.comp3_conv1(compaction3_input))
        compaction3_out = F.relu(self.comp3_conv2(compaction3_out))
        compaction3_size = compaction3_out.size()                       # should be 32*32*64

        compaction3_pooled_out, indices3 = self.comp3_pool(compaction3_out)  # should be 16*16*64

        # contraction path 4
        compaction4_input = compaction3_pooled_out                      # should be 16*16*64
        compaction4_out = F.relu(self.comp4_conv1(compaction4_input))
        compaction4_out = F.relu(self.comp4_conv2(compaction4_out))
        compaction4_size = compaction4_out.size()                       # should be 16*16*128

        compaction4_pooled_out, indices4 = self.comp4_pool(compaction4_out)  # should be 8*8*128

        # contraction path 5
        compaction5_input = compaction4_pooled_out                      # should be 8*8*256
        compaction5_out = F.relu(self.comp5_conv1(compaction5_input))
        compaction5_out = F.relu(self.comp5_conv2(compaction5_out))
        compaction5_size = compaction5_out.size()                       # should be 8*8*256

        ###### Start upsampling ######
        upsampled0_out = self.expan0_upsample(compaction5_out)        # should be 16*16*128

        # expansion path 1
        expansion1_input = torch.cat((upsampled0_out, compaction4_out), dim=1) # should be 16*16*256                      # should be 16*16*256
        expansion1_out = F.relu(self.expan1_conv1(expansion1_input))
        expansion1_out = F.relu(self.expan1_conv2(expansion1_out))
        expansion1_size = expansion1_out.size()                       # should be 16*16*128

        upsampled1_out = self.expan1_upsample(expansion1_out)           # should be 32*32*64

        # expansion path 2
        expansion2_input = torch.cat((upsampled1_out, compaction3_out), dim=1)  # should be 32*32*128
        expansion2_out = F.relu(self.expan2_conv1(expansion2_input))
        expansion2_out = F.relu(self.expan2_conv2(expansion2_out))
        expansion2_size = expansion2_out.size()                         # should be 32*32*64

        upsampled2_out = self.expan2_upsample(expansion2_out)           # should be 64*64*32

        # expansion path 3
        expansion3_input = torch.cat((upsampled2_out, compaction2_out), dim=1)  # should be 64*64*64
        expansion3_out = F.relu(self.expan3_conv1(expansion3_input))
        expansion3_out = F.relu(self.expan3_conv2(expansion3_out))
        expansion3_size = expansion3_out.size()                         # should be 64*64*32

        upsampled3_out = self.expan3_upsample(expansion3_out)  # should be 128*128*16

        # expansion path 4
        expansion4_input = torch.cat((upsampled3_out, compaction1_out), dim=1)  # should be 128*128*32
        expansion4_out = F.relu(self.expan4_conv1(expansion4_input))
        expansion4_out = F.relu(self.expan4_conv2(expansion4_out))
        expansion4_size = expansion4_out.size()                                 # should be 128*128*16

        final = self.final_conv(expansion4_out)
        return final