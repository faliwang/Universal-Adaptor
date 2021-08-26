""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, res_add=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        self.res_add = res_add
        if res_add:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.res_add:
            return x1 + x2
        else:
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv_affine(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, config_len, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.double_conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cond_a = nn.Linear(config_len, mid_channels)
        self.cond_b = nn.Linear(config_len, mid_channels)
        self.cond_nl = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, config):
        x = self.double_conv_1(x)
        Batch = config.size(0)
        Ca = self.cond_a(config).view(Batch,-1,1,1)
        Cb = self.cond_b(config).view(Batch,-1,1,1)
        x = self.cond_nl(Ca * x + Cb)
        return self.double_conv_2(x)


class Down_affine(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, config_len):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv_affine(in_channels, out_channels, config_len)
        

    def forward(self, x, config):
        x = self.maxpool(x)
        return self.conv(x, config)


class Up_affine(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, config_len, bilinear=True, res_add=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_affine(in_channels, out_channels, config_len, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_affine(in_channels, out_channels, config_len)
        
        self.res_add = res_add
        if res_add:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv_affine(out_channels, out_channels, config_len)


    def forward(self, x1, x2, config):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.res_add:
            x = x1 + x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x,config)


class OutConv_affine(nn.Module):
    def __init__(self, in_channels, out_channels, config_len):
        super(OutConv_affine, self).__init__()
        self.cond_a = nn.Linear(config_len, in_channels)
        self.cond_b = nn.Linear(config_len, in_channels)
        self.cond_nl = nn.PReLU(num_parameters=1, init=0.25)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, config):
        Batch = config.size(0)
        Ca = self.cond_a(config).view(Batch,-1,1,1)
        Cb = self.cond_b(config).view(Batch,-1,1,1)
        x = self.cond_nl(Ca * x + Cb)
        return self.conv(x)
