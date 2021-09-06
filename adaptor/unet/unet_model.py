""" Full assembly of the parts to form the complete network """

from dataset import convert_config
import torch.nn.functional as F
import torch.nn as nn

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, num_layers=4, base=16, bilinear=True, res_add=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.res_add = res_add
        self.num_layers = num_layers

        self.inc = DoubleConv(n_channels, base)
        self.down_layers = nn.ModuleList(
            [Down(base*(2**i), base*2*(2**i)) for i in range(num_layers-1)]
        )
        factor = 2 if bilinear else 1
        self.final_down = Down(base//2 *(2**num_layers), base*(2**num_layers) // factor)
        self.up_layers = nn.ModuleList(
            [Up(base*(2**i), base//2*(2**i) // factor, bilinear, res_add) for i in range(num_layers, 1, -1)]
        )
        self.final_up = Up(base*2, base, bilinear, res_add)
        self.outc = OutConv(base, n_channels)

        # self.down1 = Down(16, 32)
        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(128, 256 // factor)
        # self.up1 = Up(256, 128 // factor, bilinear, res_add)
        # self.up2 = Up(128, 64 // factor, bilinear, res_add)
        # self.up3 = Up(64, 32 // factor, bilinear, res_add)
        # self.up4 = Up(32, 16, bilinear, res_add)
        # self.outc = OutConv(16, n_channels)

    def forward(self, x):
        x_list = []
        x = self.inc(x)
        x_list.append(x)
        for i in range(self.num_layers-1):
            x = self.down_layers[i](x)
            x_list.append(x)
        x_up = self.final_down(x)
        for i in range(self.num_layers-1):
            x_up = self.up_layers[i](x_up, x_list[self.num_layers-1-i])
        x_up = self.final_up(x_up, x_list[0])
        logits = self.outc(x_up)
        
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        return logits


class UNet_affine(nn.Module):
    def __init__(self, n_channels, config_len, num_layers=4, base=16, bilinear=True, res_add=False):
        super(UNet_affine, self).__init__()
        self.n_channels = n_channels
        self.config_len = config_len
        self.bilinear = bilinear if not res_add else False
        self.res_add = res_add
        self.num_layers = num_layers

        self.inc = DoubleConv_affine(n_channels, base, config_len)
        self.down_layers = nn.ModuleList(
            [Down_affine(base*(2**i), base*2*(2**i), config_len) for i in range(num_layers-1)]
        )
        factor = 2 if bilinear else 1
        self.final_down = Down_affine(base//2 *(2**num_layers), base*(2**num_layers) // factor, config_len)
        self.up_layers = nn.ModuleList(
            [Up_affine(base*(2**i), base//2*(2**i) // factor, config_len, bilinear, res_add) for i in range(num_layers, 1, -1)]
        )
        self.final_up = Up_affine(base*2, base, config_len, bilinear, res_add)
        self.outc = OutConv_affine(base, n_channels, config_len)

    def forward(self, x, config):
        x_list = []
        x = self.inc(x, config)
        x_list.append(x)
        for i in range(self.num_layers-1):
            x = self.down_layers[i](x, config)
            x_list.append(x)
        x_up = self.final_down(x, config)
        for i in range(self.num_layers-1):
            x_up = self.up_layers[i](x_up, x_list[self.num_layers-1-i], config)
        x_up = self.final_up(x_up, x_list[0], config)
        logits = self.outc(x_up, config)
        return logits
