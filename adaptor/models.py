import torch
import torch.nn as nn
import numpy as np

from .unet import UNet, UNet_affine
from .ResNet import ResNet, BasicBlock, Bottleneck
from .R2AttU_Net import R2AttU_Net, R2AttU_Net_config

class Refiner_UNet(nn.Module):
    def __init__(self, n_channels=1, num_layers=4, base=16, bilinear=True, res_add=False):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, num_layers=num_layers, base=base, bilinear=bilinear, res_add=res_add)
    
    def forward(self, input, config):
        input_w = torch.unsqueeze(input, -1)
        input_w = torch.transpose(input_w, 1, 3)
        # B * chan * n_mels * len
        output = self.unet(input_w)
        output = torch.transpose(output, 1, 3)
        output = torch.squeeze(output, -1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


class Refiner_UNet_with_config(nn.Module):
    def __init__(self, n_channels=1, num_layers=4, base=16, bilinear=True, res_add=False):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, num_layers=num_layers, base=base, bilinear=bilinear, res_add=res_add)
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        config = torch.unsqueeze(config, -1)
        config = torch.unsqueeze(config, -1)
        config = config.repeat(1, 1, num_mels, length)
        input_u = torch.cat([input_w, config], 1)
        output = self.unet(input_u)
        output = self.conv(output) + input_w
        output = torch.squeeze(output, 1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


class Refiner_UNet_affine(nn.Module):
    def __init__(self, n_channels=1, config_len=27, num_layers=4, base=16, bilinear=True, res_add=False):
        super().__init__()
        self.unet = UNet_affine(n_channels=n_channels, config_len=config_len, num_layers=num_layers, base=base, bilinear=bilinear, res_add=res_add)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        output = self.unet(input_w, config) + input_w
        output = torch.squeeze(output, 1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


class Refiner_ResNet_with_config(nn.Module):
    def __init__(self, 
        n_channels=1, 
        block='bottleneck', 
        layers=[3, 4, 6, 3], 
        planes=[64,64,128,128], 
        block_resadd=False, 
        output_layer=False, 
        groups=32, 
        width_per_group=4
    ):
        super().__init__()
        if block == 'bottleneck':
            block = Bottleneck
        elif block == 'basic':
            block = BasicBlock
        else:
            raise NotImplementedError(f"we did not implement {block} block in ResNet")
        self.resnet = ResNet(input_chan=n_channels, block=block, layers=layers, planes=planes, block_resadd=block_resadd, output_layer=output_layer, groups=groups, width_per_group=width_per_group)
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        config = torch.unsqueeze(config, -1)
        config = torch.unsqueeze(config, -1)
        config = config.repeat(1, 1, num_mels, length)
        input_u = torch.cat([input_w, config], 1)
        _outputs = self.resnet(input_u)
        outputs = []
        for _output in _outputs:
            _output = self.conv(_output) + input_w
            _output = torch.squeeze(_output, 1)
            assert input.size() == _output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), _output.size())
            outputs.append(_output)
        return outputs


class Refiner_R2AttUNet_with_config(nn.Module):
    def __init__(self, 
        n_channels=1, 
        config_len=27,
        t=2,
        layers=5, 
        base=64,
        resadd=False, 
    ):
        super().__init__()
        self.unet = R2AttU_Net_config(img_ch=n_channels,output_ch=1,config_len=config_len,t=t,layers=layers,base=base,resadd=resadd)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        output = self.unet(input_w, config) + input_w
        output = torch.squeeze(output, 1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output


class Refiner_R2AttUNet(nn.Module):
    def __init__(self, 
        n_channels=1, 
        t=2,
        layers=5, 
        base=64,
        resadd=False, 
    ):
        super().__init__()
        self.unet = R2AttU_Net(img_ch=n_channels,output_ch=1,t=t,layers=layers,base=base,resadd=resadd)
    
    def forward(self, input, config):
        batch, num_mels, length = input.shape
        input_w = torch.unsqueeze(input, 1)
        # B * chan * n_mels * len
        config = torch.unsqueeze(config, -1)
        config = torch.unsqueeze(config, -1)
        config = config.repeat(1, 1, num_mels, length)
        input_u = torch.cat([input_w, config], 1)
        output = self.unet(input_u)
        output = torch.squeeze(output, 1)
        assert input.size() == output.size(), "shape should be same after refine: input {} & output {}".format(input.size(), output.size())
        return output

