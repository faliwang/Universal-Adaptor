import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def trans_conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def trans_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    if stride > 1: output_padding = 1
    else: output_padding = 0
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, output_padding=output_padding, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.transconv1 = trans_conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.transconv2 = trans_conv3x3(planes, inplanes, stride)
        self.bn4 = norm_layer(inplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.transconv1(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.transconv2(out)
        out = self.bn4(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        resadd: bool = False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.transconv1 = trans_conv1x1(planes * self.expansion, width)
        self.bn4 = norm_layer(width)
        self.transconv2 = trans_conv3x3(width, width, stride, groups, dilation)
        self.bn5 = norm_layer(width)
        self.transconv3 = trans_conv1x1(width, inplanes)
        self.bn6 = norm_layer(inplanes)
        self.downsample = downsample
        self.stride = stride
        self.resadd = resadd

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2(out)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out2 = self.conv3(out1)
        out2 = self.bn3(out2)
        out2 = self.relu(out2)

        out3 = self.transconv1(out2)
        out3 = self.bn4(out3)
        out3 = self.relu(out3)
        if self.resadd: out3 = out3 + out1

        out4 = self.transconv2(out3)
        out4 = self.bn5(out4)
        out4 = self.relu(out4)
        if self.resadd: out4 = out4 + out

        out5 = self.transconv3(out4)
        out5 = self.bn6(out5)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out5 = self.relu(out5)
        out5 = out5 + identity

        return out5


class ResNet(nn.Module):

    def __init__(
        self,
        input_chan: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        planes: List[int],
        block_resadd: bool = False,
        output_layer: bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_chan = input_chan
        self.inplanes = 64
        self.dilation = 1
        self.planes = planes
        self.layers = layers
        self.output_layer = output_layer
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False for _ in range(len(layers))]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.input_chan, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        assert len(planes) == len(layers), f'The len of planes {planes} and layers {layers} should be the same'
        self.layers = nn.ModuleList(
            [self._make_layer(block, planes[i], layers[i], resadd=block_resadd, stride=1, dilate=replace_stride_with_dilation[i]) for i in range(len(layers))]
        )
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 64, layers[1], stride=1,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 128, layers[2], stride=1,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=1,
        #                                dilate=replace_stride_with_dilation[2])
        self.trans_conv1 = nn.ConvTranspose2d(self.inplanes, self.input_chan, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn2 = norm_layer(self.input_chan)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    resadd: bool = False, stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, resadd))
        # self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, resadd=resadd))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        Xs = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.output_layer and i != len(self.layers) - 1:
                _x = self.trans_conv1(x)
                _x = self.bn2(_x)
                Xs.append(_x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.trans_conv1(x)
        x = self.bn2(x)
        Xs.append(x)

        return Xs

    def forward(self, x: Tensor):
        return self._forward_impl(x)

