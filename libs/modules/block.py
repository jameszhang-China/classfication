import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, bn=True, act=nn.ReLU(inplace=False)):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = act if act else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.act(x) if self.act is not None else x
        return x

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=nn.ReLU(inplace=False)):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, bn=True, act=act)
        self.conv2 = Conv(out_channels, out_channels, 3, stride=1, padding=1, bias=False, bn=True, act=None)
        self.downsample = None
        self.act = act
        if stride != 1 or in_channels!= out_channels:
            self.downsample = Conv(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, bn=True, act=None)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=nn.ReLU(inplace=False), first = False):
        super(ResNetBottleneckBlock, self).__init__()
        if not first and (in_channels != out_channels or stride != 1):
            Warning("ResNetBottleneckBlock: in_channels != out_channels")
        c1 = out_channels//4
        self.conv1 = Conv(in_channels, out_channels=c1, kernel_size=1, stride=stride, padding=0, bias=False, bn=True, act=act)
        self.conv2 = Conv(c1, out_channels=c1, kernel_size=3, stride=1, padding=1, bias=False, bn=True, act=act)
        self.conv3 = Conv(c1, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, bn=True, act=None)
        if first:
            self.downsample = Conv(in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False, bn=True, act=None)
        if act is not None:
            self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += (self.downsample(identity) if hasattr(self, 'downsample') else identity)
        out = self.act(out)
        return out
    
class ResNetBasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=1, stride=1):
        super(ResNetBasicLayer, self).__init__()
        self.layer = self._make_layer(in_channels, out_channels, block_num, stride)

    def _make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(ResNetBasicBlock(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(ResNetBasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class ResNetBottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=1, stride=1):
        super(ResNetBottleneckLayer, self).__init__()
        self.layer = self._make_layer(in_channels, out_channels, block_num, stride)

    def _make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(ResNetBottleneckBlock(in_channels, out_channels, stride=stride, first=True))
        for i in range(1, block_num):
            layers.append(ResNetBottleneckBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, self.d)

