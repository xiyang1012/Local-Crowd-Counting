from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from train_config import cfg

net = cfg.NET
if '18' in net:
    resnet_layer = 18
elif '50' in net:
    resnet_layer = 50


class ResNet_LCM(nn.Module):
    def __init__(self, load_weights=False, stage_num=[1,1,1], count_range=100, lambda_i=1., lambda_k=1.):
        super(ResNet_LCM, self).__init__()

        # cfg
        self.stage_num = stage_num
        self.lambda_i = lambda_i
        self.lambda_k = lambda_k
        self.count_range = count_range
        self.multi_layer = True         # layer5 4 3 in MRM module
        self.multi_fuse = False          # SAM module
        self.soft_interval = False       # ASIM module
        self.ratio = 5

        # backbone
        if resnet_layer == 18:
            self.resnet = ResNet18()
            self.base_ch = 128
        elif resnet_layer == 50:
            self.resnet = ResNet50()
            self.base_ch = 512

        # MRM
        self.count_layer5 = Count_layer(inplanes=self.base_ch*4, pool=2)
        self.layer5_p = nn.Sequential(
            nn.Conv2d(128, self.stage_num[0], kernel_size=1),
            nn.ReLU() )

        if self.multi_layer:
            self.count_layer4 = Count_layer(inplanes=self.base_ch*2, pool=4)
            self.layer4_p = nn.Sequential(
                nn.Conv2d(128, self.stage_num[1], kernel_size=1),
                nn.Tanh() )

            self.count_layer3 = Count_layer(inplanes=self.base_ch, pool=8)
            self.layer3_p = nn.Sequential(
                nn.Conv2d(128, self.stage_num[2], kernel_size=1),
                nn.Tanh() )

        if load_weights:
            self._initialize_weights()
            
            if resnet_layer == 18:
                pretrain_path = './models/Pretrain_Model/resnet18-5c106cde.pth'
                print('Loading resnet18-5c106cde.pth')
            elif  resnet_layer == 50:
                pretrain_path = './models/Pretrain_Model/resnet50-19c8e357.pth'
                print('Loading resnet50-19c8e357.pth')

            pretrained_dict = torch.load(pretrain_path)
            model_dict = self.resnet.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.resnet.load_state_dict(model_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.resnet(x)
        [x3, x4, x5] = x

        # ------------ MRM ------------
        # layer5
        x5_= self.count_layer5(x5)
        p5 = self.layer5_p(x5_)
        
        if self.multi_layer:
            # layer4
            x4_ = self.count_layer4(x4)
            x4_ = x5_ - x4_
            p4 = self.layer4_p(x4_)

            # layer3
            x3_ = self.count_layer3(x3)
            x3_ = x4_ - x3_
            p3 = self.layer3_p(x3_)

        regress_count_map = p5 * self.count_range
        
        if self.multi_layer:
            regress_count_map += p4 * self.count_range / self.ratio + \
                                 p3 * self.count_range / (self.ratio * self.ratio)
        
        return regress_count_map


class Count_layer(nn.Module):
    def __init__(self, inplanes=512, pool=2):
        super(Count_layer, self).__init__()
        outplanes = 128
        self.avgpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=pool, stride=pool, padding=0, ceil_mode=True),
        )
        self.maxpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool, stride=pool, padding=0, ceil_mode=True),
        )
        self.conv1x1= nn.Sequential(
            nn.Conv2d(outplanes*2, outplanes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_avg = self.avgpool_layer(x)
        x_max = self.maxpool_layer(x)

        x = torch.cat([x_avg, x_max], 1)
        x = self.conv1x1(x)
        return x


# ----------------------------- ResNet -----------------------------#

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
# ResNet18 34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
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
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResNet50 101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
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
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

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
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        multi_out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)         
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        multi_out.append(x)
        
        x = self.layer3(x)
        multi_out.append(x)
        
        x = self.layer4(x)
        multi_out.append(x)
        
        return multi_out
        
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


if __name__ == '__main__':
    '''
    net = ResNet_LCM_REG()
    out = net(torch.randn(1,3,512,512))
    print(out.size())
    '''

    net = ResNet18()
    out = net(torch.randn(1,3,512,512))
    for i in out:
        print(i.size())
    
