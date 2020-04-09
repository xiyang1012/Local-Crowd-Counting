import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *


class VGG16_LCM(nn.Module):
    def __init__(self, load_weights=True):
        super(VGG16_LCM, self).__init__()

        self.layer5 = self.VGG_make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                                            512, 512, 512, 'M', 512, 512, 512, 'M'])

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.AvgPool2d(2, 2),
        )

        if load_weights:
            mod = models.vgg16(pretrained=False)
            pretrain_path = './models/Pretrain_Model/vgg16-397923af.pth'
            mod.load_state_dict(torch.load(pretrain_path))
            print("loaded pretrain model: " + pretrain_path)

            self._initialize_weights()
            self.layer5.load_state_dict(mod.features[0:31].state_dict())

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
        x = self.layer5(x)
        x = self.reg_layer(x)

        return x

    def VGG_make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=1):
        d_rate = dilation
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
