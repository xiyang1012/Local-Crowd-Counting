import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from test_config import cfg


class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, pretrained=True):
        super(CrowdCounter, self).__init__()
        
        if model_name == 'CSRNet_LCM':
            from .SCC_Model.CSRNet_LCM import CSRNet_LCM as net
        elif model_name == 'VGG16_LCM':
            from .SCC_Model.VGG16_LCM import VGG16_LCM as net
        elif model_name == 'VGG16_LCM_REG':
            from .SCC_Model.VGG16_LCM_REG import VGG16_LCM_REG as net

        self.CCN = net(pretrained)

        if len(gpus) > 1: # for multi gpu
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda(gpus[0])
        else:   # for one gpu
            self.CCN=self.CCN.cuda()

        self.loss_sum_fn = nn.L1Loss().cuda()

        self.SumLoss = True

    @property
    def loss(self):
        return self.loss_total

    def loss_sum(self):
        return self.loss_sum

    def forward(self, img, gt_map):
        count_map = self.CCN(img)
        gt_map = torch.unsqueeze(gt_map, 1)
        self.loss_total, self.loss_sum = self.build_loss(count_map, gt_map)

        return count_map

    def build_loss(self, count_map, gt_map):
        loss_total = 0.

        if self.SumLoss:
            gt_map_ = gt_map / cfg.LOG_PARA
            kernal3, kernal4, kernal5 = 2, 4, 8

            # filter3 = torch.ones(1, 1, kernal3, kernal3, requires_grad=False).cuda()
            # filter4 = torch.ones(1, 1, kernal4, kernal4, requires_grad=False).cuda()
            filter5 = torch.ones(1, 1, kernal5, kernal5, requires_grad=False).cuda()

            # gt_map_3 = F.conv2d(gt_map, filter3, stride=kernal3)
            # gt_map_4 = F.conv2d(gt_map, filter4, stride=kernal4)
            gt_map_5 = F.conv2d(gt_map_, filter5, stride=kernal5)
            
            loss_sum_all = self.loss_sum_fn(count_map, gt_map_5)

            loss_total += loss_sum_all

        return loss_total, loss_sum_all

    def test_forward(self, img):
        count_map = self.CCN(img)
        return count_map
