import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from train_config import cfg

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, pretrained=True):
        super(CrowdCounter, self).__init__()        

        if model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet_DM':
            from .SCC_Model.CSRNet_DM import CSRNet_DM as net
        elif model_name == 'VGG16_DM':
            from .SCC_Model.VGG16_DM import VGG16_DM as net
            
        self.CCN = net(pretrained)

        if len(gpus) > 1: # for multi gpu
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda(gpus[0])
        else:   # for one gpu
            self.CCN=self.CCN.cuda()

        self.loss_fn = nn.MSELoss().cuda()

    @property
    def loss(self):
        return self.loss_total

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        gt_map = torch.unsqueeze(gt_map, 1)
        self.loss_total = self.build_loss(density_map, gt_map)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss_total = 0.
        loss_total += self.loss_fn(density_map, gt_data)

        return loss_total

    def test_forward(self, img):                               
        density_map = self.CCN(img)
        
        return density_map
