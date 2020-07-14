import os
import cv2
import math
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as standard_transforms

from misc.utils import *
from test_config import cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


''' prepare model config '''
model_net = cfg.NET
model_path = cfg.MODEL_PATH

cfg_GPU_ID = cfg.GPU_ID
torch.cuda.set_device(cfg_GPU_ID[0])
torch.backends.cudnn.benchmark = True


''' prepare data config '''
data_mode = cfg.DATASET
if data_mode is 'SHHA':
    from datasets.SHHA.setting import cfg_data
elif data_mode is 'SHHB':
    from datasets.SHHB.setting import cfg_data
elif data_mode is 'QNRF':
    from datasets.QNRF.setting import cfg_data
elif data_mode is 'UCF50':
    from datasets.UCF50.setting import cfg_data
    val_index = cfg_data.VAL_INDEX
    
mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])


image_dir = './demo_image'


def main():
    file_list = [filename for root, dirs, filename in os.walk(image_dir)][0]
    file_list.sort()
    print(file_list)
    test(file_list, model_path)


def test(file_list, model_path):
    ''' model '''
    if 'LCM' in model_net:
        from models.CC_LCM import CrowdCounter
    elif 'DM' in model_net:
        from models.CC_DM import CrowdCounter
    net = CrowdCounter(cfg_GPU_ID, model_net, pretrained=False)
    
    ''' single-gpu / multi-gpu trained model '''
    if len(cfg_GPU_ID) == 1:
        net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        load_gpus_to_gpu(net, model_path)
    net.cuda()
    net.eval()

    index = 0
    MAE = []
    MSE = []
    for filename in file_list:
        index += 1
        print(index, filename)

        # read img
        imgname = os.path.join(image_dir, filename)
        
        # model testing
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)
            
        ''' MAE/MSE'''
        if 'LCM' in model_net:
            pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :])
        elif 'DM' in model_net:
            pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :]) / cfg.LOG_PARA 
        print("count is:", pred_value)
        
        
        ''' pred counting map '''
        den_frame = plt.gca()
        image = pred_map.cpu().data.numpy()[0, 0, :, :]
        if 'LCM' in model_net:
            plt.imshow(image)
        if 'DM' in model_net:
            image = cv2.resize(image, (image.shape[1]*8, image.shape[0]*8))
            plt.imshow(image, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        
        save_dir = os.path.join(image_dir, 'result')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, filename.split('.')[0] + '_predmap_' + str(int(pred_value + 0.5)) + '.jpg'),
                    bbox_inches='tight', pad_inches=0, dpi=150)            

        ''' pred image '''
        text = "count:" + str(int(pred_value + 0.5))
        img_cv = cv2.imread(imgname)
        cv2.putText(img_cv, text, (10,30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,255), 2)
        cv2.imwrite(os.path.join(save_dir, filename.split('.')[0] + '_predcount_' + str(int(pred_value + 0.5)) + '.jpg'), img_cv)
            
            
if __name__ == '__main__':
    
    main()