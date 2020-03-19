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

data_root = os.path.join(cfg_data.DATA_PATH, 'test')
mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])

''' result save path '''
exp_name = './test_res'
#os.path.join('./test_res', model_net + '_' + data_mode)
if not os.path.exists(exp_name):
    os.mkdir(exp_name)
exp_name = os.path.join('./test_res', data_mode + '_' + model_net)
if not os.path.exists(exp_name):
    os.mkdir(exp_name)
if not os.path.exists(exp_name + '/pred'):
    os.mkdir(exp_name + '/pred')
if not os.path.exists(exp_name + '/gt'):
    os.mkdir(exp_name + '/gt')


def main():
    print(data_root + '/img/')
    file_list = [filename for root, dirs, filename in os.walk(data_root + '/img/')][0]
    file_list.sort()
    test(file_list, model_path)

def test(file_list, model_path):
    if 'LCM' in model_net:
        from models.CC_LCM import CrowdCounter
    elif 'DM' in model_net:
        from models.CC_DM import CrowdCounter
    net = CrowdCounter(cfg_GPU_ID, model_net, pretrained=False)
    ''' single-gpu / multi-gpu trained model '''
    if len(cfg_GPU_ID) == 1:
        net.load_state_dict(torch.load(model_path))
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

        # read img and den
        imgname = os.path.join(data_root, 'img', filename)
        filename_no_ext = filename.split('.')[0]
        denname = os.path.join(data_root, 'den', filename_no_ext+'.csv')

        den = pd.read_csv(denname, sep=',', header=None).values
        den_map = den.astype(np.float32, copy=False)
        if 'LCM' in model_net:
            lcm_map = convert_DM_to_LCM(den_map)

        # model testing
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)

        ''' MAE/MSE'''
        gt_value = np.sum(den_map)
        if 'LCM' in model_net:
            pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :])
        elif 'DM' in model_net:
            pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :]) / cfg.LOG_PARA
            
        mae = abs(gt_value - pred_value)
        mse = (gt_value - pred_value) * (gt_value - pred_value)
        print("   mae:{:.2f}".format(mae))
        MAE.append(mae)
        MSE.append(mse)

        ''' save pred/gt .csv '''
        csv_path = os.path.join(exp_name, 'gt', filename_no_ext + '.csv')
        if 'LCM' in model_net:
            data_den = pd.DataFrame(lcm_map)
        elif 'DM' in model_net:
            data_den = pd.DataFrame(den_map)
        data_den.to_csv(csv_path, header=False, index=False)

        csv_path = os.path.join(exp_name, 'pred', filename_no_ext + '.csv')
        data_den = pd.DataFrame(pred_map.squeeze().cpu().numpy())
        data_den.to_csv(csv_path, header=False, index=False)

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
        plt.savefig(os.path.join(exp_name, filename_no_ext + '_pred_' + str(round(float(pred_value), 2)) + '.jpg'),
                    bbox_inches='tight', pad_inches=0, dpi=150)

        ''' gt counting map '''
        den_frame = plt.gca()
        if 'LCM' in model_net:
              plt.imshow(lcm_map)
        if 'DM' in model_net:
              plt.imshow(den_map, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        plt.savefig(os.path.join(exp_name, filename_no_ext + '_gt_' + str(int(gt_value)) + '.jpg'),
                    bbox_inches='tight', pad_inches=0, dpi=150)

    avg_MAE = sum(MAE)/index
    avg_MSE = math.sqrt( sum(MSE)/index )
    print("test result: MAE:{:2f}, MSE:{:2f}".format(avg_MAE, avg_MSE))


def load_gpus_to_gpu(model, model_path):
    ''' convert multi-gpu trained model to sigle model '''
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    # create new OrderedDict that does not contain 'module.'
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[0:3] + k[10:]  # remove 'module.'
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
    return model

def convert_DM_to_LCM(den_map, patch_size=64):
    ''' input density map(numpy)
        output local counting map(numpy) '''
    den_map = torch.from_numpy(den_map)
    den_map = den_map.unsqueeze(0)    # 2D to 4D
    den_map = den_map.unsqueeze(0)
    filter = torch.ones(1, 1, patch_size, patch_size, requires_grad=False)
    lc_map = F.conv2d(den_map, filter, stride=patch_size)
    lc_map = lc_map.squeeze()
    lc_map = lc_map.numpy()
    return lc_map

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:', total_num)
    print('Trainable:', trainable_num)
    return 0

if __name__ == '__main__':
    main()
