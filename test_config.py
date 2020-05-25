import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ------------------------------TEST------------------------
# if train batch=1, use only one GPU!!
__C.GPU_ID = [0] 		    # sigle gpu: [0], [1] ...; multi gpus: [0,1]

__C.NET = 'VGG16_LCM_REG' 	# net selection as train_config.py

__C.DATASET = 'QNRF' 	    # SHHA, SHHB, QNRF, UCF50

__C.SHHBPATCHMAX = 30.
__C.SHHAPATCHMAX = 100.
__C.QNRFPATCHMAX = 100.
__C.CC50PATCHMAX = 100.
                            # testing model path
__C.MODEL_PATH = './exp/QNRF_mae_86.61_mse_152.19.pth'

__C.LOG_PARA = 100.

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX
