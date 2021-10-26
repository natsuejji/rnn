from typing import FrozenSet
from torch.utils.data.dataset import Dataset
import argparse
from model import BasicVSR
import torch
import torch.utils.data as DATA
from vimeo_loader import Vimeo90k_dataset
from metric import PSNR 
from swin import SwinIR as vsr
import numpy as np
import cv2

parser = argparse.ArgumentParser()  
batch_size = 2
parser.add_argument("-model_iter", type=int, required=True)
args = parser.parse_args()
v90k = Vimeo90k_dataset(dataset_mode='test')
vimeo_test = DATA.DataLoader(v90k, batch_size=batch_size, num_workers=0)
vsr = vsr()
vsr.load_state_dict(torch.load('model/' + str(args.model_iter) + '_swinvsr.pth')['model'])
vsr.eval().cuda()
print("開始了")
mean_psnr = 0.0

for sample in vimeo_test:
    #load data
    hr = sample['hr']*255.0
    lr = sample['lr']
    B,C,H,W = hr.size()
    hr = hr.contiguous().view([B,H,W,C]).numpy()
    
    #pred hr
    lr = lr.cuda()
    pred_hr = vsr(lr).detach().cpu().squeeze(0)
    pred_hr = pred_hr.contiguous().view([B,H,W,C]).numpy()
    print(f'pred max: {np.max(pred_hr)}, pred min : {np.min(pred_hr)}')
    cur_psnr = 0.0
    for i in range(B):
        # rgb to y
        y_pred_hr = cv2.cvtColor(pred_hr[i,:,:,:], cv2.COLOR_BGR2YCR_CB)[:,:,0]
        y_hr = cv2.cvtColor(hr[i,:,:,:], cv2.COLOR_BGR2YCR_CB)[:,:,0]
        # psnr
        cur_psnr += cv2.PSNR(y_hr, y_pred_hr)
        print(cur_psnr, np.max(y_hr), np.max(y_pred_hr))
    mean_psnr += (cur_psnr/B)
    #print(mean_psnr)
mean_psnr = mean_psnr/v90k.__len__()
print(f'trained iter : {args.model_iter}, average psnr:{mean_psnr:.2f}.')

