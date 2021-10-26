import torch
from torch._C import ThroughputBenchmark
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from loss import CharbonnierLoss
from metric import PSNR
from swin import SwinIR
from prefetcher import CUDAPrefetcher
import vimeo_loader
import time 
import numpy as np
import model.BasicVSR as vsr
import os
import cv2

def train(batch_size=2, num_iter=3e5, test_step=5000, test_iter=100):
    torch.backends.cudnn.benchmark = True
    model = vsr()
    model.to(device=torch.device("cuda:0"))
    model.train()
    
    train_generator = DataLoader(vimeo_loader.Vimeo90k_dataset(), batch_size=batch_size, num_workers=8)
    test_genetator = DataLoader(vimeo_loader.Vimeo90k_dataset(dataset_mode='test'), batch_size=batch_size, num_workers=8)
    train_prefect = CUDAPrefetcher(train_generator)
    test_prefect = CUDAPrefetcher(test_genetator)
    criterion = CharbonnierLoss()
    lr = 2e-4
    # optimizer = optim.Adam([
    #     {'params': model.spynet.parameters(), 'lr': lr*0.125},
    #     {'params': model.conv_after_body.parameters()},
    #     {'params': model.conv_before_upsample.parameters()},
    #     {'params': model.upsample.parameters()},
    #     {'params': model.conv_last.parameters()},
    #     {'params': model.layers.parameters()},
    #     {'params': model.conv_first.parameters()},
    #         ], lr=lr)

    optimizer = optim.Adam([
        {'params': model.flow_estimator.parameters(), 'lr': lr*0.125},
        {'params': model.forwrad_prop.parameters()},
        {'params': model.backwrad_prop.parameters()},
        {'params': model.fusion.parameters()},
        {'params': model.upsample1.parameters()},
        {'params': model.upsample2.parameters()},
        {'params': model.conv_hr.parameters()},
        {'params': model.conv_last.parameters()},
        {'params': model.img_upsample.parameters()},
            ], lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-7, T_max=lr)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    average_loss = 0.0
    start.record()
    for iter_idx in range(int(num_iter)):
        #前五千次不訓練spynet的weight
        if iter_idx+1 == 1:
            for k,v in model.named_parameters():
                if 'spynet' in k:
                    v.requires_grad = False
        elif iter_idx+1 == 5000+1:
            for v in model.parameters():
                v.requires_grad = True
        
        sample = train_prefect.next()
        hr = sample['hr']
        lr = sample['lr']
        optimizer.zero_grad()
        pred_hr = model(lr)
        loss = criterion(pred_hr, hr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():

            if (iter_idx+1) % 100  == 0:
                end.record()
                torch.cuda.synchronize()
                print(f'{iter_idx+1}/{num_iter}  cur_iter_loss : {loss.item():.2f}. \
                    Consume time:{start.elapsed_time(end)/1000.0:.2f} second.')
                start.record()
                
            average_loss += loss.item()

            if iter_idx % test_step == test_step-1:
                mean_psnr = 0.0
                average_loss /= test_step
                for test_iter_idx in range(int(test_iter)):
                    sample = test_prefect.next()
                    hr = sample['hr']
                    lr = sample['lr']

                    pred_hr = model(lr).detach()
                    mean_psnr += PSNR()(hr*255.0, pred_hr*255.0).item()
                mean_psnr /= test_iter
                end.record()
                print(f'{iter_idx+1}/{num_iter}  psnr : {mean_psnr:.2f}, loss : {average_loss:.2f} \
                        Consume time:{start.elapsed_time(end)/1000.0:.2f} second.')
                average_loss = 0
                state = {
                        'model' : model.state_dict(),
                        'optim' : optimizer.state_dict()
                }
                torch.save(state, f'./model/{iter_idx+1}_swinvsr.pth')
                start.record()

if __name__ ==  '__main__':
    print(torch.cuda.is_available())
    train(batch_size=2, test_iter=100, test_step=5000)
            
            
    

        
