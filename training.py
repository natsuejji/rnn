import torch
from torch._C import ThroughputBenchmark
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from loss import CharbonnierLoss
from metric import PSNR
from model import BasicVSR
from prefetcher import CUDAPrefetcher
import vimeo_loader
import time 
import os
import cv2

def train(batch_size=2, num_iter=3e6, test_step=5000, test_iter=100):
    torch.backends.cudnn.benchmark = True
    model = BasicVSR()
    model.to(device=torch.device("cuda:0"))
    model.train()
    train_generator = DataLoader(vimeo_loader.Vimeo90k_dataset(), batch_size=batch_size, num_workers=1)
    test_genetator = DataLoader(vimeo_loader.Vimeo90k_dataset(dataset_mode='test'), batch_size=1, num_workers=1)
    train_prefect = CUDAPrefetcher(train_generator)
    test_prefect = CUDAPrefetcher(test_genetator)
    criterion = CharbonnierLoss()
    lr = 2e-4
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
                if 'flow_estimator' in k:
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
        #調整學習率
        scheduler.step()
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
                pred_hr = model(lr)
                mean_psnr = PSNR()(hr, pred_hr).item()
            pred_hr = pred_hr.reshape([batch_size*7,3,256,256]).permute([0,2,3,1])

            #儲存圖片
            if not os.path.exists(f'img_results/basicvsr/{iter_idx+1}'):
                os.makedirs(f'img_results/basicvsr/{iter_idx+1}')

            for i in range(pred_hr.size()[0]):
                cur_frame = pred_hr[i,:,:,:].squeeze(0).cpu().detach().numpy()
                print(cur_frame)
                cv2.imwrite(f'img_results/basicvsr/{iter_idx+1}/{i}.jpg',cur_frame)
            end.record()
            print(f'{iter_idx+1}/{num_iter}  psnr : {mean_psnr:.2f}, loss : {average_loss:.2f} \
                Consume time:{start.elapsed_time(end)/1000.0:.2f} second.')
            average_loss = 0
            torch.save(model.state_dict(), f'./model/{iter_idx}_basicvsr.pth')
            start.record()
        


if __name__ ==  '__main__':
    print(torch.cuda.is_available())
    train(batch_size=1, test_iter=1, test_step=1)
            
            
    

        
