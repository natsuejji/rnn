import torch
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
    model = BasicVSR()
    model.to(device=torch.device("cuda:0"))
    model.train()
    train_generator = DataLoader(vimeo_loader.Vimeo90k_dataset(), batch_size=batch_size, num_workers=8)
    test_genetator = DataLoader(vimeo_loader.Vimeo90k_dataset(dataset_mode='test'), batch_size=1, num_workers=8)
    train_prefect = CUDAPrefetcher(train_generator)
    test_prefect = CUDAPrefetcher(test_genetator)
    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    average_loss = 0.0
    start.record()
    for iter_idx in range(int(num_iter)):

        sample = train_prefect.next()
        hr = sample['hr']
        lr = sample['lr']
        optimizer.zero_grad()

        pred_hr = model(lr)
        loss = criterion(pred_hr, hr)
        loss.backward()
        optimizer.step()

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
            pred_hr = pred_hr.squeeze(0).view([7,256,448,3])

            #儲存圖片
            if not os.path.exists(f'img_results/basicvsr/{iter_idx+1}'):
                os.makedirs(f'img_results/basicvsr/{iter_idx+1}')

            for i in range(pred_hr.size()[0]):
                cur_frame = pred_hr[i,:,:,:].squeeze(0).cpu().detach().numpy()
                cv2.imwrite(f'img_results/basicvsr/{iter_idx+1}/{i}.jpg',cur_frame)

            print(f'{iter_idx+1}/{num_iter}  psnr : {mean_psnr:.2f}, loss : {average_loss:.2f}')
            average_loss = 0
            torch.save(model.state_dict(), f'./model/{iter_idx}_basicvsr.pth')
        


if __name__ ==  '__main__':
    train(batch_size=3, test_iter=100, test_step=5000)
            
            
    

        
