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
def train(batch_size=2, num_iter=3e6, test_step=5000, test_iter=100):
    model = BasicVSR()
    model.to(device=torch.device("cuda:0"))

    train_generator = DataLoader(vimeo_loader.Vimeo90k_dataset(), batch_size=batch_size)
    test_genetator = DataLoader(vimeo_loader.Vimeo90k_dataset(dataset_mode='test'), batch_size=batch_size)
    train_prefect = CUDAPrefetcher(train_generator)
    test_prefect = CUDAPrefetcher(test_genetator)
    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    running_loss = 0.0
    for iter_idx in range(int(num_iter)):
        
        sample = train_prefect.next()
        hr = sample['hr']
        lr = sample['lr']
        optimizer.zero_grad()

        pred_hr = model(lr)
        loss = criterion(pred_hr, hr)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if iter_idx % test_step == test_step-1:
            mean_psnr = 0.0
            for test_iter_idx in range(int(test_iter)):
                sample = test_prefect.next()
                hr = sample['hr']
                lr = sample['lr']
                pred_hr = model(lr)
                mean_psnr = PSNR()(hr, pred_hr).item()
            print(f'{iter_idx}/{num_iter}  psnr : {mean_psnr}, loss : {running_loss}')
            torch.save(model.state_dict(), f'./model/{iter_idx}_basicvsr.pth')
        


if __name__ ==  '__main__':
    print(torch.cuda.is_available())
    train(batch_size=1, test_iter=2, test_step=1)
            
            
    

        
