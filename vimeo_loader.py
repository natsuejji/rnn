from os import replace
import torch
from torch.utils import data as data
import numpy as np
import cv2.cv2 as cv2
import random
class Vimeo90k_dataset(data.Dataset):

    def __init__(self, 
                 root_path ='E:/dataset/vimeo_septuplet/',
                 scale = 4,
                 dataset_mode = 'train',
                 degrade_mode ='BI'
                ):
        super(Vimeo90k_dataset, self).__init__()
        if degrade_mode not in ['BI', 'BD']:
            raise ValueError(f'只有bi或bd兩種下採樣')
        if dataset_mode not in ['train', 'test']:
            raise ValueError(f'只有train或test兩種資料集')
        self.dataset_mode = dataset_mode
        self.degrade_img_path = root_path + 'sequences/LR/' + degrade_mode + '/x4/' 
        self.img_path = root_path + 'sequences/HR/'
        self.scale = scale
        self.metadata_path = root_path + 'sep_trainlist.txt' if dataset_mode == 'train' else root_path + 'sep_testlist.txt'
        self.all_clip = []
        with open(self.metadata_path, 'r') as f:
            self.all_clip = [x.replace('\n', '') for x in f.readlines()]

    def read_frame_seq(self, clip):
        hr_clip = []
        lr_clip = []
        
        for i in range(7):
            cur_frame_idx = i+1
            cur_img_path = self.img_path + clip + '/im' + str(cur_frame_idx) + '.png'
            degraded_img_path = self.degrade_img_path + clip + '/im' + str(cur_frame_idx) + '.png'

            cur_frame = cv2.imread(cur_img_path)
            hr_clip.append(cur_frame)
            cur_degraded_frame = cv2.imread(degraded_img_path)
            lr_clip.append(cur_degraded_frame)
        hr_clip = torch.as_tensor(hr_clip)
        t,h,w,c = hr_clip.size()
        hr_clip = hr_clip.view(t,c,h,w)
        lr_clip = torch.as_tensor(lr_clip)
        t,h,w,c = lr_clip.size()
        lr_clip = lr_clip.view(t,c,h,w)
        
            
        return {'hr': hr_clip.float(), 'lr': lr_clip.float()}

    def __len__(self):
        return len(self.all_clip)

    def __getitem__(self, index):
        return self.read_frame_seq(self.all_clip[index])
    
    
if __name__ == '__main__':

    vimeo90k = Vimeo90k_dataset()

    test2 = vimeo90k.__getitem__(0)

    loader = data.DataLoader(dataset=vimeo90k, batch_size=1, shuffle=True, num_workers=2)

    for i, sample in enumerate(loader):
        print(sample[0].size())
    




        

