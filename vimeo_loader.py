from os import replace
import torch
from torch.utils import data as data
import numpy as np
import cv2.cv2 as cv2
import random
from torchvision import transforms
import torchvision

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    """Paired random crop.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).
    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.
    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.
    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs[:7], imgs[7:]


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.
    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

class Vimeo90k_dataset(data.Dataset):

    def __init__(self, 
                 root_path ='/aipr/vimeo_septuplet/',
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
        self.degrade_img_path = root_path + 'LR/' + degrade_mode + '/x4/' 
        self.img_path = root_path + 'HR/'
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

            cur_degraded_frame = cv2.imread(degraded_img_path)
            
            #normalization to [0,1]
            cur_frame = cur_frame/255.0
            cur_degraded_frame = cur_degraded_frame/255.0

            hr_clip.append(cur_frame)
            lr_clip.append(cur_degraded_frame)            

        hr_clip, lr_clip = paired_random_crop(hr_clip, lr_clip, 64*self.scale, self.scale)
        
        hr_clip, lr_clip = augment(hr_clip + lr_clip)
        hr_clip = torch.as_tensor(hr_clip)
        hr_clip = hr_clip.permute(0,3,1,2)[3,:,:,:]
        

        lr_clip = torch.as_tensor(lr_clip)
        lr_clip = lr_clip.permute(0,3,1,2)
        
        return {'hr': hr_clip.float(), 'lr': lr_clip.float()}

    def __len__(self):
        return len(self.all_clip)

    def __getitem__(self, index):
        return self.read_frame_seq(self.all_clip[index])
    
    
if __name__ == '__main__':
    vimeo90k = Vimeo90k_dataset()

    test2 = vimeo90k.__getitem__(0)

    loader = data.DataLoader(dataset=vimeo90k, batch_size=1, shuffle=True, num_workers=0)

    for i, sample in enumerate(loader):

        hr = sample['hr']
        print(f'hr size {hr.size()}')
        hr = hr.squeeze(0)[0,:,:,:]
        hr = (hr.permute([1,2,0]).numpy()*255.0).astype(np.uint8)
        
        print(np.shape(hr))
        cv2.imshow('a', hr)
        cv2.waitKey(0)
        
        
        
    




        
# import torch
# import torch.functional as F
# import torch.utils.data as data
# import torchvision
# from torchvision import transforms
# import util
# import random
# import os.path as path
# from PIL import Image
# import skimage.color as color
# import numpy as np

# class Vimeo90k_dataset(data.Dataset):

#     def __init__(self, 
#                  root ='/aipr/vimeo_septuplet/', 
#                  scale = 4,
#                  degrade_mode = 'BI', 
#                  dataset_mode = 'train'):
#         super(Vimeo90k_dataset, self).__init__()

#         if degrade_mode not in ['BI', 'BD']:
#             raise ValueError(f'只有bi或bd兩種下採樣')
#         if dataset_mode not in ['train', 'test']:
#             raise ValueError(f'只有train或test兩種資料集')

#         self.scale = scale
#         self.dataset_mode = dataset_mode
#         self.degrade_mode = degrade_mode
#         #圖片位置
#         self.root = root
#         self.degrade_img_path = self.root + 'LR/' + degrade_mode + '/x4/' 
#         self.img_path = self.root + 'HR/'

#         self.metadata_path = self.root + 'sep_trainlist.txt' if dataset_mode == 'train' else self.root + 'sep_testlist.txt'
#         self.all_clip = []
#         self.num_frame = 7
#         with open(self.metadata_path, 'r') as f:
#             self.all_clip = [x.replace('\n', '') for x in f.readlines()]
#         # crop
#         self.crop_lr = transforms.CenterCrop(64)
#         self.crop_gt = transforms.CenterCrop(64*scale)

#     def preprocessing(self, gt, lr):
        
#         # 資料增強 flag
#         hflip = random.random() < 0.5
#         vflip = random.random() < 0.5 
#         if hflip:
#             gt = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in gt]
#             lr = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in lr]
#         if vflip:
#             gt = [x.transpose(Image.FLIP_TOP_BOTTOM) for x in gt]
#             lr = [x.transpose(Image.FLIP_TOP_BOTTOM) for x in lr]

#         lr = [self.crop_lr(x) for x in lr]
#         gt = [self.crop_gt(x) for x in gt]

#         lr_tensor = torch.stack([transforms.ToTensor()(x) for x in lr])
#         gt_tensor  = torch.stack([transforms.ToTensor()(x) for x in gt])
#         return gt_tensor, lr_tensor

#     def __getitem__(self, index):
#         return self.read_img_seq(self.all_clip[index])

#     def __len__(self):
#         return len(self.all_clip)

#     def read_img_seq(self, clip):
#         hr_clip = []
#         lr_source_clip = []

#         for i in range(self.num_frame):
            
#             cur_frame_idx = i+1
#             cur_img_path = self.img_path + clip + '/im' + str(cur_frame_idx) + '.png'
#             degraded_img_path = self.degrade_img_path + clip + '/im' + str(cur_frame_idx) + '.png'

#             cur_img = Image.open(cur_img_path)
#             degraded_img = Image.open(degraded_img_path)

#             lr_source_clip.append(degraded_img)
#             if i == 3:
#                 hr_clip.append(cur_img)
        
#         hr_clip, lr_source_clip = self.preprocessing(hr_clip, lr_source_clip)

#         return {    
#             'hr' : hr_clip,
#             'lr' : lr_source_clip,
#         }


# if __name__ == '__main__':

#     v = Vimeo90k_dataset()
#     v.__getitem__(0)


