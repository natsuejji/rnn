from os import sep
import numpy as np
import torch
import torch.functional as F
import cv2.cv2 as cv2
import os

def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.
    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.
    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.
        Args:
            x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
            kernel_size (int): Kernel size. Default: 13.
            scale (int): Downsampling factor. Supported scale: (2, 3, 4).
                Default: 4.
        Returns:
            Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x


def degradation(clips, 
                data_path,
                scale=4,
                mode = 'BI'):
    if mode not in ['BI', 'BD']:
        raise ValueError('只支援BI和BD的degradation.')
    
    degraded_path = data_path + 'LR/' + mode + '/x' + str(scale) + '/'
    data_path = data_path + 'HR/'
    for clip in clips:
        clip = clip.replace('\n','')
        if not os.path.exists(degraded_path + clip):
            os.makedirs(degraded_path + clip)
        frames = []
        for i in range(7):
            cur_frame_idx = i+1
            cur_img_path = data_path + clip + '/im' + str(cur_frame_idx) + '.png'
            degraded_img_path = degraded_path + clip + '/im' + str(cur_frame_idx) + '.png'
            
            cur_frame = cv2.imread(cur_img_path)
            h,w,c = np.shape(cur_frame) 
            if mode == 'BD':
                frame_tensor = torch.tensor(cur_frame)
                degraded_img = degradation(frame_tensor)
            if mode == 'BI':
                degraded_img = cv2.resize(cur_frame,(w//scale,h//scale), cv2.INTER_CUBIC)

            cv2.imwrite(degraded_img_path, degraded_img)
        print(f'clip {clip} is degraded.')
            
            

def main():
    
    root_path='E:/dataset/vimeo_septuplet/'
    test_set_txt = 'sep_testlist.txt'
    train_set_txt = 'sep_trainlist.txt'

    train_set = []
    test_set = []

    with open(root_path + test_set_txt, 'r') as f:
        test_set = f.readlines()
    with open(root_path + train_set_txt, 'r') as f:
        train_set = f.readlines()

    degradation(train_set, data_path=root_path+'sequences/')
    degradation(test_set, data_path=root_path+'sequences/')    

    
    

if __name__ == '__main__':
    main()
    
    
    
