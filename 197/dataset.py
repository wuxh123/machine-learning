# -*- coding: utf-8 -*-
import argparse
import os, sys, glob
import numpy as np
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

#import config as cfg
#from utils import mkdir

import torch
from torch.utils.data import Dataset

class UNetDataset(Dataset):
    def __init__(self, image_list=None, mask_list=None, phase='train'):
        super(UNetDataset, self).__init__()
        self.phase = phase
        #read imgs
        if phase != 'test':
            assert mask_list, 'mask list must given when training'
            self.mask_file_list = mask_list

            self.img_file_list = [f.replace("mask", 'image') for f in mask_list]

            assert len(self.img_file_list) == len(self.mask_file_list), 'the count of image and mask not equal'
        if phase == 'test':
            assert image_list, 'image list must given when testing'
            self.img_file_list = image_list

    def __getitem__(self, idx):
        img_name = self.img_file_list[idx]
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        img = np.expand_dims(img, 0)

        mask_name = self.mask_file_list[idx]
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE).astype(int)
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)

        assert (np.array(img.shape[1:]) == np.array(mask.shape[1:])).all(), (img.shape[1:], mask.shape[1:])
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.img_file_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create dataset')
    parser.add_argument('--images', '-l', help='images dir', default='')
    parser.add_argument('--masks', '-m', help='masks dir', default='')
    parser.add_argument('--target', '-t', help='target dir', default='')
    args = parser.parse_args()

    if not args.target:
        from torch.utils.data import DataLoader
        import torchvision
        mask_list = glob.glob('./crack_seg_dir/mask/*.jpg')
        dataset = UNetDataset(mask_list=mask_list, phase='train')
        data_loader = DataLoader(
            dataset,
            batch_size = 100,
            shuffle = True,
            num_workers = 8,
            pin_memory=False)
        print (len(dataset))
        count = 0.
        pos = 0.
        for i, (data, target) in enumerate(data_loader, 0):
            if i % 100 == 0:
                print (i)
            count += np.prod(data.size())
            pos += (data==1).sum()
        print (pos / count)