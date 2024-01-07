from PIL import Image
import os
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import argparse


def diffusers(args):

    file = open('resize_mask_rate.txt', 'w')
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        mask = torch.load(f"mask_otsu/mask_pt/mask_{cnt}.pt")
        mask = torch.nn.MaxPool2d(kernel_size=(8, 8))(mask.float().repeat(1, 1, 1, 1)) 
        mask = (mask > 0.5)
        mask = mask.numpy().reshape(-1)
        mask_rate = mask.sum() / mask.shape[-1]
        print(mask_rate, file=file)

    file.close()

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=2, help='cuda number')
    parser.add_argument('--slice', default=0, help='slice number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)