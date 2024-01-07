from PIL import Image
import requests
import os
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import argparse


def diffusers(args):
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        diff = Image.open(f"mask_otsu/diff/diff_{cnt}.png")
        diff = diff.convert("L")
        diff = np.array(diff)
        temp = cv.equalizeHist(diff)
        # thre, temp = cv.threshold(temp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thre, temp = cv.threshold(temp, 64, 255, cv.THRESH_BINARY)
        kernel = np.ones((2, 2))
        temp = cv.erode(temp, kernel, 1)
        temp = cv.dilate(temp, kernel, 1)
        # kernel = np.ones((5, 5))
        # temp = cv.erode(temp, kernel, 1)
        # temp = cv.dilate(temp, kernel, 1)
        mask = torch.tensor((temp == 255))
        torch.save(mask, f'mask_0_25/mask_pt/mask_{cnt}.pt')
        # print(mask)
        mask_rate = mask.sum() / np.multiply(*mask.size())
        print("mask占比:", mask_rate)
        im = Image.fromarray((mask * 255).cpu().numpy().astype(np.uint8))
        im = im.convert('L')  
        im.save(f'mask_0_25/mask_img/mask_{cnt}.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=2, help='cuda number')
    parser.add_argument('--slice', default=0, help='slice number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)