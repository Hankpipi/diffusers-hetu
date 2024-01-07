from diffusers import StableDiffusionInpaintPipeline
import PIL
import requests
import os
import torch
import numpy as np
from tqdm import tqdm
import json_lines
import argparse


def diffusers(args):

    device = f'cuda:{args.cuda}'

    model_id = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # run a single sample
    '''
    prompt = "A dog sitting on the sofa with a hat."
    prompt = "Mountaineering Wallpapers under fireworks."
    image = PIL.Image.open(f"sample/0.png")
    image = image.convert("RGB")
    mask_image = PIL.Image.open(f"mask/mask_img/mask_0.png").resize((768, 768))
    mask_image = mask_image.convert("L")
    # image and mask_image should be PIL images.
    # The mask structure is white for inpainting and black for keeping as is
    image = pipe(prompt=prompt, image=image, mask_image=mask_image, height=768, width=768).images[0]
    image.save("sample/inpaint.png")
    '''

    texts = []
    with open('gpt-generated-prompts.jsonl', 'rb') as f: 
        for item in json_lines.reader(f):
            # print(item['input'], item['output'])
            texts.append([item['input'], item['output']])
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        text = texts[cnt]
        # print(text)
        image = PIL.Image.open(f"dataset/hetu_origin/{cnt}.png")
        image = image.convert("RGB")
        mask_image = PIL.Image.open(f"data/mask/mask_img/mask_{cnt}.png").resize((768, 768))
        mask_image = mask_image.convert("L")
        # image and mask_image should be PIL images.
        # The mask structure is white for inpainting and black for keeping as is
        image = pipe(prompt=text[1], image=image, mask_image=mask_image, height=768, width=768).images[0]
        image.save(f"dataset/inpaint/{cnt}.png")
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, help='cuda number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)