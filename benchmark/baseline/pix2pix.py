import PIL
import requests
import os
import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline
from tqdm import tqdm
import json_lines
import argparse


def diffusers(args):

    device = f'cuda:{args.cuda}'

    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # run a single sample
    ''' 
    text = 'Add a river under the sky.'
    image = PIL.Image.open(f"../image1.png")
    image = image.convert("RGB")
    images = pipe(text, image=image, num_inference_steps=50).images
    images[0].save(f"pix2pix_sample.png")
    '''

    
    texts = []
    with open('gpt-generated-prompts.jsonl', 'rb') as f: 
        for item in json_lines.reader(f):
            # print(item['input'], item['output'])
            texts.append(item['edit'])
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        text = texts[cnt]
        image = PIL.Image.open(f"dataset/hetu_origin/{cnt}.png")
        image = image.convert("RGB")
        images = pipe(text, image=image, num_inference_steps=50).images
        images[0].save(f"dataset/pix2pix/{cnt}.png")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, help='cuda number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)
