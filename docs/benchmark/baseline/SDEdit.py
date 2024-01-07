import PIL
import requests
import os
import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from tqdm import tqdm
import json_lines
import argparse


def diffusers(args):

    device = f'cuda:{args.cuda}'

    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    texts = []
    with open('gpt-generated-prompts.jsonl', 'rb') as f: 
        for item in json_lines.reader(f):
            # print(item['input'], item['output'])
            texts.append([item['input'], item['output']])
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        text = texts[cnt]
        image = PIL.Image.open(f"dataset/hetu_origin/{cnt}.png")
        image = image.convert("RGB")
        images = pipe(prompt=text[1], image=image, strength=0.75, guidance_scale=7.5).images
        images[0].save(f"dataset/SDEdit/{cnt}.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, help='cuda number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)
