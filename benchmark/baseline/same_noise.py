from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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

    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    texts = []
    with open('gpt-generated-prompts.jsonl', 'rb') as f: 
        for item in json_lines.reader(f):
            # print(item['input'], item['output'])
            texts.append([item['input'], item['output']])
    
    for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

        text = texts[cnt]
        latents = torch.load(f'data/random_seed/{cnt}.pt', map_location=device).float()
        # latents = torch.randn((1, 4, 96, 96), dtype=torch.float32).repeat(1, 1, 1, 1)
        image = pipe(prompt=text[1], latents=latents, height=768, width=768).images[0]
        image.save(f"dataset/same_noise/{cnt}.png")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, help='cuda number')
    parser.add_argument('--base_num', default=0, help='base image number')
    parser.add_argument('--limit_num', default=3000, help='limit image number')
    args = parser.parse_args()
    diffusers(args)