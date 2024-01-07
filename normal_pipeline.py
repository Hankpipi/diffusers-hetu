import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
import json_lines # pip install json-lines
import argparse
import os
import random

def diffusers(args):

    device = f'cuda:{args.cuda}'
    if args.download:
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", force_download=True)
    else:
        # may change the root
        pipe = StableDiffusionPipeline.from_pretrained("/root/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    prompt = [
        "A dog sitting on the sofa.",
    ]

    latents = torch.randn((1, 4, args.height, args.width), dtype=torch.float32)
    images = pipe(prompt, num_inference_steps=50, latents=latents).images

    for i in range(len(images)):
        images[i].save(f"image_origin.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--download', type=int, default=0, help='need to download the model from huggingface')

    # Only support height = width for now.
    parser.add_argument('--height', type=int, default=96, help='height of the initial latents')
    parser.add_argument('--width', type=int, default=96, help='width of the initial latents')

    args = parser.parse_args()
    diffusers(args)
