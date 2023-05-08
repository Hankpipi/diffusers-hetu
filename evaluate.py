import torch
from diffusers import HetuUnetConfig, StableDiffusionPipelineEdit, DPMSolverMultistepScheduler
from tqdm import tqdm
import json_lines # pip install json-lines
import argparse
import os
import random

def diffusers(args):

    device = f'cuda:{args.cuda}'
    access_token = "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"
    if args.download:
        pipe = StableDiffusionPipelineEdit.from_pretrained("stabilityai/stable-diffusion-2-1", use_auth_token=access_token)
    else:
        # may change the root
        pipe = StableDiffusionPipelineEdit.from_pretrained("/root/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1", use_auth_token=access_token)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    config = HetuUnetConfig(
                height = args.height,
                width = args.width,
                ctx = pipe.device,
                profile = args.profile,
                latent_scale_linear = args.latent_scale_linear,
                latent_scale_conv = args.latent_scale_conv,
                latent_scale_attn = args.latent_scale_attn,
                fuse_gn_av_conv = args.fuse_gn_av_conv,
                merge_rate = args.merge_rate,
                fuse_attn1_attn2_ff = args.fuse_attn1_attn2_ff,
                fuse_self_attn = args.fuse_self_attn,
                fuse_cross_attn = args.fuse_cross_attn,
                radical = args.radical
                )

    if args.run_sample:
    
        '''
        prompt = [
            "Mountaineering Wallpapers.",
        ]
        prompt_edited = [
            "Mountaineering Wallpapers under fireworks.",
        ]
        '''
        prompt = [
            "A dog sitting on the sofa.",
        ]
        prompt_edited = [
            "A dog sitting on the sofa with a hat on its head.",
        ]

        latents = torch.randn((1, 4, args.height, args.width), dtype=torch.float32)
        # torch.save(latents, "seed.pt")
        # latents = torch.load("seed.pt", map_location=device)
        images = pipe(prompt, num_inference_steps=50, latents=latents, save_checkpoint=True, config=config).images

        for i in range(len(images)):
            images[i].save(f"image_origin.png")
        
        # mask = torch.load(f'mask_temp.pt')
        images = pipe(prompt_edited, num_inference_steps=50, latents=latents, save_checkpoint=False).images

        for i in range(len(images)):
            images[i].save(f"image_edited.png")
    

    if args.run_dataset:

        texts = []
        with open('gpt-generated-prompts.jsonl', 'rb') as f: 
            for item in json_lines.reader(f):
                texts.append([item['input'], item['output']])
        print('dataset size:', len(texts))
        
        for cnt in tqdm(range(int(args.base_num), int(args.limit_num))):

            text_pair = texts[cnt]

            latents = torch.load(f'random_seed/{cnt}.pt', map_location=device)

            images = pipe(text_pair[0], num_inference_steps=50, latents=latents, save_checkpoint=True).images
            images[0].save(f"dataset/hetu_origin/{cnt}.png")
            mask = torch.load(f'mask/mask_pt/mask_{cnt}.pt')
            if mask.sum() == 0:
                images[0].save(f"dataset/hetu_edit_conv/{cnt}.png")
                continue
            images = pipe(text_pair[1], num_inference_steps=50, latents=latents, save_checkpoint=False, mask=mask).images
            images[0].save(f"dataset/hetu_edit_conv/{cnt}.png")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=5, help='cuda number')
    parser.add_argument('--download', type=int, default=0, help='need to download the model from huggingface')

    parser.add_argument('--run_sample', type=int, default=1, help='run a sample (for performance test)')
    parser.add_argument('--run_dataset', type=int, default=0, help='run the dataset (for efficiency test)')
    parser.add_argument('--base_num', type=int, default=0, help='base image number of the dataset')
    parser.add_argument('--limit_num', type=int, default=3000, help='limit image number of the dataset')

    # Only support height = width for now.
    parser.add_argument('--height', type=int, default=96, help='height of the initial latents')
    parser.add_argument('--width', type=int, default=96, help='width of the initial latents')

    # Use to profile sparse op calculation time. The data will store in profile_[conv/linear/attention].pkl file.
    parser.add_argument('--profile', type=int, default=0, help='profile the performance')

    parser.add_argument('--latent_scale_linear', type=int, default=0, help='limit of the latent scale to do sparse linear')
    parser.add_argument('--latent_scale_conv', type=int, default=0, help='limit of the latent scale to do sparse conv')
    parser.add_argument('--latent_scale_attn', type=int, default=0, help='limit of the latent scale to do sparse attention')

    parser.add_argument('--fuse_gn_av_conv', type=int, default=1, help='fuse_gn_av_conv')

    # If we want to use merge_rate, we need to set fuse_attn1_attn2_ff to False.
    parser.add_argument('--merge_rate', type=float, default=0.9, help='merge_rate to edit the images')
    parser.add_argument('--fuse_attn1_attn2_ff', type=int, default=0, help='fuse_attn1_attn2_ff')

    parser.add_argument('--fuse_self_attn', type=int, default=1, help='fuse_self_attn')
    parser.add_argument('--fuse_cross_attn', type=int, default=1, help='fuse_cross_attn')
    parser.add_argument('--fuse_ln_ff_linear_av_add', type=int, default=1, help='fuse_ln_ff_linear_av_add')

    parser.add_argument('--radical', type=int, default=1, help='use sparse k and v in self attention')

    args = parser.parse_args()
    diffusers(args)
