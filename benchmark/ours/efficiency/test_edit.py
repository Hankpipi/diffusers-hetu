import json
import torch
from diffusers import StableDiffusionPipelineEdit, DPMSolverMultistepScheduler
import argparse


def diffusers(args):

    device = f'cuda:{args.cuda}'
    access_token = "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"
    pipe = StableDiffusionPipelineEdit.from_pretrained("stabilityai/stable-diffusion-2-1", use_auth_token=access_token)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    f = open(f'{args.edit_size}/prompt.txt', 'r')
    prompts = json.load(f)

    prompt = prompts['input']
    prompt_edited = prompts['output']

    latents = torch.load(f"{args.edit_size}/random_seed.pt", map_location=device)
    images = pipe(prompt, num_inference_steps=50, latents=latents, save_checkpoint=True).images

    for i in range(len(images)):
        images[i].save(f"image_origin.png")
    
    mask = torch.load(f'{args.edit_size}/mask.pt')
    images = pipe(prompt_edited, num_inference_steps=50, latents=latents, save_checkpoint=False, mask=mask).images

    for i in range(len(images)):
        images[i].save(f"image_edited.png")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--edit_size', type=int, default=5, help='edit size')

    args = parser.parse_args()
    diffusers(args)
