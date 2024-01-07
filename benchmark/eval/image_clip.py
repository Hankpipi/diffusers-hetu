import numpy as np
import clip
import os
import torch
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda:0")


################ CLIP ################
print(clip.available_models())

model, preprocess = clip.load("ViT-B/32")
model.to(device).eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)



################ images ################
images_feature = []
with torch.no_grad():
    for i in tqdm(range(3000)):
        image = Image.open(f'dataset/hetu_edit/{i}.png').convert("RGB")
        image = preprocess(image).to(device).repeat(1,1,1,1)
        image_feature = model.encode_image(image).float()
        image = image.cpu()
        image_feature = image_feature.cpu()
        images_feature.append(image_feature)
    
images_feature = torch.cat(images_feature)
torch.save(images_feature, f"hetu_edit.pth")  