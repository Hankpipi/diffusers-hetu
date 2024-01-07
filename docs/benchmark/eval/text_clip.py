import numpy as np
import clip
import os
import torch
from tqdm import tqdm
import json_lines
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

################ texts ################
texts_origin = []
texts_edit = []
cnt = 0
with open('gpt-generated-prompts.jsonl', 'rb') as f: 
    for item in json_lines.reader(f):
        # print(item['input'], item['output'])
        texts_origin.append(item['input'])
        texts_edit.append(item['output'])
        cnt += 1
        if cnt == 3000:
            break

all_embeddings = []
for text in tqdm(texts_origin):
    with torch.no_grad():
        if len(text) > 77:
            text = text[:77]
        text_tokens = clip.tokenize(text).to(device)
        text_embeddings = model.encode_text(text_tokens).float()
        text_embeddings = text_embeddings.cpu()
        all_embeddings.append(text_embeddings)
        
all_embeddings = torch.cat(all_embeddings)

# 保存Tensor为pth文件
torch.save(all_embeddings, f"texts_origin.pth") 

all_embeddings = []
for text in tqdm(texts_edit):
    with torch.no_grad():
        if len(text) > 77:
            text = text[:77]
        text_tokens = clip.tokenize(text).to(device)
        text_embeddings = model.encode_text(text_tokens).float()
        text_embeddings = text_embeddings.cpu()
        all_embeddings.append(text_embeddings)
        
all_embeddings = torch.cat(all_embeddings)

# 保存Tensor为pth文件
torch.save(all_embeddings, f"texts_edit.pth") 
