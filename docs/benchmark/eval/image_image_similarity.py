import numpy as np
import os
import torch
from tqdm import tqdm

device = torch.device("cuda:0")

################ Cosine Similarity ################
embeddings_1 = torch.load("hetu_origin.pth", map_location=device)
embeddings_2 = torch.load("hetu_origin.pth", map_location=device)

with torch.no_grad():
    not_nan = []
    for i in range(3000):
        if embeddings_1.norm(dim=-1)[i] == 0 or embeddings_2.norm(dim=-1)[i] == 0:
            continue
        not_nan.append(i)
    embeddings_1 = embeddings_1[not_nan]
    embeddings_2 = embeddings_2[not_nan]
    print('len:', embeddings_1.shape[0])
    embeddings_1 /= embeddings_1.norm(dim=-1, keepdim=True)
    embeddings_2 /= embeddings_2.norm(dim=-1, keepdim=True)

    # 余弦相似度方法
    similarity = embeddings_1 @ embeddings_2.T
    print((similarity.trace() / len(similarity)).item())