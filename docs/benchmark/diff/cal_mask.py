import torch
from PIL import Image
import numpy as np

'''
file = open('mask_rate.txt', 'w')
for i in range(3000):
    mask = torch.load(f'mask_otsu/mask_pt/mask_{i}.pt')
    print((mask.sum() / mask.numel()).item(), file=file)
file.close()
'''

mask_level = {}
for i in range(3000):
    diff = Image.open(f"mask_otsu/diff/diff_{i}.png")
    diff = np.array(diff)
    for j in diff:
        for k in j:
            if k not in mask_level:
                mask_level[k] = 1
            else:
                mask_level[k] += 1
file = open('mask_rate.txt', 'w')
sum = 0
for i in range(256):
    print(mask_level[i], file=file)
file.close()
