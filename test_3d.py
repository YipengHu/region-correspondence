
import os
import time

import numpy as np
import nibabel as nib
import torch

FILENAME = "./data/7013610-masks/001001_mask.nii"

t0 = time.time()
masks = nib.load(FILENAME).get_fdata().astype(np.int64)
masks = torch.nn.functional.one_hot(torch.tensor(masks), num_classes=masks.max()+1).to(torch.int8)[...,1:]  # remove the background
print(time.time()-t0)

t0 = time.time()
masks = nib.load(FILENAME).get_fdata().astype(np.int8)
masks = torch.stack([torch.tensor(masks==i,dtype=torch.int8) for i in range(1,masks.max()+1)], dim=3)  # remove the background
print(time.time()-t0)
