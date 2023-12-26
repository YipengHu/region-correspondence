
# import os
# import time

# import numpy as np
import nibabel as nib
import torch

FILENAME = "./data/7013610-masks/001001_mask.nii"



masks = nib.load(FILENAME).get_fdata()
masks = torch.stack([torch.tensor(masks==i) for i in range(1,int(masks.max()+1))], dim=3) # remove the background
# masks = torch.nn.functional.one_hot(torch.tensor(masks,dtype=torch.int64), num_classes=int(masks.max()+1))[...,1:]  # remove the background
