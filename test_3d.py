
# import os
# import time

# import numpy as np
import nibabel as nib
import torch

from region_correspondence.paired_regions import PairedRegions



FILENAME_MOV = "./data/7013610-masks/002007_mask.nii"
FILENAME_FIX = "./data/7013610-masks/001001_mask.nii"
# FILENAME_FIX = "./data/7013610-masks/002008_mask.nii" (for the same sized masks)


one_hot = lambda m : torch.stack([torch.tensor(m==i) for i in range(1,int(m.max()+1))], dim=0)
# one_hot = lambda m : torch.nn.functional.one_hot(torch.tensor(m,dtype=torch.int64), num_classes=int(m.max()+1))[...,1:]  # remove the background
masks_mov = one_hot(nib.load(FILENAME_MOV).get_fdata())
masks_fix = one_hot(nib.load(FILENAME_FIX).get_fdata())


paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix)
ddf = paired_rois.get_dense_correspondence()
