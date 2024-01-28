# Test data are from https://zenodo.org/records/7013610, see make_test_data.py for details.

# import os
# import time

# import numpy as np
import nibabel as nib
import torch

from region_correspondence.paired_regions import PairedRegions


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# read test masks
masks_mov = torch.stack([torch.tensor(nib.load("./data/test0_mask{}.nii.gz".format(idx)).get_fdata()) for idx in range(8)], dim=0).to(torch.bool)
masks_fix = torch.stack([torch.tensor(nib.load("./data/test1_mask{}.nii.gz".format(idx)).get_fdata()) for idx in range(8)], dim=0).to(torch.bool)

paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=device)
ddf = paired_rois.get_dense_correspondence(verbose=True)


## save warped images for visulisation
