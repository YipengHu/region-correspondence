# Description: This script is used to generate test data, data source: https://zenodo.org/records/7013610


import sys

import nibabel as nib
import torch
from PIL import Image
import numpy as np


def make_test_data_3d():
    FILENAME_MOV = "./data/7013610-masks/002007_mask.nii"
    FILENAME_FIX = "./data/7013610-masks/001001_mask.nii"
    # FILENAME_FIX = "./data/7013610-masks/002007_mask.nii" (debug for the same masks)
    # FILENAME_FIX = "./data/7013610-masks/002008_mask.nii" (debug for the same sized masks)

    def volume_resize(vol, new_size, mode='trilinear'):
        '''
        vol: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
        new_size: (D1,H1,W1)
        mode: 'trilinear' or 'nearest'
        '''
        resized = torch.nn.functional.interpolate(
            input=vol.unsqueeze(0),
            size=new_size,
            mode=mode,
            align_corners=True,
        )  # (C,W1,H1,D1)
        return resized.squeeze(0)
    
    one_hot = lambda m : torch.stack([torch.tensor(m==i) for i in range(1,int(m.max()+1))], dim=0)
    # one_hot = lambda m : torch.nn.functional.one_hot(torch.tensor(m,dtype=torch.int64), num_classes=int(m.max()+1))[...,1:]  # remove the background
    masks_mov = one_hot(nib.load(FILENAME_MOV).get_fdata())
    masks_fix = one_hot(nib.load(FILENAME_FIX).get_fdata())

    # resize - use different sizes for debug purposes
    masks_mov = (volume_resize(masks_mov.to(torch.float32), new_size=(76,78,32))).to(torch.uint8)
    masks_fix = (volume_resize(masks_fix.to(torch.float32), new_size=(58,60,36))).to(torch.uint8)

    # save
    for idx in range(masks_mov.shape[0]):
        nib.save(nib.Nifti1Image(masks_mov[idx].numpy(),affine=torch.eye(4).numpy()), "./data/3d/test0_mask{}.nii.gz".format(idx))
        nib.save(nib.Nifti1Image(masks_fix[idx].numpy(),affine=torch.eye(4).numpy()), "./data/3d/test1_mask{}.nii.gz".format(idx))


def make_test_data_2d():
    FOLDERNAME = "./data/Prostate/sample1"
    for idx in range(8):
        img_mov = Image.open("{}/mask_1_{:d}.png".format(FOLDERNAME, idx+1))
        img_fix = Image.open("{}/mask_2_{:d}.png".format(FOLDERNAME, idx+1))
        img_mov = np.array(img_mov.getdata(),dtype=np.uint8).reshape(img_mov.size)
        img_fix = np.array(img_fix.getdata(),dtype=np.uint8).reshape(img_fix.size)
        Image.fromarray(img_mov*255).save("./data/2d/test0_mask{}.png".format(idx))
        Image.fromarray(img_fix*255).save("./data/2d/test1_mask{}.png".format(idx))

if len(sys.argv) > 1:
    if sys.argv[1] == '2d':
        print("Generating 2d test data")
        make_test_data_2d()
    elif sys.argv[1] == '3d':
        print("Generating 3d test data")
        make_test_data_3d()
    else:
        pass

## utility functions
def load_test_data(type='3d'):

    match type.lower():
        case '2d':
            FOLDERNAME = './data/2d'
            MASK_SIZE = (100,110)
            NUM_MASKS = 6
            masks_mov = [torch.tensor(Image.open("{}/test0_mask{}.png".format(FOLDERNAME, idx)).resize(MASK_SIZE).getdata(),dtype=torch.bool).reshape(MASK_SIZE) for idx in range(NUM_MASKS)]
            masks_fix = [torch.tensor(Image.open("{}/test1_mask{}.png".format(FOLDERNAME, idx)).resize(MASK_SIZE).getdata(),dtype=torch.bool).reshape(MASK_SIZE) for idx in range(NUM_MASKS)]
            masks_mov = torch.stack(masks_mov,dim=0)
            masks_fix = torch.stack(masks_fix,dim=0)
        case '3d':
            FOLDERNAME = './data/3d'
            NUM_MASKS = 8
            masks_mov = torch.stack([torch.tensor(nib.load("{}/test0_mask{}.nii.gz".format(FOLDERNAME,idx)).get_fdata()) for idx in range(NUM_MASKS)], dim=0).to(torch.bool)
            masks_fix = torch.stack([torch.tensor(nib.load("{}/test1_mask{}.nii.gz".format(FOLDERNAME,idx)).get_fdata()) for idx in range(NUM_MASKS)], dim=0).to(torch.bool)
        case _:
            raise ValueError("Unknown type: {}".format(type))
    
    return masks_mov, masks_fix


def save_test_data(masks_warped):    
    if masks_warped.ndim == 4:  # 3d        
        FOLDERNAME = './data/3d'
        for idx in range(masks_warped.shape[0]):
            nib.save(nib.Nifti1Image(masks_warped[idx].cpu().numpy(),affine=torch.eye(4).numpy()), "{}/warped_mask{}.nii.gz".format(FOLDERNAME,idx))
        print("Saved {}/warped_mask*.nii.gz".format(FOLDERNAME))
    elif masks_warped.ndim == 3:  # 2d
        FOLDERNAME = './data/2d'
        for idx in range(masks_warped.shape[0]):
            Image.fromarray(masks_warped[idx].cpu().numpy()).save("{}/warped_mask{}.png".format(FOLDERNAME,idx))
        print("Saved {}/warped_mask*.png".format(FOLDERNAME))
