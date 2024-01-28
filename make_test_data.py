# Description: This script is used to generate test data, data source: https://zenodo.org/records/7013610


import nibabel as nib
import torch


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


FILENAME_MOV = "./data/7013610-masks/002007_mask.nii"
FILENAME_FIX = "./data/7013610-masks/001001_mask.nii"
# FILENAME_FIX = "./data/7013610-masks/002007_mask.nii" (debug for the same masks)
# FILENAME_FIX = "./data/7013610-masks/002008_mask.nii" (debug for the same sized masks)


one_hot = lambda m : torch.stack([torch.tensor(m==i) for i in range(1,int(m.max()+1))], dim=0)
# one_hot = lambda m : torch.nn.functional.one_hot(torch.tensor(m,dtype=torch.int64), num_classes=int(m.max()+1))[...,1:]  # remove the background
masks_mov = one_hot(nib.load(FILENAME_MOV).get_fdata())
masks_fix = one_hot(nib.load(FILENAME_FIX).get_fdata())

# resize - use different sizes for debug purposes
masks_mov = (volume_resize(masks_mov.to(torch.float32), new_size=(76,78,32))).to(torch.uint8)
masks_fix = (volume_resize(masks_fix.to(torch.float32), new_size=(58,60,36))).to(torch.uint8)

# save
for idx in range(masks_mov.shape[0]):
    nib.save(nib.Nifti1Image(masks_mov[idx].numpy(),affine=torch.eye(4).numpy()), "./data/test0_mask{}.nii.gz".format(idx))
    nib.save(nib.Nifti1Image(masks_fix[idx].numpy(),affine=torch.eye(4).numpy()), "./data/test1_mask{}.nii.gz".format(idx))