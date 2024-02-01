# Test data are from https://zenodo.org/records/7013610, see make_test_data.py for details.

import nibabel as nib
import torch

from region_correspondence.paired_regions import PairedRegions
from region_correspondence.utils import warp_by_ddf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# read test masks
masks_mov = torch.stack([torch.tensor(nib.load("./data/test0_mask{}.nii.gz".format(idx)).get_fdata()) for idx in range(8)], dim=0).to(torch.bool)
masks_fix = torch.stack([torch.tensor(nib.load("./data/test1_mask{}.nii.gz".format(idx)).get_fdata()) for idx in range(8)], dim=0).to(torch.bool)

paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=device)
ddf = paired_rois.get_dense_correspondence(transform_type='ddf', max_iter=int(1e4), lr=1e-3, w_ddf=1.0, verbose=True)
# TODO: adjust w_ddf for desired smoothness of the dense correspondence  

## save warped ROIs for visulisation
masks_warped = (warp_by_ddf(masks_mov.to(dtype=torch.float32, device=device), ddf)*255).to(torch.uint8)
for idx in range(masks_warped.shape[0]):
    nib.save(nib.Nifti1Image(masks_warped[idx].cpu().numpy(),affine=torch.eye(4).numpy()), "./data/warped_mask{}.nii.gz".format(idx))
print("Saved ./data/warped_mask*.nii.gz")
