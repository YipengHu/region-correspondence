# Test data are from https://zenodo.org/records/7013610, see make_test_data.py for details.

import torch

from make_test_data import load_test_data, save_test_data
from region_correspondence.paired_regions import PairedRegions
from region_correspondence.gridded import warp_by_ddf


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# read test masks
masks_mov, masks_fix = load_test_data('3d')  # alternatively, '2d'

# estimate dense correspondence
paired_rois = PairedRegions(masks_mov=masks_mov, masks_fix=masks_fix, device=device)
ddf = paired_rois.get_dense_correspondence()
#ddf = paired_rois.get_dense_correspondence(transform_type='ddf', pre_alignment=None, max_iter=int(1e4), lr=1e-3, w_ddf=1.0, verbose=True)
#ddf = paired_rois.get_dense_correspondence(transform_type='affine')

# save warped ROIs for visulisation
masks_warped = (warp_by_ddf(masks_mov.to(dtype=torch.float32, device=device), ddf)*255).to(torch.uint8)
save_test_data(masks_warped) 
