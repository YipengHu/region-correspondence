# Implements transformation on image point-based features

import torch

from region_correspondence.control import get_reference_grid


def get_foreground_coordinates(masks, device=None):
   '''
    masks: torch.tensor of shape (C,D,H,W) where C is the number of masks   
    Returns a list of length C, where each element is a torch.tensor of shape (N,3) for 3d
                                                                       (N,2) for 2d
   '''
   
   grid = get_reference_grid(grid_size=masks.shape[1:], device=device)
   coords = mask_indexing(grid, masks)

   return coords


def mask_indexing(grid, masks):
    '''
    grid: torch.tensor of shape (D,H,W,3) where dim=3 is the coordinate vector xyz (<- ijk)
          torch.tensor of shape (H,W,2) where dim=2 is the coordinate vector xy (<- ij)
    masks: torch.tensor of shape (C,D,H,W) where C is the number of masks
           torch.tensor of shape (C,H,W) where C is the number of masks
    Returns a list of length C, where each element is a torch.tensor of shape (N,3) for 3d
                                                                       (N,2) for 2d
    '''
    return [torch.stack([grid[...,d][masks[n,...]] for d in range(grid.shape[-1])],dim=-1) for n in range(masks.shape[0])]

def get_foreground_centroids(masks, device=None):
    '''
    masks: torch.tensor of shape (C,D,H,W) where C is the number of masks
    Returns a torch.tensor of shape (C,3) for 3d
                                (C,2) for 2d
    '''
    coords = get_foreground_coordinates(masks, device)
    return torch.stack([c.mean(dim=0) for c in coords],dim=0)