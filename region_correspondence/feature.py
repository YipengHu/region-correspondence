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


def ddf_by_affine(grid_size, affine_matrix, translation, inverse=True, device=None):
    '''
    grid_size: tuple of 3 ints for 3d, tuple of 2 ints for 2d
    affine_matrix: torch.tensor of shape (3,3) for 3d, (2,2) for 2d
    translation: torch.tensor of shape (3) for 3d, (2) for 2d
    Returns a dense displacement field (DDF) of shape (D,H,W,3) where the dim=3 is the displacement vector, such that
        if inverse: mov_grid @ affine_matrix + translation = fix_grid 
                    => (fix_grid - translation) @ affine_matrix^(-1) = mov_grid = fix_grid + ddf
                    => ddf = [fix_grid @ affine_matrix^(-1) - fix_grid] - translation @ affine_matrix^(-1)
         otherwise: fix_grid @ affine_matrix + translation = mov_grid = fix_grid + ddf 
                    => ddf = [fix_grid @ affine_matrix - fix_grid] + translation
    '''
    grid = get_reference_grid(grid_size, device)
    if inverse:
        affine_matrix_inv = torch.linalg.inv(affine_matrix)
        ddf = grid @ (affine_matrix_inv  - torch.eye(grid.dim()-1,device=device)) - translation @ affine_matrix_inv
    else:
        ddf = grid @ (affine_matrix  - torch.eye(grid.dim()-1,device=device)) + translation
    return ddf


def ls_affine(mov_pts, fix_pts):
    '''
    Implements least squares estimation of affine transformation between two sets of points
        such that |mov_pts @ affine_matrix + translation - fix_pts|^2 is minimised
    mov_pts: torch.tensor of shape (N,3) for 3d, where N is the number of points, (N,2) for 2d
    fix_pts: torch.tensor of shape (N,3) for 3d, where N is the number of points, (N,2) for 2d
    Returns affine_matrix: torch.tensor of shape (3,3) for 3d, (2,2) for 2d
            translation: torch.tensor of shape (3) for 3d, (2) for 2d
    '''
    fix_cent = fix_pts.mean(dim=0) 
    mov_cent = mov_pts.mean(dim=0) 
    affine_matrix = torch.linalg.lstsq(mov_pts-mov_cent, fix_pts-fix_cent).solution 
    translation = fix_cent - mov_cent @ affine_matrix
    return affine_matrix, translation


def ls_rigid(mov_pts, fix_pts):
    '''
    Implements least squares estimation of rigid transformation (rotation and translation) between two sets of points
        such that |fix_pts - (mov_pts @ rotation_matrix + translation)|^2 is minimised, where rotation_matrix is orthogonal
    mov_pts: torch.tensor of shape (N,3) for 3d, where N is the number of points, (N,2) for 2d
    fix_pts: torch.tensor of shape (N,3) for 3d, where N is the number of points, (N,2) for 2d
    Returns rotation_matrix: torch.tensor of shape (3,3) for 3d, (2,2) for 2d
            translation: torch.tensor of shape (3) for 3d, (2) for 2d
    '''
    fix_cent = fix_pts.mean(dim=0) 
    mov_cent = mov_pts.mean(dim=0) 
    mov_centered = mov_pts - mov_cent
    fix_centered = fix_pts - fix_cent
    U, _, V = torch.svd(fix_centered.t() @ mov_centered)
    rotation_matrix = V @ U.t()
    translation = fix_cent - mov_cent @ rotation_matrix
    return rotation_matrix, translation