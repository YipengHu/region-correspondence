# implements the optimisation methods for region correspondence estimation

import torch

from region_correspondence.metrics import DDFLoss, ROILoss
from region_correspondence.gridded import get_reference_grid, upsample_control_grid, sampler
from region_correspondence.scattered import get_foreground_centroids, ddf_by_affine, ls_affine, ls_rigid


def gridded_transform(mov, fix, control_grid_size=None, device=None, max_iter=int(1e5), lr=1e-3, w_ddf=1.0, verbose=False):
    '''
    Optimise a transformation based on gridded control points (control_grid), using an iterative algorithm
        when control_grid_size = None, the dense displacement field (DDF) estimation is estimated using the iterative optimisation
    mov: torch.tensor of shape (C,D0,H0,W0) for 3d, where C is the number of masks, (C,H0,W0) for 2d
    fix: torch.tensor of shape (C,D1,H1,W1) for 3d, where C is the number of masks, (C,H1,W1) for 2d
    control_grid_size: 
        None for DDF estimation
        when specified, tuple of 3 ints for 3d, tuple of 2 ints for 2d, or tuple of 1 int for the same size in all dimensions
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the dim=3 contains the displacement vectors
    '''
    if isinstance(control_grid_size,int):
        if mov.dim() == 4:
            control_grid_size = (control_grid_size,control_grid_size,control_grid_size)
        elif mov.dim() == 3:
            control_grid_size = (control_grid_size,control_grid_size)

    if verbose:
            if control_grid_size is None:
                print("Optimising DDF (dense displacement field):")
            elif len(control_grid_size) == 3:
                print("Optimising FFD (free-form deformation) with control grid size ({},{},{}):".format(*control_grid_size))
            elif len(control_grid_size) == 2:
                print("Optimising FFD (free-form deformation) with control grid size ({},{}):".format(*control_grid_size))
    
    ref_grid = get_reference_grid(grid_size=fix.shape[1:], device=device)
    if control_grid_size is not None:
        control_grid = get_reference_grid(grid_size=control_grid_size, device=device)
    else:  #ddf
        control_grid = get_reference_grid(grid_size=ref_grid.shape[:-1], device=device)
    control_grid += torch.normal(mean=0, std=1e-5, size=control_grid.shape, dtype=torch.float32, device=device)  # initialise to break symmetry
    control_grid.requires_grad = True

    optimizer = torch.optim.Adam(params=[control_grid], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.1) 
    loss_ddf = DDFLoss(type='bending')
    metric_overlap = ROILoss(w_overlap=1.0, w_class=0.0)

    for iter in range(max_iter):
        
        optimizer.zero_grad()

        if control_grid_size is not None:
            sample_grid = upsample_control_grid(control_grid, ref_grid)
        else:  #ddf
            sample_grid = control_grid
        warped = sampler(mov, sample_grid)
        ddf = sample_grid-ref_grid

        loss_value_roi = loss_roi(warped,fix)
        loss_value_ddf = loss_ddf(ddf)
        loss = loss_value_roi + loss_value_ddf*w_ddf
        
        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (deform={:0.5f}, overlap={:0.5f})".format(iter, loss, loss_value_ddf, 1-metric_overlap(warped,fix)))
        
        loss.backward()
        optimizer.step()
    
    return ddf, control_grid 


def scattered_transform(mov, fix, transform_type='rigid', feature_type='centroid', device=None):
    '''
    Optimise a parametric transformation, using scattered control points 
    mov: torch.tensor of shape (C,D0,H0,W0) for 3d, where C is the number of masks, (C,H0,W0) for 2d
    fix: torch.tensor of shape (C,D1,H1,W1) for 3d, where C is the number of masks, (C,H1,W1) for 2d
    transform_type: str, one of ["affine", "rigid", "rigid7"]

    '''
    if transform_type not in ["affine", "rigid", "rigid7"]:
        raise ValueError("Unknown transform type: {}".format(transform_type))
    if feature_type not in ["centroid", "surface"]:
        raise ValueError("Unknown feature type: {}".format(feature_type))
    
    if feature_type == "centroid":
        mov_centroids = get_foreground_centroids(mov, device=device)
        fix_centroids = get_foreground_centroids(fix, device=device)
        if transform_type == "rigid":
            affine_matrix, translation = ls_rigid(mov_centroids, fix_centroids)
        elif transform_type == "affine":
            affine_matrix, translation = ls_affine(mov_centroids, fix_centroids)
        else:
            raise NotImplementedError("Transform_type {} with feature type {} with is not implemented yet.".format(transform_type,feature_type))
    elif feature_type == "surface":
        raise NotImplementedError("Surface feature type is not implemented yet.")
    else:
        raise NotImplementedError("Transform_type {} with feature type {} with is not implemented yet.".format(transform_type,feature_type))
    
    ddf = ddf_by_affine(grid_size=fix.shape[1:], affine_matrix=affine_matrix, translation=translation, inverse=True, device=device)

    return ddf, affine_matrix, translation
