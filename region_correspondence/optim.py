# implements the optimisation methods for region correspondence estimation

import torch

from region_correspondence.metrics import DDFLoss, ROILoss
from region_correspondence.gridded import get_reference_grid, upsample_control_grid, sampler
from region_correspondence.scattered import get_foreground_centroids, get_foreground_bounding_corners, affine_to_ddf, ls_affine, ls_rigid


def gridded_transform(mov, fix, control_grid_size=None, device=None, max_iter=int(1e4), lr=1e-3, w_ddf=1.0, verbose=True):
    '''
    Compute a transformation based on gridded control points (control_grid), using an iterative optimisation
        when control_grid_size = None, the dense displacement field (DDF) estimation is estimated using the iterative optimisation
    mov: torch.tensor of shape (C,D0,H0,W0) for 3d, where C is the number of masks, (C,H0,W0) for 2d
    fix: torch.tensor of shape (C,D1,H1,W1) for 3d, where C is the number of masks, (C,H1,W1) for 2d
    control_grid_size: 
        None for DDF estimation
        when specified, tuple of 3 ints for 3d, tuple of 2 ints for 2d, or tuple of 1 int for the same size in all dimensions
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the dim=3 contains the displacement vectors
    '''
    if isinstance(control_grid_size, int):
        if mov.dim() == 4:
            control_grid_size = (
                control_grid_size, control_grid_size, control_grid_size)
        elif mov.dim() == 3:
            control_grid_size = (control_grid_size, control_grid_size)

    if verbose:
        if control_grid_size is None:
            print("Optimising DDF (dense displacement field):")
        elif len(control_grid_size) == 3:
            print("Optimising gridded control points with a grid size of ({},{},{}):".format(
                *control_grid_size))
        elif len(control_grid_size) == 2:
            print("Optimising gridded control points with a grid size of ({},{}):".format(
                *control_grid_size))

    ref_grid = get_reference_grid(grid_size=fix.shape[1:], device=device)
    if control_grid_size is not None:
        control_grid = get_reference_grid(
            grid_size=control_grid_size, device=device)
    else:  # ddf
        control_grid = get_reference_grid(
            grid_size=ref_grid.shape[:-1], device=device)
    control_grid += torch.normal(mean=0, std=1e-5, size=control_grid.shape,
                                 dtype=torch.float32, device=device)  # initialise to break symmetry
    control_grid.requires_grad = True

    optimizer = torch.optim.Adam(params=[control_grid], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.1)
    loss_ddf = DDFLoss(type='bending')
    metric_overlap = ROILoss(w_overlap=1.0, w_class=0.0)

    for iter in range(max_iter):

        optimizer.zero_grad()

        if control_grid_size is not None:
            sample_grid = upsample_control_grid(control_grid, ref_grid)
        else:  # ddf
            sample_grid = control_grid
        warped = sampler(mov, sample_grid)
        ddf = sample_grid-ref_grid

        loss_value_roi = loss_roi(warped, fix)
        loss_value_ddf = loss_ddf(ddf)
        loss = loss_value_roi + loss_value_ddf*w_ddf

        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (deform={:0.5f}, overlap={:0.5f})".format(
                    iter, loss, loss_value_ddf, 1-metric_overlap(warped, fix)))

        loss.backward()
        optimizer.step()

        # TODO configurable convergence criterion based on metric_overlap

    return ddf, control_grid


def scattered_transform(mov, fix, parametric_type='rigid', control_point_type='bounding', device=None):
    '''
    Compute a parametric transformation, using scattered control points 
    mov: torch.tensor of shape (C,D0,H0,W0) for 3d, where C is the number of masks, (C,H0,W0) for 2d
    fix: torch.tensor of shape (C,D1,H1,W1) for 3d, where C is the number of masks, (C,H1,W1) for 2d
    parametric_type: str, one of ["affine", "rigid", "rigid7"]

    '''
    if parametric_type not in ["affine", "rigid", "rigid7"]:
        raise ValueError("Unknown transform type: {}".format(parametric_type))
    if control_point_type not in ["bounding", "centroid", "voxel", "surface"]:
        raise ValueError("Unknown feature type: {}".format(control_point_type))

    if control_point_type == "bounding":
        mov_control_points = get_foreground_bounding_corners(mov, device=device)
        fix_control_points = get_foreground_bounding_corners(fix, device=device)
    elif control_point_type == "centroid":
        mov_control_points = get_foreground_centroids(mov, device=device)
        fix_control_points = get_foreground_centroids(fix, device=device)
        Warning("Centroid-based control points are not robust for symmetric regions.")
    elif control_point_type == "surface":
        raise NotImplementedError(
            "Surface feature type is not implemented yet.")
    else:
        raise NotImplementedError(
            "Control point type {} is not implemented yet.".format(control_point_type))

    if parametric_type == "rigid":
        affine_inv, translation_inv = ls_rigid(
            fix_control_points, mov_control_points)
    elif parametric_type == "affine":
        affine_inv, translation_inv = ls_affine(
            fix_control_points, mov_control_points)
    else:
        raise NotImplementedError(
            "Parametric_type {} is not implemented yet.".format(parametric_type))

    ddf = affine_to_ddf(
        grid_size=fix.shape[1:], affine_matrix=affine_inv, translation=translation_inv, device=device)

    return ddf
