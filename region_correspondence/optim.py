
import torch

from region_correspondence.utils import get_reference_grid, warp_by_ddf, upsample_control_grid, sampler
from region_correspondence.metrics import DDFLoss, ROILoss


def ddf_iterative(mov, fix, device=None, max_iter=int(1e7), lr=1e-4, w_ddf=0.1, verbose=False):
    '''
    Implements the direct dense displacement field (DDF) estimation using iterative optimisation
    mov: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    fix: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the 3rd-dim contains the displacement vectors
    '''
    num_masks = mov.shape[0]
    if num_masks != fix.shape[0]:
        raise ValueError("mov and fix must have the same number of masks")
    
    ddf = torch.normal(mean=0, std=1e-3, size=fix.shape[1:]+(3,), dtype=torch.float32, requires_grad=True, device=device)
    ref_grid = get_reference_grid(ddf.shape[:-1], device=device)    
    optimizer = torch.optim.Adam(params=[ddf], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.0) 
    loss_ddf = DDFLoss(type='l2norm')

    for iter in range(max_iter):
        
        optimizer.zero_grad()

        warped = warp_by_ddf(mov, ddf, ref_grid=ref_grid)

        loss_value_roi = loss_roi(warped,fix)
        loss_value_ddf = loss_ddf(ddf)
        loss = loss_value_roi + loss_value_ddf*w_ddf
        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (roi={:0.5f}, ddf={:0.5f})".format(iter, loss, loss_value_roi, loss_value_ddf))
        
        loss.backward()
        optimizer.step()
    
    return ddf


def ffd_iterative(mov, fix, control_grid_size=(5,5,5), device=None, max_iter=int(1e7), lr=1e-4, w_ddf=0.1, verbose=False):
    '''
    Implements the free-form deformation (FFD) estimation based on control point grid (control_grid), using iterative optimisation
    mov: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    fix: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
    control_grid_size: tuple of 3 ints
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the 3rd-dim contains the displacement vectors
    '''
    num_masks = mov.shape[0]
    if num_masks != fix.shape[0]:
        raise ValueError("mov and fix must have the same number of masks")
    
    ref_grid = get_reference_grid(grid_size=fix.shape[1:], device=device)  #pre-compute
    control_grid = get_reference_grid(grid_size=control_grid_size, device=device) #initialisation
    control_grid.requires_grad = True
    
    optimizer = torch.optim.Adam(params=[control_grid], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.0) 
    loss_ddf = DDFLoss(type='l2norm')

    for iter in range(max_iter):
        
        optimizer.zero_grad()

        sample_grid = upsample_control_grid(control_grid, ref_grid)
        warped = sampler(mov, sample_grid)

        loss_value_roi = loss_roi(warped,fix)
        loss_value_ddf = loss_ddf(sample_grid-ref_grid)
        loss = loss_value_roi + loss_value_ddf*w_ddf
        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (roi={:0.5f}, ddf={:0.5f})".format(iter, loss, loss_value_roi, loss_value_ddf))
        
        loss.backward()
        optimizer.step()
    
    return control_grid, sample_grid-ref_grid