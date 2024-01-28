
import torch

from region_correspondence.utils import get_reference_grid, sampler, DDFLoss, ROILoss


def iterative_ddf(mov, fix, max_iter=int(1e7), lr=1e-4, w_ddf=0.1, device=None, verbose=False):
    '''
    mov: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    fix: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
    Returns a dense displacement field (DDF) of shape (D1,H1,W1,3) where the 3rd-dim contains the displacement vectors
    '''
    ddf = torch.zeros(fix.shape[1:]+(3,), dtype=torch.float32, requires_grad=True, device=device)
    num_masks = mov.shape[0]
    if num_masks != fix.shape[0]:
        raise ValueError("mov and fix must have the same number of masks")
    
    ref_grid = get_reference_grid(ddf.shape[:-1], device=device)    
    optimizer = torch.optim.Adam(params=[ddf], lr=lr)
    loss_roi = ROILoss(w_overlap=1.0, w_class=0.0) 
    loss_ddf = DDFLoss(type='l2norm')

    for iter in range(max_iter):
        
        optimizer.zero_grad()

        warped_grid = ref_grid + ddf
        warped = sampler(mov, warped_grid)

        loss_value_roi = loss_roi(warped,fix)
        loss_value_ddf = loss_ddf(ddf)
        loss = loss_value_roi + loss_value_ddf*w_ddf
        if verbose:
            if iter % 100 == 0:
                print("iter={}: loss={:0.5f} (roi={:0.5f}, ddf={:0.5f})".format(iter, loss, loss_value_roi, loss_value_ddf))
        
        loss.backward()
        optimizer.step()
    
    return ddf
