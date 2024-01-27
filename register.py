
import torch

def iterative_ddf(mov, fix, max_iter=1e3, lr=0.1, sigma=1.0, alpha=0.1, beta=0.1, gamma=0.1, verbose=False):
    '''
    mov: torch.tensor of shape (C,H0,W0,D0) where C is the number of masks
    fix: torch.tensor of shape (C,H1,W1,D1) where C is the number of masks
    Returns a dense displacement field (DDF) of shape (3,H1,W1,D1) where the 0th-dim contains the displacement vectors
    '''
    ddf = torch.zeros((3,)+fix.shape[1:])
    num_masks = mov.shape[0]
    if num_masks != fix.shape[0]:
        raise ValueError("mov and fix must have the same number of masks")
    
    ref_grids = get_reference_grid(ddf.shape[1:])
    
    optimizer = torch.optim.SGD(ddf, lr=lr, momentum=0.9)

    for iter in range(max_iter):
        optimizer.zero_grad()

        warped_grids = ref_grids.repeat(ddfs.shape[0],1,1,1) + ddfs
        warped_grids = warped_grids.permute(0,2,3,1)[...,[1,0]]
        warped = torch.nn.functional.grid_sample(mov, warped_grids, align_corners=False)  # warping images in each channel

        loss = loss_fn(fix,warped,ddf)
        
        loss.backward()
        optimizer.step()
    
    return ddf

def get_reference_grid(grid_size):
    # grid_size: [height, width, depth]
    return torch.stack(torch.meshgrid(
        torch.linspace(-1,1,grid_size[0]),
        torch.linspace(-1,1,grid_size[1]),
        torch.linspace(-1,1,grid_size[2]),
        indexing='ij',
        ), 
        dim=0)