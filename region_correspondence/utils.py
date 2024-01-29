
import torch


def get_reference_grid(grid_size, device=None):
    # grid_size: [depth, height, width]
    return torch.stack(torch.meshgrid(
        torch.linspace(-1,1,grid_size[0]),
        torch.linspace(-1,1,grid_size[1]),
        torch.linspace(-1,1,grid_size[2]),
        indexing='ij',
        ), dim=3).to(device)[...,[2,1,0]]  # reverse: ijk->xyz


def sampler(vol, sample_grid):
    '''
    vol: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    sample_grid: torch.tensor of shape (D1,H1,W1,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    Returns a warped image of shape (C,D1,H1,W1)
    '''
    warped = torch.nn.functional.grid_sample(
        input=vol.unsqueeze(0),
        grid=sample_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ) 
    return warped.squeeze(0)


def warp_by_ddf(vol, ddf, ref_grid=None):
    '''
    vol: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
    ddf: torch.tensor of shape (D1,H1,W1,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    ref_grid: optional - torch.tensor of shape (D1,H1,W1,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    Returns a warped image of shape (C,D1,H1,W1)
    '''
    if ref_grid is None:
        ref_grid = get_reference_grid(ddf.shape[:-1], device=ddf.device)
    warped_grid = ref_grid + ddf
    warped = sampler(vol, warped_grid)
    return warped


def upsample_control_grid(control_grid, ref_grid):
    '''
    implements the up-sampling of the control grid to the sampling grid with linear interpolation
    control_grid: torch.tensor of shape (D,H,W,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    ref_grid: torch.tensor of shape (D1,H1,W1,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    Returns a sample_grid of shape (D1,H1,W1,3) where the 3rd-dim is the displacement vector xyz (<- ijk)
    '''
    sample_grid = sampler(control_grid.permute(3,0,1,2),ref_grid).permute(1,2,3,0)
    return sample_grid
