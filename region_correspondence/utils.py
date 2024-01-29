
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


class ROILoss():
    def __init__(self, w_overlap=1.0, w_class=1.0) -> None:
        self.w_overlap = w_overlap
        self.w_class = w_class

    def __call__(self, roi0, roi1):
        '''
        Implements Dice as the overlap loss cross all masks
        roi0: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        roi1: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        '''
        loss = 0
        if self.w_overlap != 0:
            loss += self.w_overlap * self.overlap_loss_cross(roi0, roi1)
        if self.w_class != 0:
            loss += self.w_class * self.class_loss(roi0, roi1)
        return loss
    
    def overlap_loss_cross(self, roi0, roi1, eps=1e-8):
        intersection = (roi0 * roi1).sum()
        union = roi0.sum() + roi1.sum()
        overlap = 2*intersection / (union+eps)
        return 1 - overlap

    def overlap_loss(self, roi0, roi1, eps=1e-8):
        '''
        Implements Dice as the overlap loss
        '''
        intersection = (roi0 * roi1).sum(dim=(1,2,3))
        union = roi0.sum(dim=(1,2,3)) + roi1.sum(dim=(1,2,3))
        overlap = 2*intersection / (union+eps)
        return 1 - overlap.mean()
    
    def class_loss(self, roi0, roi1):
        '''
        Implements mean-square-error as the classification loss
        '''
        mse = ((roi0 - roi1)**2).mean(dim=(1,2,3))
        return mse.mean()


class DDFLoss():
    def __init__(self, type='l2norm') -> None:
        self.type = type

    def __call__(self, ddf):
        '''
        ddf: torch.tensor of shape (H1,W1,D1,3)
        '''
        if self.type == "l2norm":
            loss = self.l2_gradient(ddf)
        elif self.type == "bending":
            loss = self.bending_energy(ddf)
        else:
            raise ValueError(f"Unknown DDFLoss type: {self.type}")
        return loss
    
    def l2_gradient(self, ddf):
        '''
        implements L2-norm over the ddf gradients
        '''
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)
        grad_norms = dFdx**2 + dFdy**2 + dFdz**2
        return grad_norms.mean()
    
    def bending_energy(self, ddf):
        '''
        implements bending energy estimated over the ddf
        '''
        dFdx, dFdy, dFdz = self.ddf_gradients(ddf)

        dFdx2 = torch.stack(torch.gradient(dFdx[...,0]), dim=3)
        dFdy2 = torch.stack(torch.gradient(dFdy[...,1]), dim=3)
        dFdz2 = torch.stack(torch.gradient(dFdz[...,2]), dim=3)
        #TODO: a bit waste computing the same partial derivatives, but using torch.gradient() may be faster

        bending_energy = dFdx2[...,0]**2 + dFdy2[...,1]**2 + dFdz2[...,2]**2 + \
            2*dFdx2[...,1]*dFdy2[...,0] + 2*dFdx2[...,2]*dFdz2[...,0] + 2*dFdy2[...,2]*dFdz2[...,1]
        
        raise bending_energy.mean()
    
    def ddf_gradients(ddf):
        '''
        computes ddf gradients
        '''
        dFdx = torch.stack(torch.gradient(ddf[...,0]), dim=3)
        dFdy = torch.stack(torch.gradient(ddf[...,1]), dim=3)
        dFdz = torch.stack(torch.gradient(ddf[...,2]), dim=3)
        return dFdx, dFdy, dFdz
