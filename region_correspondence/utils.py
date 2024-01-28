
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
    sample_grid: torch.tensor of shape (D1,H1,W1,3) where the 0th-dim is the displacement vector
    Returns a warped image of shape (C,D1,H1,W1)
    '''
    warped = torch.nn.functional.grid_sample(
        input=vol.unsqueeze(0),
        grid=sample_grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (C,W1,H1,D1)
    return warped.squeeze(0)


class ROILoss():
    def __init__(self, w_overlap=1.0, w_class=1.0) -> None:
        self.w_overlap = w_overlap
        self.w_class = w_class

    def __call__(self, roi0, roi1):
        loss = 0
        if self.w_overlap != 0:
            loss += self.w_overlap * self.overlap_loss_cross(roi0, roi1)
        if self.w_class != 0:
            loss += self.w_class * self.class_loss(roi0, roi1)
        return loss
    
    
    def overlap_loss_cross(self, roi0, roi1, eps=1e-8):
        '''
        Implements Dice as the overlap loss cross all masks
        roi0: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        roi1: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        '''
        intersection = (roi0 * roi1).sum()
        union = roi0.sum() + roi1.sum()
        overlap = 2*intersection / (union+eps)
        return 1 - overlap

    def overlap_loss(self, roi0, roi1, eps=1e-8):
        '''
        Implements Dice as the overlap loss
        roi0: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        roi1: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        '''
        intersection = (roi0 * roi1).sum(dim=(1,2,3))
        union = roi0.sum(dim=(1,2,3)) + roi1.sum(dim=(1,2,3))
        overlap = 2*intersection / (union+eps)
        return 1 - overlap.mean()
    
    def class_loss(self, roi0, roi1):
        '''
        Implements mean-square-error as the classification loss (cross entropy may not be suitable for this task due to extreme values in class probabilities)
        roi0: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        roi1: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        '''
        mse = ((roi0 - roi1)**2).mean(dim=(1,2,3))
        return mse.mean()


class DDFLoss():
    def __init__(self, type='l2norm') -> None:
        self.type = type

    def __call__(self, ddf):
        '''
        fix: torch.tensor of shape (C,H1,W1,D1) where C is the number of masks
        warped: torch.tensor of shape (C,H1,W1,D1) where C is the number of masks
        ddf: torch.tensor of shape (3,H1,W1,D1) where the 0th-dim is the displacement vector
        '''
        if self.type == "l2norm":
            loss = self.l2_gradient(ddf) * (-1)
        elif self.type == "bending":
            loss = self.bending_energy(ddf) * (-1)
        else:
            raise ValueError(f"Unknown DDFLoss type: {self.type}")
        return loss
    
    def l2_gradient(self, ddf):
        '''
        ddf: torch.tensor of shape (3,H1,W1,D1) where the 0th-dim is the displacement vector
        '''
        ddf_grad = torch.stack(torch.gradient(ddf), dim=0)
        l2_grad = torch.norm(ddf_grad, dim=0)
        return l2_grad.mean()
    
    def bending_energy(self, ddf):
        '''
        ddf: torch.tensor of shape (3,H1,W1,D1) where the 0th-dim is the displacement vector
        '''
        raise NotImplementedError("bending_energy is not implemented yet")


