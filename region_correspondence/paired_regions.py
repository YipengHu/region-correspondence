
import torch

from region_correspondence.optim import ddf_iterative, ffd_iterative


class PairedRegions():

    def __init__(self, masks_mov, masks_fix, device=None):
        '''
        masks_mov: torch.tensor of shape (C,D0,H0,W0) where C is the number of masks
        masks_fix: torch.tensor of shape (C,D1,H1,W1) where C is the number of masks
        '''
        self.masks_mov = masks_mov
        self.masks_fix = masks_fix
        self.device = device
        if self.device is not None:
            self.masks_mov = self.masks_mov.to(device)
            self.masks_fix = self.masks_fix.to(device)
        self.method = "iterative_ddf"

    def get_dense_correspondence(self, transform_type='ddf', **kwargs):
        '''
        transform_type: str, one of ['ddf', 'ffd', 'affine', 'spline']
            ddf implements the direct dense displacement field. 
            ffd implements the free-form deformation based on control points.
        Returns a dense displacement field (DDF) of shape (H1,W1,D1,3) where the 0th-dim is the displacement vector
        '''
        match transform_type.lower():
            case 'ddf': # direct dense displacement field
                self.ddf = ddf_iterative(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), device=self.device, **kwargs)  # grid_sample requires float32
            case 'ffd': # control point based free-form deformation
                self.control_points, self.ddf = ddf_iterative(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), device=self.device, **kwargs)  # grid_sample requires float32
            case 'affine':
                raise NotImplementedError("TPS transform is not implemented yet.")
            case 'spline':
                raise NotImplementedError("TPS transform is not implemented yet.")
            case _:
                raise ValueError("Unknown transform type: {}".format(transform_type))
        
        return self.ddf
    