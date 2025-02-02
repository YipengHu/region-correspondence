
import torch

from region_correspondence.optim import gridded_transform, scattered_transform


class PairedRegions():

    def __init__(self, masks_mov, masks_fix, device=None):
        '''
        masks_mov: torch.tensor of shape (C,D0,H0,W0) for 3d, 
                            (C,H0,W0) for 2d, where C is the number of masks
        masks_fix: torch.tensor of shape (C,D1,H1,W1) for 3d, 
                            (C,H1,W1) for 2d, where C is the number of masks
        '''
        self.check_input(masks_mov, masks_fix)
        self.masks_mov = masks_mov
        self.masks_fix = masks_fix
        self.device = device
        if self.device is not None:
            self.masks_mov = self.masks_mov.to(device)
            self.masks_fix = self.masks_fix.to(device)
    
    def check_input(self, mov, fix):
        if mov.shape[0] != fix.shape[0]:
            raise ValueError("mov and fix must have the same number of masks.")
        if mov.dim() != fix.dim():
            raise ValueError("mov and fix must have the same dimensionality.")


    def get_dense_correspondence(self, transform_type='ffd', **kwargs):
        '''
        transform_type: str, one of ['ddf', 'ffd', 'affine']
            ddf implements the direct dense displacement field optimisation. 
            ffd implements the free-form deformation based on a control point grid.
        Returns a dense displacement field (DDF) of shape (H1,W1,D1,3) where the dim=0 is the displacement vector
        '''
        match transform_type.lower():
            case 'ddf':
                self.ddf, _ = gridded_transform(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), control_grid_size=None, device=self.device, **kwargs)  # grid_sample requires float32
            case 'ffd':
                self.ddf, self.control_grid = gridded_transform(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), control_grid_size=10, device=self.device, **kwargs) 
            case 'affine': 
                self.ddf, self.affine_matrix, self.translation = scattered_transform(mov=self.masks_mov, fix=self.masks_fix, parametric_type=transform_type, device=self.device, **kwargs)
            case 'spline':
                raise NotImplementedError("TPS transform is not implemented yet.")
            case _:
                raise ValueError("Unknown transform type: {}".format(transform_type))
        
        return self.ddf
