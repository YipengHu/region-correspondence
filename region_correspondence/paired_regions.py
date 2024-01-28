
import torch

from region_correspondence.optim import iterative_ddf


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

    def get_dense_correspondence(self, verbose=False):
        '''
        Returns a dense displacement field (DDF) of shape (H1,W1,D1,3) where the 0th-dim is the displacement vector
        '''
        ddf = iterative_ddf(mov=self.masks_mov.type(torch.float32), fix=self.masks_fix.type(torch.float32), device=self.device, verbose=verbose)  # grid_sample requires float32
    
        return ddf
    