
import torch

from region_correspondence.optim import iterative_ddf


class PairedRegions():

    def __init__(self, masks_mov, masks_fix):
        '''
        masks_mov: torch.tensor of shape (C,H0,W0,D0) where C is the number of masks
        masks_fix: torch.tensor of shape (C,H1,W1,D1) where C is the number of masks
        '''
        self.masks_mov = masks_mov
        self.masks_fix = masks_fix
        self.method = "iterative_ddf"

    def get_dense_correspondence(self):
        '''
        Returns a dense displacement field (DDF) of shape (3,H1,W1,D1) where the 0th-dim is the displacement vector
        '''
        ddf = iterative_ddf(mov=self.masks_mov, fix=self.masks_fix)
    
        return ddf
    