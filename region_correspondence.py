
import torch


class PairedRegions():

    def __init__(self, masks_mov, masks_fix):
        '''
        masks_mov: torch.tensor of shape (i0,j0,k0,n) where n is the number of masks
        masks_fix: torch.tensor of shape (i1,i1,i1,n) where n is the number of masks
        '''
        self.masks_mov = masks_mov
        self.masks_fix = masks_fix

    def get_dense_correspondence(self):
        '''
        Returns a dense displacement field (DDF) of shape (i1,j1,k1,3) where the last dimension is the displacement vector
        '''
        ddf = torch.zeros(self.masks_fix.shape[:-1]+(3,))
    
        return ddf
     