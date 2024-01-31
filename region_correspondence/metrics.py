# Implementations of loss functions and metrics that are useful for both estimation and evaluation
import torch


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

        d2Fdxx, d2Fdxy, d2Fdxz = self.ddf_gradients(dFdx)
        d2Fdyx, d2Fdyy, d2Fdyz = self.ddf_gradients(dFdy)
        d2Fdzx, d2Fdzy, d2Fdzz = self.ddf_gradients(dFdz)

        bending_energy = d2Fdxx**2 + d2Fdyy**2 + d2Fdzz**2 + \
            2*d2Fdxy*d2Fdyx + 2*d2Fdxz*d2Fdzx + 2*d2Fdyz*d2Fdzy
        
        return bending_energy.mean()
    
    @staticmethod
    def ddf_gradients(ddf):
        '''
        computes ddf gradients
        '''
        dXdx, dXdy, dXdz = torch.gradient(ddf[...,0])
        dYdx, dYdy, dYdz = torch.gradient(ddf[...,1])
        dZdx, dZdy, dZdz = torch.gradient(ddf[...,1])
        dFdx = torch.stack([dXdx, dYdx, dZdx], dim=3)
        dFdy = torch.stack([dXdy, dYdy, dZdy], dim=3)
        dFdz = torch.stack([dXdz, dYdz, dZdz], dim=3)
        return dFdx, dFdy, dFdz
