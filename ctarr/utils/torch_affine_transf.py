import torch
import torch.nn.functional as F
import numpy as np

class AffineTransf(torch.nn.Module):    
    def __init__(self, 
                 a_init=np.ones(3),
                 b_init=np.zeros(3),
                 len_per_ax=[-1, -1, -1]):
        super().__init__()
        self.a_init = a_init
        self.b_init = b_init
        self.len_per_ax = len_per_ax
        
        if isinstance(self.a_init, np.ndarray):
            self.a_init = torch.from_numpy(self.a_init).float()
        if isinstance(self.b_init, np.ndarray):
            self.b_init = torch.from_numpy(self.b_init).float()
        if isinstance(self.len_per_ax, np.ndarray):
            self.len_per_ax = torch.from_numpy(self.len_per_ax).float()
        
        self.initialise_params()
    
    def initialise_params(self):
        self.a = torch.nn.Parameter(self.a_init)
        self.b = torch.nn.Parameter(-2*self.b_init/self.len_per_ax)
    
    def get_lin_params(self):
        ai = self.a.detach().cpu().numpy()
        bi = self.b.detach().cpu().numpy()
        lpa = self.len_per_ax.cpu().numpy()
        return 1/ai, -0.5*lpa * bi / ai
    
    def forward(self, xb):
        
        theta = torch.zeros((1,3,4), device=xb.device, dtype=xb.dtype)
        theta[0, [0,1,2], [0,1,2]] = self.a
        theta[0, :, -1] = self.b
        
        grid = F.affine_grid(theta, xb.shape)        
        return F.grid_sample(xb, grid)
