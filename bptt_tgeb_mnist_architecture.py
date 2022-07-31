## Imports

import numpy as np
import os
import sys
sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

class RNNModule:
    """An RNN cell responsible for a single timestep.

    Args:
        inp_dim (int): Input size.
        hid_dim (int): Hidden size.
        out_dim (int): Output size.
    """
    def __init__(self, inp_dim, hid_dim, out_dim, t_vec_ih, t_vec_hh):
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.tvec_ih = t_vec_ih
        self.tvec_hh = t_vec_hh
        
        n_classes = t_vec_ih.shape[0] ## NOTE: Edit as needed!!

        ## Wih, Whh, Woh are the parameters, so we set requires_grad=True
        self.Wih = torch.empty(hid_dim, inp_dim, requires_grad=True)
        self.Whh = torch.empty(hid_dim, hid_dim, requires_grad=True)
        self.Woh = torch.empty(out_dim, hid_dim, requires_grad=True)

        ## These are the gradients on Wih, Whh, and Woh computed manually
        ## Will be compared to the gradients computed by PyTorch
        self.Wih_grad = torch.zeros_like(self.Wih)
        self.Whh_grad = torch.zeros_like(self.Whh)
        self.Woh_grad = torch.zeros_like(self.Woh)

        self.Wih_grad_geb = torch.zeros_like(self.Wih)
        self.Whh_grad_geb = torch.zeros_like(self.Whh)
        self.Woh_grad_geb = torch.zeros_like(self.Woh)

        self.Wih_grad_all = {}
        self.Whh_grad_all = {}
        self.Woh_grad_all = {}
        
        self.Wih_grad_geb_all = {}
        self.Whh_grad_geb_all = {}
        self.Woh_grad_geb_all = {}
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters.

        The parameters will be initialized from the distribution U(-s, s).
        """
        s = 0.05  # larger value may make the gradients explode/vanish
        torch.nn.init.uniform_(self.Wih, -s, s)
        torch.nn.init.uniform_(self.Whh, 0, s)
        torch.nn.init.uniform_(self.Woh, 0, s)
        
    def zero_grad(self):
        """Set the gradients to zero."""
        self.Wih_grad.zero_()
        self.Whh_grad.zero_()
        self.Woh_grad.zero_()

        self.Wih_grad_geb.zero_()
        self.Whh_grad_geb.zero_()
        self.Woh_grad_geb.zero_()

    def forward(self, x, hp, kk):
        """Perform the forward computation.
        
        Args:
            x (Tensor): Input at the current timestep.
            hp (Tensor): Hidden state at the previous timestep.
            
        Returns:
            Tensor: Output at the current timestep.
            Tensor: Hidden state at the current timestep.
        """
        _, _, h, _, y = self._get_internals(x, hp, kk)
        return y, h

    def backward(self, y_grad, rn_grad, x, hp, kk):
        """Perform the backward computation.
        
        Args:
            y_grad (Tensor): Gradient on output at the current timestep.
            rn_grad (Tensor): Gradient on vector r at the next timestep.
            x (Tensor): Input at the current timestep that was passed to `forward`.
            hp (Tensor): Hidden state at the previous timestep that was passed to `forward`.
            
        Returns:
            Tensor: Gradient on vector r at the current timestep.
        """
        n_classes = y_grad.shape[0]
        
        # Get internal vectors r, h, and s from forward computation
        rint, r, h, s, _ = self._get_internals(x, hp, kk)

        ## BPTT calculations
        s_grad = y_grad * torch.sigmoid(s) * (1-torch.sigmoid(s)) ## note manual differentiation!!
        h_grad = self.Woh.t().matmul(s_grad) + self.Whh.t().matmul(rn_grad)
        r_grad = h_grad * ((self.tvec_hh[kk]*r)>0)*1 ## note manual differentiation!!
        rint_grad = r_grad*((self.tvec_ih[kk]*rint)>0)*1 ## note manual differentiation
        
        # Parameter gradients are accumulated
        self.Wih_grad += rint_grad.view(-1, 1).matmul(x.view(1, -1))
        self.Whh_grad += r_grad.view(-1, 1).matmul(hp.view(1, -1))
        self.Woh_grad += s_grad.view(-1, 1).matmul(h.view(1, -1))

        self.Wih_grad_all[kk] = r_grad.view(-1, 1).matmul(x.view(1, -1))
        self.Whh_grad_all[kk] = r_grad.view(-1, 1).matmul(hp.view(1, -1))
        self.Woh_grad_all[kk] = s_grad.view(-1, 1).matmul(h.view(1, -1))
        
        ## R-GEB calculation
        
        ## Indicator of post synaptic activity
        ## Pre-synaptic activation
        ## Global error vector

        x_y_grad = y_grad*x
        rint_post = ((self.tvec_ih[kk]*rint)>0)*1.
        
        hp_y_grad = y_grad*hp
        r_post = ((self.tvec_hh[kk]*r)>0)*1.
        
        self.Wih_grad_geb += torch.outer(rint_post,x_y_grad) #done!
        self.Whh_grad_geb += torch.outer(r_post,hp_y_grad) #done! 
        self.Woh_grad_geb += s_grad.view(-1,1).matmul(h.view(1, -1)) #done!
        
        self.Wih_grad_geb_all[kk] = torch.outer(rint_post,x_y_grad)
        self.Whh_grad_geb_all[kk] = torch.outer(r_post,hp_y_grad)
        self.Woh_grad_geb_all[kk] = s_grad.view(-1,1).matmul(h.view(1, -1))

        return r_grad
    
    def _get_internals(self, x, hp, kk):
        # Actual forward computations
        rint = self.Wih.matmul(x)
        r = ((self.tvec_ih[kk]*rint)>0)*rint + self.Whh.matmul(hp)
        h = ((self.tvec_hh[kk]*r)>0)*r
        s = self.Woh.matmul(h)
        y = torch.sigmoid(s)
        
        return rint, r, h, s, y

class RNN:
    def __init__(self, cell):
        self.cell = cell
    
    def forward(self, xs, h0):
        """Perform the forward computation for all timesteps.
        
        Args:
            xs (Tensor): 2-D tensor of inputs for each timestep. The
                first dimension corresponds to the number of timesteps.
            h0 (Tensor): Initial hidden state.
            
        Returns:
            Tensor: 2-D tensor of outputs for each timestep. The first
                dimension corresponds to the number of timesteps.
            Tensor: 2-D tensor of hidden states for each timestep plus
                `h0`. The first dimension corresponds to the number of
                timesteps.
        """

        ys, hs = [], [h0]
        for ii, x in enumerate(xs):
            y, h = self.cell.forward(x, hs[-1],ii)
            ys.append(y)
            hs.append(h)
        return torch.stack(ys), torch.stack(hs)
    
    def backward(self, ys_grad, xs, hs):
        """Perform the backward computation for all timesteps.
        
        Args:
            ys_grad (Tensor): 2-D tensor of the gradients on outputs
                for each timestep. The first dimension corresponds to
                the number of timesteps.
            xs (Tensor): 2-D tensor of inputs for each timestep that
                was passed to `forward`.
            hs (Tensor): 2-D tensor of hidden states that is returned
                by `forward`.
        """
        # For the last timestep, the gradient on r is zero
        rn_grad = torch.zeros(self.cell.hid_dim)
        
        n_classes = xs.shape[0]

        for ii, (y_grad, x, hp) in enumerate(reversed(list(zip(ys_grad, xs, hs)))):
            rn_grad  = self.cell.backward(y_grad, rn_grad, x, hp, n_classes-ii-1)