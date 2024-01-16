from __future__ import print_function
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid
import argparse
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

def C( x ):
    return (2. * torch.arctanh(1. - 2.*x))/(1. - 2.*x) 
def log_C( x ):
    # Numerically stable implementation from https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE/blob/master/notebooks/Continuous_Bernoulli.ipynb
    if abs(x - 0.5) < 1e-3:
        # Taylor Approximation around 0.5
        value = torch.log(torch.tensor(2.))
        taylor = torch.tensor(1); nu = 1. - 2. * x; e = 1; k = nu**2
        for i in range(1, 10): 
            e *= k; taylor += e / (2. * i + 1) 
        return value + torch.log( taylor )
    return torch.log( C(x) )

def sumlogC( x , eps = 1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation
        
    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    ''' 
    x = torch.clamp(x, eps, 1.-eps) 
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    return far_values.sum() + close_values.sum()


og = torch.tensor([0.0, 1e-7, 0.2, 0.5, 0.5, 0.5 + 1e+9, 0.8, 1.0 - 1e-9, 1.0])
fg = torch.distributions.ContinuousBernoulli(1).log_prob(og)
print(f"OG: {og}")
print(f"FG: {fg}")

