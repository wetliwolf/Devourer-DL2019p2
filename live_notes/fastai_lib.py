# ========================================================================
# Notebook 0
# ========================================================================

TEST = 'test'

# ========================================================================
# Notebook 1
# ========================================================================

import operator
def test(a,b,cmp,cname=None):
    if cname is None:
        cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b):
    test(a,b,operator.eq,'==')

from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor
from torch import optim

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

def near(a,b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)

def test_near(a,b):
    test(a,b,near)
    
# ========================================================================
# Notebook 2
# ========================================================================

from torch.nn import init
from torch import nn

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s):
    return (x-m)/s

def test_near_zero(a,tol=1e-3): 
    assert a.abs()<tol, f"Near zero: {a}"


def mse(output, targ):
    return (output.squeeze(-1) - targ).pow(2).mean()

def stats(x): 
    """ Returns mean and std """
    return x.mean(), x.std()

import torch.nn.functional as F
import numpy as np

def f1(x, leak_amt=0):
    return F.leaky_relu(l1(x), leak_amt) 


# for a leaky relu
# kaiming init method
def gain(leaky_amt):
    return math.sqrt(2.0 / (1 + leaky_amt **2))


class Flatten(nn.Module):
    def forward(self, x):
        """ unrolls input tensor to a single shape"""
        return x.view(-1)

# ========================================================================
# Notebook 3
# ========================================================================

    
def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]
    

#export
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

#export
def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))

# ========================================================================
# Notebook 4
# ========================================================================

class DataBunch():
    