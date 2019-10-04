import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

class ReLu:
    def __init__(self):
        self.x=None

    def forward(self,x):
        self.x=np.maximum()
        out=self.x
        return out

    def backward(self,dout):
        dx=dout
        dx[self.x<=0] = 0
        return dx

class _sigmoid:
    def __init__(self):
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self,dout):
        dx=dout*self.out*(1-self.out)
        return dx