import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def numberical_gradient(f,x):
    h=1e-6
    grad=np.zeros_like