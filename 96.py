import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class simpleNet:
    def cross_entropy_error(self,p, y):
        delta = 1e-7
        batch_size = p.shape[0]
        return (-np.sum(y * np.log(p + delta))) / batch_size

    def _softmax(self,x):
        if x.ndim == 2:
            c = np.max(x, axis=1)
            x = x.T - c
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        c = np.max(x)
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x)

    def __init__(self):
        np.random.seed(0)
        self.W=np.random.randn(2,3)

    def   forward(self,x):
        return np.dot(x,self.W)

    def loss(self,x,y):
        z=self.forward(x)
        p1=self._softmax(z)
        loss=self.cross_entropy_error(p1,y)
        return loss

    def predict(self,x):
        y = self.forward(x)
        return y

net =simpleNet()
print(net.W)
X=np.array([0.6,0.9])
# p=net.predict(X)
# print(p)
# print(np.argmax(p))
y=np.array([0,0,1])
print(net.loss(X,y))

