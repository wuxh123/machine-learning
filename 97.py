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

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=1000):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

net =simpleNet()
print(net.W)
X=np.array([[1,2]])
# p=net.predict(X)
# print(p)
# print(np.argmax(p))
y=np.array([0,0,1])
#print(net.loss(X,y))
f = lambda w: net.loss(X,y)
dw = gradient_descent(f, net.W)  # 主要需要更新的是W
print(dw)

print('损失值:',net.cross_entropy_error(net._softmax(np.dot(X,dw)),y))
print(np.dot(X,dw))
print('预测值:',np.argmax(np.dot(X,dw)))