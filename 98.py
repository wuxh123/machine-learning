import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']
        a1=np.dot(x,W1)+b1
        z1=_relu(a1)
        a2=np.dot(z1,W2)+b2
        p=_softmax(a2)
        return p

    def loss(self,x,y):
        p1=self.predict(x)
        loss=cross_entropy_error(p1,y)
        return loss

    # x:输入数据, y:监督数据
    def numerical_gradient(self, x, y):
        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def accuracy(self, x, y):
        p = self.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)

        accuracy = np.sum(y == p) / float(x.shape[0])
        return accuracy

def _relu(x):
    return np.maximum(0,x)

def _softmax(x):
    if x.ndim == 2:
        c = np.max(x, axis=1)
        x = x.T - c
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

def cross_entropy_error(p, y):
    delta = 1e-7
    batch_size = p.shape[0]
    return (-np.sum(y * np.log(p + delta))) / batch_size

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

batch_size=100

train_dataset=dsets.MNIST(root='m1/pymnist',train=True,transform=None,download=True)
test_dataset=dsets.MNIST(root='m1/pymnist',train=False,transform=None,download=True)

#加载数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

x_train = train_dataset.data.numpy().reshape(-1,28*28)
y_train_tmp=train_dataset.targets.reshape(train_dataset.targets.shape[0],1)
y_train=torch.zeros(y_train_tmp.shape[0],10).scatter_(1,y_train_tmp,1).numpy()

x_test = test_dataset.data.numpy().reshape(-1,28*28)
y_test_tmp=test_dataset.targets.reshape(test_dataset.targets.shape[0],1)
y_test=torch.zeros(y_test_tmp.shape[0],10).scatter_(1,y_test_tmp,1).numpy()

# 超参数
iters_num = 1000  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
epoch = 5

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(epoch):
    for i in range(iters_num):
        print(i)
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.numerical_gradient(x_batch, y_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 记录学习过程
        loss = network.loss(x_batch, y_batch)
        if i % 100 == 0:
            print(loss)

print(network.accuracy(x_test,y_test))
