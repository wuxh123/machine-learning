import torch
import numpy as np
import operator
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_dataset=dsets.MNIST(root='m1/pymnist',train=True,transform=None,download=True)
test_dataset=dsets.MNIST(root='m1/pymnist',train=False,transform=None,download=True)

def init_network():
    network={}
    weight_scale=1e-3
    network['w1']=np.random.randn(784,50)*weight_scale
    network['b1']=np.ones(50)
    network['w2'] = np.random.randn(50, 100)*weight_scale
    network['b2'] = np.ones(100)
    network['w3'] = np.random.randn(100, 10)*weight_scale
    network['b3'] = np.ones(10)
    return network

def _relu(x):
    return np.maximum(0,x)

def forward(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1=x.dot(w1) +b1
    z1=_relu(a1)
    a2 = z1.dot(w2) + b2
    z2 = _relu(a2)
    a3 = z2.dot(w3) + b3
    y=a3
    return y

def mean_squared_error(p,y):
    return np.sum((p-y)**2)/y.shape[0]

def cross_entropy_error(p,y):
    delta=1e-7
    batch_size=p.shape[0]
    return (-np.sum(y*np.log(p+delta)))/batch_size


def _softmax(x):
    if x.ndim==2:
        c=np.max(x,axis=1)
        x=x.T-c
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    c=np.max(x)
    exp_x=np.exp(x-c)
    return exp_x/np.sum(exp_x)

'''
network=init_network()
accuracy_cnt=0
x=test_dataset.data.numpy().reshape(-1,28*28)
labels=test_dataset.targets.numpy()
for i in range(len(x)):
    y=forward(network,x[i])
    p=np.argmax(y)
    if p==labels[i]:
        accuracy_cnt+=1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)*100)+"%")
'''
network=init_network()
accuracy_cnt=0
batch_size=100
x=test_dataset.data.numpy().reshape(-1,28*28)
labels=test_dataset.targets.numpy()
findalllabels=labels.reshape(labels.shape[0],1)
print(findalllabels.shape)
bestloss=float('inf')
for i in range(0,int(len(x)),batch_size):
    network=init_network()
    x_batch=x[i:i+batch_size]
    y_batch=forward(network,x_batch)
    one_hot_labels=torch.zeros(batch_size,10)
    array=findalllabels[i:i+batch_size]
    one_hot_labels=one_hot_labels.scatter_(1,torch.LongTensor(array),1)
    loss = cross_entropy_error(one_hot_labels.numpy(),y_batch)
    if loss < bestloss:
        bestloss=loss
        bestw1,bestw2,bestw3=network['w1'],network['w2'],network['w3']
print("best loss is %f"%(bestloss))
