import torch
import numpy as np
import operator 
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from knn import Knn

def getXmean(x_train):
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    mean_image=np.mean(x_train,axis=0)
    return mean_image

def centralized(x_test,mean_image):
    x_test = np.reshape(x_test,(x_test.shape[0],-1))
    x_test = x_test.astype(np.float)
    x_test -= mean_image
    return x_test

if __name__ == "__main__":
    batch_size = 100

    train_dataset = dsets.CIFAR10(root='m1/pycifar',train=True,download=True)
    test_dataset = dsets.CIFAR10(root='m1/pycifar',train=False,download=True)

    #加载数据
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

    x_train = train_loader.dataset.data
    x_train=x_train.reshape(x_train.shape[0],-1)
    mean_image = getXmean(x_train)
    x_train=centralized(x_train,mean_image)
    y_train=train_loader.dataset.targets
    y_train=np.array(y_train)
    x_test=test_loader.dataset.data
    #print(x_test)
    x_test=x_test.reshape(x_test.shape[0],-1)
    #print(x_test)
    x_test=centralized(x_test,mean_image)
    y_test=test_loader.dataset.targets
    y_test=np.array(y_test)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    #讲训练集分成五部分，每部分轮流作为验证集
    num_folds=5
    k_choices=[1,3,5,8,10,12,15,20]
    num_training=x_train.shape[0]
    x_train_folds=[]
    y_train_folds=[]
    indices=np.array_split(np.arange(num_training),indices_or_sections=num_folds)
    for i in indices:
        x_train_folds.append(x_train[i])
        y_train_folds.append(y_train[i])
    k_to_accuracies={}
    for k in k_choices:
        acc=[]
        for i in range(num_folds):
            x=x_train_folds[0:i]+x_train_folds[i+1:]
            x=np.concatenate(x,axis=0)

            y=y_train_folds[0:i]+y_train_folds[i+1:]
            y=np.concatenate(y,axis=0)

            test_x = np.array(x_train_folds[i])
            test_y = np.array(y_train_folds[i])

            classifier = Knn()
            print(x)
            classifier.fit(np.array(x),np.array(y))

            y_pred = classifier.predict(k,'M',test_x)
            accuracy=np.mean(y_pred==test_y)
            acc.append(accuracy)
    k_to_accuracies[k] = acc
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k=%d,accuracy=%f' % (k,accuracy))