import torch
import numpy as np
import operator 
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size = 100

def KNN_classify(k,dis,X_train,y_train,Y_test):
    assert dis=='E' or dis=='M'
    num_test = Y_test.shape[0]
    labellist=[]
    if (dis=='E'):
        for i in range(num_test): 
            distances=np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))
            print(i)
            nearest_k=np.argsort(distances)
            topK=nearest_k[:k]
            classCount={}
            #print(x_train[i])
            for i in topK:
            #    print(x_train[i])
                classCount[y_train[i]]=classCount.get(y_train[i],0)+1
            #print(classCount)
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            #print(sortedClassCount)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
    elif (dis=='M'):
        for i in range(num_test): 
            distances=np.sqrt(np.sum((np.fabs(X_train-np.tile(Y_test[i],(X_train.shape[0],1)))),axis=1))
            print(i)
            nearest_k=np.argsort(distances)
            topK=nearest_k[:k]
            classCount={}
            #print(x_train[i])
            for i in topK:
            #    print(x_train[i])
                classCount[y_train[i]]=classCount.get(y_train[i],0)+1
            #print(classCount)
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            #print(sortedClassCount)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

def getXmean(x_train):
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    mean_image=np.mean(x_train,axis=0)
    return mean_image

def centralized(x_test,mean_image):
    x_test = np.reshape(x_test,(x_test.shape[0],-1))
    x_test = x_test.astype(np.float)
    x_test -= mean_image
    return x_test

train_dataset = dsets.CIFAR10(root='m1/pycifar',train=True,download=True)
test_dataset = dsets.CIFAR10(root='m1/pycifar',train=False,download=True)

#加载数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
digit=train_loader.dataset.data[0]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
print(classes[train_loader.dataset.targets[0]])
x_train=train_loader.dataset.data
mean_image=getXmean(x_train)
x_train=centralized(x_train,mean_image)
y_train=train_loader.dataset.targets
x_test=test_loader.dataset.data[:100]
x_test=centralized(x_test,mean_image)
y_test=test_loader.dataset.targets[:100]
num_test=len(y_test)
y_test_pred=KNN_classify(6,'E',x_train,y_train,x_test)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('got %d/%d correct=>accuracy:%f' %(num_correct,num_test,accuracy))