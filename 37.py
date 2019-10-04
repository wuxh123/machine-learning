import torch
import numpy as np
import operator 
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

batch_size=100

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

train_dataset=dsets.MNIST(root='m1/pymnist',train=True,transform=None,download=True)
test_dataset=dsets.MNIST(root='m1/pymnist',train=False,transform=None,download=True)

#加载数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

print("train_data:",train_dataset.data.size())
print("train_labels:",train_dataset.targets.size())
print("test_dataset:",test_dataset.data.size())
print("test_labels:",test_dataset.targets.size())

'''digit=train_loader.dataset.train_data[0]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
print(train_loader.dataset.targets[0])'''

if __name__ == '__main__':
    x_train = train_loader.dataset.data.numpy()
    x_train = x_train.reshape(x_train.shape[0],28*28)
    print(x_train.shape[0])
    y_train=train_loader.dataset.targets.numpy()
    X_test=test_loader.dataset.data[:100].numpy()
    X_test=X_test.reshape(X_test.shape[0],28*28)
    y_test=test_loader.dataset.targets[:100].numpy()
    num_test=y_test.shape[0]
    y_test_pred=KNN_classify(10,"E",x_train,y_train,X_test)
    num_correct=np.sum(y_test_pred==y_test)
    accuracy=float(num_correct)/num_test
    print('got %d/%d correct=>accuracy:%f' %(num_correct,num_test,accuracy))