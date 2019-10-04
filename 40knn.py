import torch
import numpy as np
import operator 
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Knn:
    batch_size=100

    def fit(self,x_train,y_train):
        self.Xtr=x_train
        self.Ytr=y_train

    def predict(self,k,dis,x_test):
        assert dis=='E' or dis=='M'
        num_test = self.Ytr.shape[0]
        labellist=[]
        if (dis=='E'):
            for i in range(num_test): 
                distances=np.sqrt(np.sum(((self.Xtr-np.tile(x_test[i],(self.Xtr.shape[0],1)))**2),axis=1))
                print(i)
                nearest_k=np.argsort(distances)
                topK=nearest_k[:k]
                classCount={}
                #print(x_train[i])
                for i in topK:
                #    print(x_train[i])
                    classCount[self.Xtr[i]]=classCount.get(self.Xtr[i],0)+1
                #print(classCount)
                sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
                #print(sortedClassCount)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)
        elif (dis=='M'):
            for i in range(num_test): 
                distances=np.sqrt(np.sum((np.fabs(self.Xtr-np.tile(x_test[i],(self.Xtr.shape[0],1)))),axis=1))
                print(distances)
                nearest_k=np.argsort(distances)
                topK=nearest_k[:k]
                classCount={}
                print(self.Xtr[i])
                for i in topK:
                    print(self.Xtr[i])
                    classCount[self.Xtr[i]]=classCount.get(self.Xtr[i],0)+1
                print(classCount)
                sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
                print(sortedClassCount)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)