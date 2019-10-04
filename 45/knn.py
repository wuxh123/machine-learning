import torch
import numpy as np
import operator 
from torch.utils.data import dataloader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Knn:
    def __init__(self):
        pass

    def fit(self,x_train,y_train):
        self.Xtr=x_train
        self.Ytr=y_train

    def predict(self,k,dis,x_test):
        assert dis=='E' or dis=='M'
        num_test = self.Ytr.shape[0]
        labellist=[]
        #欧拉公式
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
                    classCount[self.Ytr[i]]=classCount.get(self.Ytr[i],0)+1
                #print(classCount)
                sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
                #print(sortedClassCount)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)
        #曼哈顿公式
        elif (dis=='M'):
            print("M")
            for i in range(num_test): 
                distances=np.sqrt(np.sum((np.fabs(self.Xtr-np.tile(x_test[i],(self.Xtr.shape[0],1)))),axis=1))
                print(i,distances)
                nearest_k=np.argsort(distances)
                topK=nearest_k[:k]
                classCount={}
               
                #print(self.Xtr[i])
                for i in topK:
                    print("------------------")
                    print(self.Xtr[i])
                #    print(self.Xtr[i])
                    print("------------------")
                    classCount[self.Ytr[i]]=classCount.get(self.Ytr[i],0)+1
                #print(classCount)
                sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
                #print(sortedClassCount)
                labellist.append(sortedClassCount[0][0])
            return np.array(labellist)