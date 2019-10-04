import numpy as np
import matplotlib.pyplot as plt
import operator 

def KNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis=='E' or dis=='M'
    num_test = Y_test.shape[0]
    labellist=[]
    if (dis=='E'):
        for i in range(num_test):     
            '''            print('--------------------')       
            print(Y_test[i])
            print((X_train.shape[0],1))
            print(np.tile(Y_test[i],(X_train.shape[0],1)))
            print(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2))
            print('--------------------')  '''     
            distances=np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))
            print(distances)
            nearest_k=np.argsort(distances)
            topK=nearest_k[:k]
            classCount={}
            print('+++++++')   
            print(topK)  
            print(x_train[i])
            for i in topK:
                print(x_train[i])
                classCount[x_train[i]]=classCount.get(x_train[i],0)+1
            print(classCount)
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            print(sortedClassCount)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
    elif (dis=='M'):
        for i in range(num_test): 
            distances=np.sqrt(np.sum((np.fabs(X_train-np.tile(Y_test[i],(X_train.shape[0],1)))),axis=1))
            print(distances)
            nearest_k=np.argsort(distances)
            topK=nearest_k[:k]
            classCount={}
            print(x_train[i])
            for i in topK:
                print(x_train[i])
                classCount[x_train[i]]=classCount.get(x_train[i],0)+1
            print(classCount)
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            print(sortedClassCount)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

def CreateDataSet():
    group=np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels=np.array(['a','a','b','b','a','b'])
    return group,labels

if __name__ == '__main__':
    group,labels=CreateDataSet()
    y_test_pred = KNN_classify(1,'M',group,labels,np.array([[1.0,2.1],[0.4,2.0]]))
    print(y_test_pred)

