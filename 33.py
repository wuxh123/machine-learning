import numpy as np
import matplotlib.pyplot as plt

def CreateDataSet():
    group=np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels=np.array(['a','a','b','b','a','b'])
    return group,labels

if __name__ == '__main__':
    group,labels=CreateDataSet()
    plt.scatter(group[labels=='a',0],group[labels=='a',1],color='r',marker='*')
    plt.scatter(group[labels=='b',0],group[labels=='b',1],color='g',marker='+')
    plt.show()