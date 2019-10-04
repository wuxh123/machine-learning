import numpy as np
def _sigmoid(in_data):
    return 1/(1+np.exp(-in_data))

#输入层
x=np.array([0.9,0.1,0.8])

#隐藏层,需要计算输入层到中间隐藏层每个节点的组合。
w1=np.array([[0.9,0.3,0.4],
             [0.2,0.8,0.2],
             [0.1,0.5,0.6]])
w2=np.array([[0.3,0.7,0.5],[0.6,0.5,0.2],[0.8,0.1,0.9]])

print(w1.dot(x))
xhidden = _sigmoid(w1.dot(x))
print(xhidden)
print("----------")
xoutput=w2.dot(xhidden)
print(xoutput)