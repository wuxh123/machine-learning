import torch
from torch import nn
import numpy as np


# 编码one_hot
def one_hot(y):
    '''
    y: (N)的一维tensor，值为每个样本的类别
    out:
        y_onehot: 转换为one_hot 编码格式
    '''
    y = y.view(-1, 1)
    y_onehot = torch.FloatTensor(3, 5)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot


def cross_entropy_one_hot(target):
    # 解码
    _, labels = target.max(dim=1)
    return labels
    # 如果需要调用cross_entropy，则还需要传入一个input_
    # return F.cross_entropy(input_, labels)


x = np.array([1, 2, 3])
x_tensor = torch.from_numpy(x)
print(one_hot(x_tensor))
x2 = np.array([[0, 1, 0, 0, 0]])
x2_tensor = torch.from_numpy(x2)
print(cross_entropy_one_hot(x2_tensor))