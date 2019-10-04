import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self): #在这里定义卷积神经网络需要的元素
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #定义第一个卷积层
        self.pool = nn.MaxPool2d(2, 2) #池化层
        self.conv2 = nn.Conv2d(6, 16, 5) #定义第二个卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #全连接层
        self.fc2 = nn.Linear(120, 84) #全连接层
        self.fc3 = nn.Linear(84, 10) #最后一个全连接层用作10分类

    def forward(self, x): #使用__init__中的定义，构建卷积神经网络结构
        x = self.pool(F.relu(self.conv1(x))) #第一个卷积层首先要经过relu做激活，然后使用前面定义好的nn.MaxPool2d(2, 2)方法做池化
        x = self.pool(F.relu(self.conv2(x))) #第二个卷积层也要经过relu做激活，然后使用前面定义好的nn.MaxPool2d(2, 2)方法做池化
        x = x.view(-1, 16 * 5 * 5) #对特征层tensor维度进行变换
        x = F.relu(self.fc1(x)) #卷积神经网络的特征层经过第一次全连接层操作，然后再通过relu层激活
        x = F.relu(self.fc2(x)) #卷积神经网络的特征层经过第二次全连接层操作，然后再通过relu层激活
        x = self.fc3(x) #卷积神经网络的特征层经过最后一次全连接层操作，得到最终要分类的结果（10类标签）
        return x

net = Net()