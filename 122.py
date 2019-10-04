import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

batch_size = 100
# MNIST dataset
train_dataset = dsets.MNIST(root = '/pymnist', #选择数据的根目录
                           train = True, # 选择训练集
                           transform = transforms.ToTensor(), #转换成tensor变量
                           download = True) # 从网络上download图片
test_dataset = dsets.MNIST(root = '/pymnist', #选择数据的根目录
                           train = False, # 选择测试集
                           transform = transforms.ToTensor(), #转换成tensor变量
                           download = True) # 从网络上download图片
#加载数据
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, #使用批次数据
                                           shuffle = True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

print(train_loader.batch_size)

input_size = 784 #mnist的像素为28*28
hidden_size = 500
num_classes = 10 #输出为10个类别分别对应0-9

# 创建神经网络模型
class Neural_net(nn.Module):
#初始化函数，接受自定义输入特征的维数，隐含层特征维数以及输出层特征维数。
    def __init__(self, input_num,hidden_size, out_put):
        super(Neural_net, self).__init__()
        self.layer1 = nn.Linear(input_num, hidden_size)#从输入到隐藏层的线性处理
        self.layer2 = nn.Linear(hidden_size, out_put)#从隐层到输出层的线性处理

    def forward(self, x):
        out = self.layer1(x) #输入层到隐藏层的线性计算
        out = torch.relu(out) #隐藏层激活
        out = self.layer2(out) #输出层，注意，输出层直接接loss
        return out

net = Neural_net(input_size, hidden_size, num_classes)
print(net)

learning_rate = 1e-1  # 学习率
num_epoches = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 使用随机梯度下降
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_loader):  # 利用enumerate取出一个可迭代对象的内容
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        outputs = net(images)  # 将数据集传入网络做前向计算
        loss = criterion(outputs, labels)  # 计算loss
        optimizer.zero_grad()  # 在做反向传播之前先清除下网络状态
        loss.backward()  # loss反向传播
        optimizer.step()  # 更新参数

        if i % 100 == 0:
            print('current loss = %.5f' % loss.item())

print('finished training')

total=0
correct=0

for images,labels in test_loader:
    images=Variable(images.view(-1,28*28))
    outputs=net(images)
    _,predicts=torch.max(outputs.data,1)
    total += labels.size(0)
    correct+=(predicts==labels).sum()

print('accuracy=%.2f' % (100*correct/total))
