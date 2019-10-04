########################################
# 第1步：载入数据
########################################
import torch
import torchvision
import torchvision.transforms as transforms
import os

# 使用torchvision可以很方便地下载cifar10数据集，而torchvision下载的数据集为[0, 1]的PILImage格式，我们需要将张量Tensor归一化到[-1, 1]

transform = transforms.Compose(
    [transforms.ToTensor(),  # 将PILImage转换为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 将[0, 1]归一化到[-1, 1]
)

trainset = torchvision.datasets.CIFAR10(root='./book/classifier_cifar10/data',
                                        # root表示cifar10的数据存放目录，使用torchvision可直接下载cifar10数据集，也可直接在https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz这里下载（链接来自cifar10官网）
                                        train=True,
                                        download=True,
                                        transform=transform  # 按照上面定义的transform格式转换下载的数据
                                        )
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,  # 每个batch载入的图片数量，默认为1
                                          shuffle=True,
                                          num_workers=0  # 载入训练数据所需的子任务数
                                          )

testset = torchvision.datasets.CIFAR10(root='./book/classifier_cifar10/data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################
# 第2步：构建卷积神经网络
########################################
import math
import torch
import torch.nn as nn

cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


class VGG(nn.Module):
    def __init__(self, net_name):
        super(VGG, self).__init__()

        # 构建网络的卷积层和池化层，最终输出命名features，原因是通常认为经过这些操作的输出为包含图像空间信息的特征层
        self.features = self._make_layers(cfg[net_name])

        # 构建卷积层之后的全连接层以及分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # fc1
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),  # fc2
            nn.ReLU(True),
            nn.Linear(512, 10),  # fc3，最终cifar10的输出是10类
        )
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


net = VGG('VGG16')

########################################
# 第3步：定义损失函数和优化方法
########################################
import torch.optim as optim

# x = torch.randn(2,3,32,32)
# y = net(x)
# print(y.size())
criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 定义优化方法：随机梯度下降

########################################
# 第4步：卷积神经网络的训练
########################################
for epoch in range(5):  # 训练数据集的迭代次数，这里cifar10数据集将迭代2次
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        # 初始化
        inputs, labels = data  # 获取数据
        optimizer.zero_grad()  # 先将梯度置为0

        # 优化过程
        outputs = net(inputs)  # 将数据输入到网络，得到第一轮网络前向传播的预测结果outputs
        loss = criterion(outputs, labels)  # 预测结果outputs和labels通过之前定义的交叉熵计算损失
        loss.backward()  # 误差反向传播
        optimizer.step()  # 随机梯度下降方法（之前定义）优化权重

        # 查看网络训练状态
        train_loss += loss.item()
        if batch_idx % 2000 == 1999:  # 每迭代2000个batch打印看一次当前网络收敛情况
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, train_loss / 2000))
            train_loss = 0.0

    print('Saving epoch %d model ...' % (epoch + 1))
    state = {
        'net': net.state_dict(),
        'epoch': epoch + 1,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/cifar10_epoch_%d.ckpt' % (epoch + 1))

print('Finished Training')

########################################
# 第5步：批量计算整个测试集预测效果
########################################
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 当标记的label种类和预测的种类一致时认为正确，并计数

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 结果打印：Accuracy of the network on the 10000 test images: 73 %

########################################
# 分别查看每个类的预测效果
########################################
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        cifar10_classes[i], 100 * class_correct[i] / class_total[i]))