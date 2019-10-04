import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

tensor = torch.linspace(-6,6,200)
tensor = Variable(tensor)
np_data = tensor.numpy()

#定义激活函数
y_relu = torch.relu(tensor).data.numpy()
y_sigmoid =torch.sigmoid(tensor).data.numpy()
y_tanh = torch.tanh(tensor).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(np_data, y_relu, c='red', label='relu')
plt.legend(loc='best')

plt.subplot(222)
plt.plot(np_data, y_sigmoid, c='red', label='sigmoid')
plt.legend(loc='best')

plt.subplot(223)
plt.plot(np_data, y_tanh, c='red', label='tanh')
plt.legend(loc='best')

plt.show()