import numpy as np
import matplotlib.pyplot as plt

def J(theta):
    return (theta-2.5)**2 -1

def dJ(theta):
    return 2*(theta-2.5)

theta = 0.0
theta_history=[theta]
eta=0.1
epsilon = 1e-8
plot_x=np.linspace(-1,6,141)
plot_y=(plot_x-2.5)**2 -1 
while True:
    gradient=dJ(theta)
    last_theta=theta
    theta = theta-eta*gradient
    theta_history.append(theta)
    if (abs(J(theta)-J(last_theta)) < epsilon):
        break
plt.plot(plot_x,J(plot_x),color='r')
plt.plot(np.array(theta_history),J(np.array(theta_history)),color='b',marker='x')
plt.show()
print(theta_history)