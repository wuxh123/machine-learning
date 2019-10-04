import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    plot_x=np.linspace(-1,6,141)
    plot_y=(plot_x-2.5)**2 -1 
    plt.scatter(plot_x[5],plot_y[5],color='r')
    plt.plot(plot_x,plot_y)
    plt.xlabel('heta',fontproperties='simHei',fontsize=15)
    plt.xlabel('损失函数',fontproperties='simHei',fontsize=15)
    plt.show()