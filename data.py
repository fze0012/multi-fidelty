# 如何产生训练数据的代码

from matplotlib import pyplot as plt
import numpy as np


def func(x, y):
    func = 3 * ((1 - x) ** 2) * np.exp(-(x ** 2) - ((y + 1) ** 2)) - \
           10 * ((x / 5) - (x ** 3) - (y ** 5)) * np.exp(-(x ** 2) - ((y) ** 2)) - \
           (np.exp(-((x + 1) ** 2) - (y ** 2)) / 3)
    return func


def data_process():
    # 高精度点
    x_hi = np.linspace(-3, 3, 5)
    y_hi = np.linspace(-3, 3, 5)
    # 低精度点
    x_lo = np.linspace(-3, 3, 9)
    y_lo = np.linspace(-3, 3, 9)

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    x, y = np.meshgrid(x, y)
    xy = [i for i in zip(x.flat, y.flat)]
    xy = np.array(xy)

    x_hi, y_hi = np.meshgrid(x_hi, y_hi)
    xy_hi = [i for i in zip(x_hi.flat, y_hi.flat)]
    xy_hi = np.array(xy_hi)
    # print(xy_hi.shape)
    # print(xy_hi)

    x_lo, y_lo = np.meshgrid(x_lo, y_lo)
    xy_lo = [i for i in zip(x_lo.flat, y_lo.flat)]
    xy_lo = np.array(xy_lo)

    func_hi = func(x, y)

    func_hi_star = func(xy_hi[:, 0], xy_hi[:, 1])
    func_hi_star = func_hi_star[:, np.newaxis]

    # func_lo=0.5*func(0.57*x,1.35*y)
    # func_lo_star = 0.5 * func(0.57 * xy_lo[:,0], 1.35 * xy_lo[:,1])
    #
    func_lo = 0.95 * func(0.97 * x, 1.05 * y)
    func_lo_star = 0.95 * func(0.97 * xy_lo[:, 0], 1.05 * xy_lo[:, 1])
    # 增加维度
    func_lo_star = func_lo_star[:, np.newaxis]
    # func_lo_star_prime=
    # 定义figure
    figure = plt.figure()
    # 将figure变为3d
    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    #  ax1.plot_surface(x, y, func_hi, rstride=1, cstride=1, label='$func_H$',cmap='rainbow')

    ax1.plot_surface(x, y, func_hi, rstride=1, cstride=1, cmap='rainbow')
    ax1.invert_xaxis()
    ax1.scatter(x_hi, y_hi, func_hi_star, color='r', linewidth=2, marker='x', label='high-fidelity training data')
    ax1.legend()
    plt.grid(True)

    ax2 = figure.add_subplot(1, 2, 2, projection='3d')
    # ax2.plot_surface(x, y, func_lo, rstride=1, cstride=1, label='$func_L$',cmap='rainbow')
    ax2.plot_surface(x, y, func_lo, rstride=1, cstride=1, cmap='rainbow')
    ax2.invert_xaxis()
    ax2.scatter(x_lo, y_lo, func_lo_star, color='b', edgecolors='blue', marker='o', label='low-fidelity training data')
    ax2.legend()
    plt.grid(True)
    plt.show()
    return xy, xy_hi, xy_lo, func_hi_star, func_lo_star, func_hi, func_lo, x, y, x_hi, y_hi, x_lo, y_lo
