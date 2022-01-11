from model.NN1_nn import *
from data import *
from weights import *
from trainer import *
from torch import optim
from matplotlib import pyplot as plt


def main():
    # 通过高精度数据训练NN1得到一个模型
    xy, xy_hi, xy_lo, func_hi_star, func_lo_star, func_hi, func_lo, x, y, x_hi, y_hi, x_lo, y_lo = data_process()
    model_h = NN1(2, 20, 4, 1)
    model_h.apply(weights_init)
    optimizer = optim.Adam(model_h.parameters(), lr=0.001)
    nIter = 2000
    epoch = 0
    loss_value = 1

    while loss_value > 5e-6:
        train(model_h, xy_hi, func_hi_star, optimizer, epoch)
        epoch = epoch + 1

    # 绘制通过高精度数据训练得到的结果

    nn_pred_h = model_h(torch.from_numpy(xy).float())

    figure3 = plt.figure()
    # 将figure变为3d
    ax3 = figure3.add_subplot(1, 1, 1, projection='3d')

    nn_pred_h = nn_pred_h.detach().numpy()
    nn_pred_h.shape = x.shape


    surf = ax3.plot_surface(x, y, nn_pred_h, rstride=1, cstride=1, label='DNN though HF', color='red')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax3.invert_xaxis()
    surf1 = ax3.plot_surface(x, y, func_hi, rstride=1, cstride=1, label='Exact', color='blue')
    surf1._facecolors2d = surf1._facecolor3d
    surf1._edgecolors2d = surf1._edgecolor3d
    ax3.scatter(x_hi, y_hi, func_hi_star, color='black', linewidth=2, marker='x')
    ax3.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
