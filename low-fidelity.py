from data import *
from model.NN1_nn import *
from torch import nn, optim
from matplotlib import pyplot as plt
from weights import *
from trainer import *
import os
import mertic

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    xy, xy_hi, xy_lo, func_hi_star, func_lo_star, func_hi, func_lo, x, y, x_hi, y_hi, x_lo, y_lo = data_process()
    ###
    model_h = NN1(2, 20, 4, 1)
    model_h.apply(weights_init)
    optimizer = optim.Adam(model_h.parameters(), lr=1e-3)
    loss_value = 1

    epoch = 0

    while loss_value > 1e-4:
        loss_value = train(model_h, xy_lo, func_lo_star, optimizer, epoch)
        epoch = epoch + 1

    nn_pred_h = model_h(torch.from_numpy(xy).float())

    figure4 = plt.figure()
    # 将figure变为3d
    ax4 = figure4.add_subplot(1, 1, 1, projection='3d')

    nn_pred_h = nn_pred_h.detach().numpy()
    nn_pred_h.shape = x.shape

    surf = ax4.plot_surface(x, y, nn_pred_h, rstride=1, cstride=1, label='DNN though LF', color='red')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax4.invert_xaxis()
    surf1 = ax4.plot_surface(x, y, func_lo, rstride=1, cstride=1, label='Exact', color='blue')
    surf1._facecolors2d = surf1._facecolor3d
    surf1._edgecolors2d = surf1._edgecolor3d
    ax4.scatter(x_lo, y_lo, func_lo_star, color='black', linewidth=2, marker='x')
    ax4.legend()
    plt.grid(True)

    plt.show()



if __name__ == '__main__':
    main()
