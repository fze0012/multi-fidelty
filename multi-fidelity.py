from data import *
from model.NN1_nn import *
from model.NN2_nn import *
from weights import *
import mertic
from xx_high import *
from torch import optim
from trainer import train
from matplotlib import pyplot as plt


def main():
    xy, xy_hi, xy_lo, func_hi_star, func_lo_star, func_hi, func_lo, x, y, x_hi, y_hi, x_lo, y_lo = data_process()
    model_h = NN1(2, 20, 4, 1)
    model_h.apply(weights_init)
    optimizer1 = optim.Adam(model_h.parameters(), lr=1e-3)
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1,
                                                [60, 90, 120, 150, 180],
                                                gamma=0.25, last_epoch=-1)
    # [3000, 4500, 6000, 7500, 9000]
    for i in range(200):
        train(model_h, xy_lo, func_lo_star, optimizer1, i)
        scheduler1.step()

    alpha = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
    alpha.data.uniform_(0, 1)

    # NN1有激活函数
    model3 = NN2(3, 10, 1, 1)
    # NN2没有激活函数
    model4 = NN1(3, 10, 2, 1)
    model3.apply(weights_init)
    model4.apply(weights_init)
    optimizer2 = optim.AdamW([{'params': model3.parameters(), 'weight_decay': 0.01},
                              {'params': model_h.parameters()},
                              {'params': model4.parameters(), 'weight_decay': 0.01},
                              {'params': alpha}], lr=1e-3)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2,
                                                [300, 450, 600, 750, 900],
                                                gamma=0.25, last_epoch=-1)
    # [3000, 4500, 6000, 7500, 9000]
    xy_lo_r = torch.from_numpy(xy_lo).float()
    xy_lo_r.requires_grad_()

    for i in range(1000):
        # 低精度数据经过NN1
        pred_h = model_h(xy_lo_r)
        loss3 = torch.mean(torch.square(pred_h - torch.from_numpy(func_lo_star).float()))
        pred_2 = pred_do(alpha, model3, model4, model_h, xy_hi)
        loss2 = torch.mean(torch.square(pred_2 - torch.from_numpy(func_hi_star).float())) + loss3
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        scheduler2.step()
        if i % 50 == 0:
            print('epoch:', i, 'Loss:', loss2.item())
        xx_high = xx_high_cal(pred_do(alpha, model3, model4, model_h, xy), x)

        mertic.mertic(func_hi, xx_high)

    xx_high = xx_high_cal(pred_do(alpha, model3, model4, model_h, xy), x)

    figure5 = plt.figure()
    ax5 = figure5.add_subplot(1, 1, 1, projection='3d')
    surf = ax5.plot_surface(x, y, xx_high, rstride=1, cstride=1, label='DNN though multi-fidelity model',
                            color='red')
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    ax5.invert_xaxis()
    surf1 = ax5.plot_surface(x, y, func_hi, rstride=1, cstride=1, label='Exact', color='blue')
    surf1._facecolors2d = surf1._facecolor3d
    surf1._edgecolors2d = surf1._edgecolor3d

    ax5.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
