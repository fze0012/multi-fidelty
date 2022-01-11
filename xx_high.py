import torch


def pred_do(alpha, model3, model4, model_h, xy):
    pred = alpha * model3(torch.cat((torch.from_numpy(xy).float(), model_h(torch.from_numpy(xy).float())), 1)) + \
           (1 - alpha) * model4(torch.cat((torch.from_numpy(xy).float(), model_h(torch.from_numpy(xy).float())), 1))
    return pred


def xx_high_cal(pred, x):
    xx_high = pred.detach().numpy()
    xx_high.shape = x.shape
    return xx_high
