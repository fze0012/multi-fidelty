import torch


def train(model, xy, func_star, optimizer, epoch):
    pred_h = model(torch.from_numpy(xy).float())
    loss = torch.mean(torch.square(pred_h - torch.from_numpy(func_star).float()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_value = loss.item()

    return loss_value
