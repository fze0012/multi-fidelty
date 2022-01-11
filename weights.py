from torch import nn

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)

        nn.init.constant_(m.bias, 0.0)