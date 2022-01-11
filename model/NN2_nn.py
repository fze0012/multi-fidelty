from .Unit_nn import *


# 类NN2类继承nn.Module
class NN2(nn.Module):
    def __init__(self, in_N, width, depth, out_N):
        # super(NN2, self).__init__()就是对继承自父类nn.Module的属性进行初始化
        super(NN2, self).__init__()
        self.in_N = in_N
        self.width = width
        self.depth = depth
        self.out_N = out_N

        self.stack = nn.ModuleList()

        self.stack.append(nn.Linear(in_N, width))

        for i in range(depth):
            self.stack.append(nn.Linear(width, width))

        self.stack.append(nn.Linear(width, out_N))
        # print(len(self.stack))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x
