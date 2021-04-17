import numpy as np


def create_net():
    from mindspore import nn, Parameter, Tensor

    class SimpleNet(nn.Cell):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.w = Parameter(Tensor(np.array([0.1], dtype=np.float32)), name="w")
            self.relu = nn.ReLU()

        def construct(self, *inputs, **kwargs):
            x, _ = inputs
            x = x + self.w
            x = self.relu(x)
            return x

    return SimpleNet()


if __name__ == '__main__':
    net = create_net()
    print(net)
