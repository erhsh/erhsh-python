import mindspore
import numpy as np
from mindspore import context, ms_function
from mindspore import nn, Tensor
from mindspore.ops import GradOperation, operations as P

context.set_context(device_target="CPU")
context.set_context(mode=context.GRAPH_MODE)


class OpNetWrapper(nn.Cell):
    def __init__(self, op, *args, **kwargs):
        super(OpNetWrapper, self).__init__()
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def construct(self, *inputs):
        return self.op(*inputs, *self.args, **self.kwargs)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


########################################################################################################################
# Test Code
########################################################################################################################

def test_00_base():
    op = P.Abs()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([-1, 0, 1]), mindspore.float32)
    outputs = op_wrapper(input_x)

    print("input is:", input_x)
    print("output is:", outputs)
    assert np.allclose(outputs.asnumpy(), [1., 0., 1.])


def test_01_grad():
    op = P.Abs()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([-3, -2, -1, -0, 0, 1, 2, 3]), mindspore.float32)
    outputs = op_wrapper(input_x)

    sens = Tensor(np.arange(int(np.prod(outputs.shape))).reshape(outputs.shape), mindspore.float32)
    backward_net = Grad(op_wrapper)
    input_grad = backward_net(input_x, sens)

    print("input is:", input_x)
    print("outputs is:", outputs)
    print("sens is:", sens.asnumpy())
    print("input_grad is:", input_grad[0].asnumpy())
    assert np.allclose(input_grad[0].asnumpy(), [-0., -1., -2., 0., 0., 5., 6., 7.])


if __name__ == '__main__':
    test_00_base()
    test_01_grad()
