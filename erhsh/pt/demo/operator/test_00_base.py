import torch


def test_simple_grad():
    # inputs
    input_x = torch.tensor([1, -1, 0, 2, -5], dtype=torch.float32, requires_grad=True)
    print("input_x:", type(input_x), input_x.detach().numpy())

    # backward
    ones = torch.ones(size=(5,))
    input_x.backward(gradient=ones)
    grad = input_x.grad
    print("input_x.grad:", type(grad), grad.numpy())


if __name__ == '__main__':
    test_simple_grad()
