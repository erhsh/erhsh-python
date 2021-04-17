import torch


def test_abs():
    # inputs
    input_x = torch.tensor([1, -1, 0, 2, -5], dtype=torch.float32, requires_grad=True)
    print("input_x:", type(input_x), input_x.detach().numpy())

    # forward
    outputs = torch.abs(input_x)
    print("outputs:", type(outputs), outputs.detach().numpy())

    # bakward
    ones = torch.ones(size=(5,)) * 10
    outputs.backward(gradient=ones)
    grad = input_x.grad
    print("input_x.grad:", type(grad), grad.numpy())


if __name__ == '__main__':
    test_abs()
