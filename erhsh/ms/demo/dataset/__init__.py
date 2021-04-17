from .ds_00_mocker import create_dataset as create_mocker_dataset
from .ds_02_mind import create_dataset as create_mind_dataset
from .ds_03_imagefolder import create_dataset as create_imagefolder_dataset
from .ds_04_cifar10 import create_dataset as create_cifar10_dataset
from .ds_05_mnist import create_dataset as create_mnist_dataset

__all__ = [
    "create_mocker_dataset",
    "create_mind_dataset",
    "create_imagefolder_dataset",
    "create_cifar10_dataset",
    "create_mnist_dataset",
]
