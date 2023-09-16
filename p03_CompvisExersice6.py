import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor





train_dataloader=datasets.MNIST(
    root="data",train=True,download=True,
    transform=ToTensor,target_transform=None
)
test_dataloader=datasets.MNIST(
    root="data",train=False,download=True,
    transform=ToTensor,target_transform=None
)