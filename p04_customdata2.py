import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
from pathlib import Path
import zipfile
import os
import random
from PIL import Image
from typing import Tuple, Dict, List
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"


train_transform_trival=torchvision.transforms.Compose([transforms.Resize(size=(64,64)),
                                   transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                       transforms.ToTensor()])
test_transform_trival=torchvision.transforms.Compose([transforms.Resize(size=(64,64)),
                                                      transforms.ToTensor()])

train_data_augmented=datasets.ImageFolder(root=train_dir,
                                          transform=train_transform_trival)
test_data_augmented=datasets.ImageFolder(root=test_dir,
                                         transform=test_transform_trival)

print(train_data_augmented)