import pandas as pd
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

# import p04_customdata1
import p04_defs

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_transform_trival = torchvision.transforms.Compose([transforms.Resize(size=(64, 64)),
                                                         transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                         transforms.ToTensor()])
test_transform_trival = torchvision.transforms.Compose([transforms.Resize(size=(64, 64)),
                                                        transforms.ToTensor()])

train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trival)
test_data_augmented = datasets.ImageFolder(root=test_dir,
                                           transform=test_transform_trival)

print(train_data_augmented)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_dataloader_augmented = DataLoader(dataset=train_data_augmented, batch_size=BATCH_SIZE,
                                        shuffle=True)
test_dataloader_simple = DataLoader(dataset=test_data_augmented, batch_size=BATCH_SIZE,
                                    shuffle=False)

torch.manual_seed(42)
model_1 = p04_defs.TinyVGG(input_shape=3, hidden_units=10,
                           output_shape=len(train_data_augmented.classes)).to(device)
print(model_1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 5
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
start_time = timer()

model_1_results = p04_defs.train(model=model_1, train_dataloader=train_dataloader_augmented,
                                 test_dataloader=test_dataloader_simple, optimizer=optimizer,
                                 loss_fn=loss_fn, epochs=NUM_EPOCHS)
end_time = timer()
print(model_1_results)
print(f"total training time for model1: {end_time - start_time:.3f} secends")

p04_defs.plot_loss_curves(model_1_results)

# ways to evaluate (see learn)
# model_0_df = p04_customdata1.model_0_results
model_1_df = model_1_results
#

# # set a plot
# plt.figure(figsize=(15, 10))
# epochs = range(len(model_0_df))
# plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

data_path = Path("/data")
custom_image_path=data_path/'04-pizza-dad.jpeg'
if not custom_image_path.is_file():
    with open(custom_image_path,'wb') as f:
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path} ...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exist, skipping download ...")

#read custom image
#type float32, shape, device
custom_image_uint8=torchvision.io.read_image(str(custom_image_path))
print(f"custom image tensor: {custom_image_uint8}")
print(f"custom image shape: {custom_image_uint8.shape}")
print(f"custom image data type: {custom_image_uint8.dtype}")
plt.imshow(custom_image_uint8.permute(1,2,0))
plt.show()

#change dtype
custom_image=torchvision.io.read_image(str(custom_image_path)).type(torch.float32)/255
plt.imshow(custom_image.permute(1,2,0))
plt.show()
#change size
custom_image_transform=transforms.Compose([transforms.Resize(size=(64,64))])
custom_image_transformed=custom_image_transform(custom_image)

print(f"Original shape: {custom_image.shape}")
print(f"Transformed shape: {custom_image_transformed.shape}")
plt.imshow(custom_image_transformed.permute(1,2,0))
plt.show()

#add batch dimention and device
custom_image_transformed.unsqueeze(dim=0)
model_1.eval()
with torch.inference_mode():
    custom_image_pred=model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
print(custom_image_pred)

custom_image_pred_prob=torch.softmax(custom_image_pred,dim=1)
custom_image_pred_label=torch.argmax(custom_image_pred_prob,dim=1).cpu()
class_names = train_data_augmented.classes
custom_image_class=class_names[custom_image_pred_label]
print(custom_image_class)

# def pred_and_plot_image(model:torch.nn.Module,image_path:str,
#                         class_names:List[str]=None,transform=None,
#                         device=device):
#     target_image=torchvision.io.read_image(str(image_path)).type(torch.float32)
#     target_image=target_image/255
#     if transform:
#         transform=transform(target_image)
#     model.to(device)
#     model.eval()
#     with torch.inference_mode():
#         target_image=target_image.unsqueeze(0)
#         target_image_pred=model(target_image.to(device))
#     target_image_pred_prob=torch.softmax(target_image_pred,dim=1)
#     target_image_label=torch.argmax(target_image_pred_prob,dim=1)
#     plt.imshow(target_image.squeeze().permute(1,2,0))
#     if class_names:
#         title=f"Pred: {class_names[target_image_label.cpu()]} | Prob: {target_image_pred_prob.max().cpu():.3f}"
#     else:
#         title=f"Pred: {target_image_label} | Prob: {target_image_pred_prob:.3f}"
#     plt.title(title)
#     plt.axis(False)
#     plt.show()

p04_defs.pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
