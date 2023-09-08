import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

print(torch.__version__, torchvision.__version__)

train_data = datasets.FashionMNIST(root="data", train=True,
                                   download=False,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=False,
                                  transform=ToTensor(),
                                  target_transform=None)

print(len(train_data), len(test_data))

image, label = (train_data[0])
classes_names = train_data.classes
print(classes_names)

class_to_idx = train_data.class_to_idx
print(class_to_idx)
print(train_data.targets)
print(f"image shape: {image.shape} -> [color_channels,height,width]")
print(f"Image label: {label}")

plt.imshow(image.squeeze())
plt.title(label)
plt.show()

plt.imshow(X=image.squeeze(), cmap="gray")
plt.title(classes_names[label])
plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label=classes_names[label])
    plt.axis(False);

plt.show()
from torch.utils.data import DataLoader

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE
                              , shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                             shuffle=False)

# for batch,(x,y) in enumerate(train_dataloader):
#     print(batch)

print(f"DataLoaders:{train_dataloader, test_dataloader} , \n Length of train_DataLoader"
      f" :{len(train_dataloader)} and len test_dataloader :{len(test_dataloader)} \n "
      f"with batch size :{BATCH_SIZE} so hole number of batch are {len(train_dataloader) / BATCH_SIZE}")

train_features_batch, train_label_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_label_batch.shape)

torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_label_batch[random_idx]
plt.imshow(X=img.squeeze(), cmap="gray")
plt.title(classes_names[label])
plt.axis(False);
plt.show()
print(f"image size: {img.shape} and label: {label} and label size: {label.shape}")

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)

print(f"shape before flattening: {x.shape} ->[color_chanel,height,width] \n"
      f"shape after flattening: {output.shape} ->[color_chanel,height*width] ")

from torch import nn


class FashionMnistModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


model_0 = FashionMnistModelV0(
    input_shape=28 * 28,
    hidden_units=10,
    output_shape=len(classes_names)
)

print(model_0)

dummy_x = torch.rand([1, 1, 28, 28])
print(f"model predict random x{model_0(dummy_x)} and shape: {model_0(dummy_x).shape}")

import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("hleper_function already exist,skiping download...")
else:
    print("Downloading helper_function.py")
    request = requests.get(
        url="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

from timeit import default_timer as timer


def print_train_time(start: float, end: float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device} :{total_time} seconds")
    return total_time


start = timer()
end = timer()
print_train_time(start=start, end=end)

# tqdm for progress bar
from tqdm.auto import tqdm

torch.manual_seed(42)
train_time_start_on_cpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch}\n------")
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
    train_loss /= len(train_dataloader)
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = model_0(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"\nTrain loss:{loss:.4f} | Test loss:{test_loss:.4f} | Test acc:{test_acc:.4f}")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(train_time_start_on_cpu,
                                            train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

            loss /= len(data_loader)
            acc /= len(data_loader)
            return {"model_name": model.__class__.__name__,
                    "model_loss": loss.item(),  # single value
                    "model_acc": acc}


model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
                             loss_fn=loss_fn, accuracy_fn=accuracy_fn)
print(model_0_results,total_train_time_model_0)

torch.save(obj=model_0.state_dict(),f="models/p03_compvis1.pth")