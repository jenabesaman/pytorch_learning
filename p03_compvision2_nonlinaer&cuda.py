import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from helper_functions import accuracy_fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X,y=X.to(device),y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

            loss /= len(data_loader)
            acc /= len(data_loader)
            return {"model_name": model.__class__.__name__,
                    "model_loss": loss.item(),  # single value
                    "model_acc": acc}

def print_train_time(start: float, end: float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device} :{total_time} seconds")
    return total_time


train_data = datasets.FashionMNIST(root="data", train=True,
                                   download=False,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False,
                                  download=False,
                                  transform=ToTensor(),
                                  target_transform=None)

image, label = (train_data[0])
classes_names = train_data.classes

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE
                              , shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                             shuffle=False)




class FashionMnistModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


model_1_fashion = FashionMnistModelV1(input_shape=28 * 28, hidden_units=10,
                              output_shape=len(classes_names)).to(device)
print(next(model_1_fashion.parameters()))
from helper_functions import accuracy_fn

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1_fashion.parameters(), lr=0.1)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1)  # Go from logits -> pred labels
                                    )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1_fashion.parameters(), lr=0.1)

train_time_start_on_gpu = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(model=model_1_fashion, data_loader=train_dataloader,
               loss_fn=loss_fn, optimizer=optimizer,
               accuracy_fn=accuracy_fn, device=device)
    test_step(model=model_1_fashion, data_loader=test_dataloader,
              loss_fn=loss_fn, accuracy_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = timer()
total_train_time_model_1_fashion = print_train_time(train_time_start_on_gpu,
                                             train_time_end_on_gpu,
                                             device=device)
print(total_train_time_model_1_fashion)
print(next(model_1_fashion.parameters()))

torch.save(obj=model_1_fashion.state_dict(),f="models/p03_compvis2.pth")

model_1_fashion_results=eval_model(model=model_1_fashion,data_loader=test_dataloader,
                                   loss_fn=loss_fn,accuracy_fn=accuracy_fn,
                                   device=device)

print(model_1_fashion_results)


