import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn import datasets
from sklearn.model_selection import train_test_split
from helper_functions import plot_predictions, plot_decision_boundary

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)

print(len(X), len(y))
print(f"first 5 sample of X: {X[:5]} \n &"
      f" first sample of y: {y[:5]}")
print(f"first 5 value of x in 0 index{X[:5, 0]}")
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
print(f"first 10 value of circle: {circles.head(10)}")
print(f"Count value of data for each label: {circles.label.value_counts()}")

plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu);

plt.show()

print(f"shape of x: {X.shape} and shape of y: {y.shape}")

x_sample = X[0]
y_sample = y[0]
print(f"Value for one sample of x: {x_sample} & the same for y: {y_sample}")
print(f"Shapes for one sample of x: {x_sample.shape} & the same for y: {y_sample.shape}")

print(f"type x: {type(X)} & dtype x: {X.dtype}")
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(f"new type of x: {type(X)} & the same for y: {type(y)}"
      f" dtype of x: {X.dtype}")

torch.manual_seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"len x&y train & test: {len(X_train), len(y_test), len(y_train), len(y_test)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"current device is: {device}")


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))


model_0 = CircleModelV0().to(device)
print(f"model state dict: {model_0.state_dict()}")
print(f"model device: {next(model_0.parameters()).device}")

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)



print(f"model 0: {model_0} & model 0 state dict: {model_0.state_dict()}")

with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"len of untrained pred: {len(untrained_preds)} & shape of untrained pred: {untrained_preds.shape}")
print(f"len of samples: {len(X_test)} & shape of samples: {X_test.shape}")
print(f"\n First 10 predictions: \n{torch.round(untrained_preds[:10])}")
print(f"\n First 10 labels :\n {y_test[:10]}")

# logit ->softmax

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(f"view the first 5 output of the forward pass on the test set: \n"
      f" {y_logits}")

# use sigmoid act fun on model logits to turn into pred prob

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    # if loss_fn=BCELoss->loss=loss_fn(torch.sigmoid(y_logits),y_train)
    accuracy = accuracy_fn(y_true=y_train, y_pred=y_pred)
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy_fn(y_true=y_test, y_pred=test_pred)
    if epoch % 10 == 0:
        print(f"Epoch:{epoch} | Loss: {loss:.5f} | Acc: {accuracy:.2f}% |"
              f" Test loss: {test_loss:.5f} | test accuracy: {test_accuracy:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()
