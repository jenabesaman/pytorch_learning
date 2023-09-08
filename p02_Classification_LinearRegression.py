import matplotlib.pyplot as plt
import torch
from helper_functions import plot_predictions
from torch import nn

# from p02_Classification2MoreLayer import *
device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01
x_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * x_regression + bias
print(y_regression[:5], x_regression[:5])

split_data = int(0.8 * len(x_regression))
x_regression_train, y_regression_train = x_regression[:split_data], y_regression[:split_data]
x_regression_test, y_regression_test = x_regression[split_data:], y_regression[split_data:]

print(len(x_regression_train), len(x_regression_test), len(y_regression_train)
      , len(y_regression_test))

plot_predictions(train_data=x_regression_train, train_labels=y_regression_train,
                 test_data=x_regression_test, test_labels=y_regression_test)
plt.show()

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

print(model_2.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.01)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
x_regression_train, y_regression_train = x_regression_train.to(device), y_regression_train.to(device)
x_regression_test, y_regression_test = x_regression_test.to(device), y_regression_test.to(device)

epoch = 1000
for epochs in range(epoch):
    model_2.train()
    y_pred = model_2(x_regression_train)
    loss = loss_fn(y_pred, y_regression_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(x_regression_test)
        test_loss = loss_fn(test_pred, y_regression_test)
        if epochs % 100 == 0:
            print(f"epoch: {epochs} | loss: {loss:.5f} | test loss: {test_loss:.5f}")

with torch.inference_mode():
    y_pred=model_2(x_regression_test)

plot_predictions(train_data=x_regression_train.cpu(),train_labels=y_regression_train.cpu(),
                 test_data=x_regression_test.cpu(),test_labels=y_regression_test.cpu(),
                 predictions=y_pred.cpu())
plt.show()