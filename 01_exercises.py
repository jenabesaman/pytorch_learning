import pathlib

import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# weight = 0.3
# bias = 0.9
# X = torch.arange(start=0, end=200, step=2).unsqueeze(dim=1)
# y = weight * X + bias

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

device = "cuda" if torch.cuda.is_available() else "cpu"

split_data = int(0.8 * len(X))
X_train = (X[:split_data]).to(device)
y_train = y[:split_data].to(device)
X_test = X[split_data:].to(device)
y_test = y[split_data:].to(device)

print(X_test.device)


def plot_prediction(train_data=X_train.cpu(),
                    train_label=y_train.cpu(),
                    test_data=X_test.cpu(),
                    test_label=y_test.cpu(),
                    prediction=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, s=4, c="g", label="train data")
    plt.scatter(test_data, test_label, s=4, c="r", label="test_data")
    if prediction is not None:
        plt.scatter(test_data, prediction, s=4, c="b", label="prediction")
    plt.title("plot data with labels")
    plt.legend(prop={"size": 14})
    plt.show()


plot_prediction()


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1.to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
        if epoch % 10 == 0:
            print(f"epoch: {epoch} | loss: {loss} |test loss: {test_loss}")
model_1.eval()

plot_prediction(prediction=test_pred.to("cpu"))

MODEL_PATH=Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME= "01_exerciese_model.pth"
MODEL_SAVE_PATH=MODEL_PATH / MODEL_NAME
print(f"Saveing model to {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),f=MODEL_SAVE_PATH)

loaded_model=LinearRegressionModelV2()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model.state_dict())
loaded_model.to(device)
with torch.inference_mode():
    loaded_pred=loaded_model(X_test)

y_test=model_1(X_test)
print(loaded_pred.cpu()==y_test.cpu())

print(loaded_model.state_dict())

