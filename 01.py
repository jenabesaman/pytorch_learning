import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# print(torch.__version__)


weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(X[:10], y[:10])
print(len(X), len(y))

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(X_test), len(y_train), len(y_test))


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# plt.scatter(x=X,y=y)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(
            1, requires_grad=True,
            dtype=torch.float
        ))
        self.bias = nn.Parameter(torch.rand(
            1, requires_grad=True,
            dtype=torch.float
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weights * x) + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
print(f"weight={weight} and bias= {bias}")

# predicting
with torch.inference_mode():
    y_preds = model_0(X_test)

print(y_test)
plot_predictions(predictions=y_preds)
plt.show()
print(model_0.state_dict())

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

torch.manual_seed(42)
epochs = 200
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    y_preds = model_0(X_train)
    loss = loss_fn(y_preds, y_train)
    # prin  t(f"Loss : {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(model_0.state_dict())
        print(f"Epoch:{epoch} | Loss:{loss} | Test loss:{test_loss}")

with torch.inference_mode():
    y_preds_new = model_0(X_test)

plot_predictions(predictions=y_preds_new)
plt.show()

print(f"epoch count:{epoch_count} loss value:{loss_values} "
      f"test loss value:{test_loss_values}")

# np.array(torch.tensor(loss_values).cpu().numpy())
plt.plot(epoch_count,np.array(torch.tensor(loss_values).numpy()),label="Train loss")
plt.plot(epoch_count,test_loss_values,label="Test loss")

plt.title("Training and Test loss curves")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend();
plt.show()

print(model_0.state_dict())

MODEL_PATH=Path("model")
MODEL_PATH.mkdir(parents=True,exist_ok=True)
MODEL_NAME="01_pytorch_model.pth"
MODEL_SAVE_PATH=MODEL_PATH / MODEL_NAME

print(f"Saving model to:{MODEL_SAVE_PATH}")

torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

print("lodinig saved model")
loaded_model_0=LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loaded_model_0.state_dict())

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds=loaded_model_0(X_test)

with torch.inference_mode():
    y_preds=model_0(X_test)

print(f"loaded_model_preds:{loaded_model_preds}")
print(f"compare loaded model preds with original model preds:"
      f" {y_preds==loaded_model_preds}")
