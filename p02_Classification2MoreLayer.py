import matplotlib.pyplot as plt
import torch
from torch import nn
from p02_classification import *


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


model_1 = CircleModelV1().to(device)
print(model_1.state_dict())

torch.manual_seed(42)
torch.cuda.manual_seed(42)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
epoch = 1000
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)
for epochs in range(epoch):
    model_1.train()
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(input=y_logits))  # logits->pred prob->pred labels
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_loss = loss_fn(test_logits, y_test)

        test_pred = torch.round(torch.sigmoid(test_logits))
        test_accuracy = accuracy_fn(y_test, test_pred)

    if epochs % 100 == 0:
        print(f"epoch: {epochs} | loss: {loss:.5f} | accuracy: {acc:.2f}%"
              f" | test loss: {test_loss:.5f} | test accuracy: {test_accuracy:.2f}% ")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()
