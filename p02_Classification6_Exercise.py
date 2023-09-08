import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torchmetrics
import helper_functions

device = "cuda" if torch.cuda.is_available() else "cpu"

Num_samples = 1000

X_moon, y_moon = make_moons(n_samples=Num_samples, noise=0.07
                            , random_state=42)

print(f"x shape: {X_moon.shape} and y shape: {y_moon.shape}")

print(X_moon.dtype)

X_moon = torch.tensor(X_moon, dtype=torch.float)
y_moon = torch.tensor(y_moon, dtype=torch.float)

# X_moon = torch.from_numpy(X_moon).type(torch.float32)
# y_moon = torch.from_numpy(y_moon).type(torch.LongTensor)

print(X_moon[:5], y_moon[:5])

print(f"x shape: {X_moon.shape} and y shape: {y_moon.shape}")

plt.figure(figsize=(12, 6))
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon, cmap=plt.cm.RdYlBu);
plt.show()

X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(X_moon, y_moon
                                                                        , test_size=0.2,
                                                                        random_state=42)


X_moon_train = X_moon_train.to(device)
X_moon_test = X_moon_test.to(device)
y_moon_train = y_moon_train.to(device)
y_moon_test = y_moon_test.to(device)


class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=14):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_5 = MoonModel(input_features=2, output_features=1).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_5.parameters(), lr=0.1)
epochs = 1001
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)

for epoch in range(epochs):
    model_5.train()
    y_logits = model_5(X_moon_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_pred_probs)

    loss = loss_fn(y_logits, y_moon_train)
    acc = accuracy_metric(y_moon_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_5.eval()
    with torch.inference_mode():
        y_logits_test = model_5(X_moon_test).squeeze()
        y_pred_probs_test=torch.sigmoid(y_logits_test)
        # print(f"y pred prob test shape {y_pred_probs_test.shape} | 5 first {y_pred_probs_test[:5]}")
        y_pred_test = torch.round(y_pred_probs_test)
        # print(y_pred_test[:5],y_pred_test.shape)
        # y_pred_test2=torch.softmax((y_logits_test).unsqueeze(1),dim=1).argmax(dim=1)
        # print(y_pred_test2.shape,y_pred_test2[:5])
        # y_pred_test=y_pred_test.squeeze(1)
        loss_test = loss_fn(y_logits_test, y_moon_test)
        acc_test = accuracy_metric(y_moon_test, y_pred_test)

    if epoch % 100 == 0:
        print(f"epoch: {epoch} | loss: {loss:.5f} | acc:{acc:.2f}"
              f" test loss: {loss_test:.5f} | acc test: {acc_test:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
helper_functions.plot_decision_boundary(model_5, X_moon_train, y_moon_train)
plt.subplot(1, 2, 2)
plt.title("Test")
helper_functions.plot_decision_boundary(model_5, X_moon_test, y_moon_test)
plt.show()
