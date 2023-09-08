import torch
import torchmetrics
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
import torchmetrics

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES,
                            centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob, test_size=0.2,
                                                                        random_state=RANDOM_SEED)

plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
plt.show()

print(X_blob_train.shape, y_blob_train[:5])
print(f"unique: {torch.unique(y_blob_train)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

X_blob_train = X_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_train = y_blob_train.to(device)
y_blob_test = y_blob_test.to(device)


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
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


model_4 = BlobModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES, hidden_units=8).to(device)

print(f"model 4:{model_4}")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)
print(f"model device:{next(model_4.parameters()).device} | X_train device: "
      f"{X_blob_train.device}")

model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# usding activation for convert logits to pred prob
y_pred_probs = torch.softmax(y_logits, dim=1)
print(f"y test:{y_blob_test[:3]} \n and y logits:{y_logits[:3]} \n "
      f"y pred prob:{y_pred_probs[:3]} and sum :{sum(y_pred_probs[0])} \n "
      f"and max:{torch.max(y_pred_probs[0])} then index number:{torch.argmax(y_pred_probs[0])} \n"
      f" then convert pred prob to pred label:{torch.argmax(y_pred_probs, dim=1)[:5]} \n"
      f"but y test :{(y_blob_test)[:5]} ")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_blob_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_pred)

    if epoch % 10 == 0:
        print(
            f"epoch: {epoch} | loss: {loss:.5f} | acc: {acc:.2f}% | test loss: {test_loss:.5f} | test acc: {test_acc:.2f}%")

model_4.eval()
with torch.inference_mode():
    y_logits=model_4(X_blob_test)

y_pred_probs=torch.softmax(y_logits,dim=1)
y_pred=torch.argmax(y_pred_probs,dim=1)
print(f"logit: {(y_logits)[:5]} , pred prob: {(y_pred)[:5]} , predict: {(y_pred)[:5]}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_4,X_blob_test,y_blob_test)
plt.show()

torchmetric_accuracy=torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)
print(torchmetric_accuracy(y_pred,y_blob_test))

