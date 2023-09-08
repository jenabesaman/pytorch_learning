import numpy as np
import matplotlib.pyplot as plt
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

import torch
from torch import nn
from helper_functions import plot_decision_boundary
from sklearn.model_selection import train_test_split
import torchmetrics

print(f"x shape {X.shape} x len {len(X)} y shape {y.shape}")

X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.LongTensor)


device="cuda" if torch.cuda.is_available() else "cpu"

Random_SEED=42
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=Random_SEED)


X_train=X_train.to(device)
X_test=X_test.to(device)
y_train=y_train.to(device)
y_test=y_test.to(device)


in_feature=2
out_feature=3
hidden_unit=10
class SpiralModelV5(nn.Module):
    def __init__(self,in_feature,out_feature,hidden_unit):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(in_features=in_feature,out_features=hidden_unit),
            nn.ReLU(),
            nn.Linear(in_features=hidden_unit,out_features=hidden_unit),
            nn.ReLU(),
            nn.Linear(in_features=hidden_unit,out_features=out_feature)
        )
    def forward(self,x):
        return self.layers(x)

model_7=SpiralModelV5(in_feature=in_feature,
         out_feature=out_feature,hidden_unit=hidden_unit).to(device)

print(f"model stattedict: {model_7.state_dict()},mdoel: {model_7}")

epochs=1001
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model_7.parameters(),lr=0.2)
acc_fn=torchmetrics.Accuracy(task="multiclass",num_classes=3).to(device)

for epoch in range(epochs):
    model_7.train()
    y_logits=model_7(X_train)
    y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss=loss_fn(y_logits,y_train)
    acc=acc_fn(y_train,y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_7.eval()
    with torch.inference_mode():
        test_logits=model_7(X_test)
        test_pred=torch.softmax(test_logits,dim=1).argmax(dim=1)
        test_loss=loss_fn(test_logits,y_test)
        test_acc=acc_fn(y_test,test_pred)
        if epoch%100==0:
            print(f"epoch: {epoch} | loss: {loss:.5f} | acc: {acc:.2f}% | "
                  f"test loss: {test_loss:.5f} | test acc: {test_acc:.2f}%")


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_7,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_7,X_test,y_test)
plt.show()