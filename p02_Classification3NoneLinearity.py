import matplotlib.pyplot as plt
import helper_functions
from sklearn.datasets import make_circles
import torch
from torch import nn
from sklearn.model_selection import train_test_split

device="cuda" if torch.cuda.is_available() else "cpu"

n_samples=1000
X,y=make_circles(n_samples=n_samples,noise=0.03,
                 random_state=42)

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu);
plt.show()

X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.float)

X_train,X_test,y_train,y_test=train_test_split(X,y
                                               ,test_size=0.2,
                                               random_state=42)
 
print(X_train[:5],y_train[:5])

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10,out_features=10)
        self.layer3=nn.Linear(in_features=10,out_features=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model_3=CircleModelV2().to(device)
print(model_3)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)

loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=model_3.parameters(),lr=0.1)
epoch=1000

for epochs in range(epoch):
    model_3.train()
    y_logits=model_3(X_train).squeeze()
    y_pred=torch.round(torch.logit(y_logits))
    loss=loss_fn(y_logits,y_train)
    acc=accuracy_fn(y_true=y_train,y_pred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_3.eval()
    with torch.inference_mode():
        test_logits=model_3(X_test).squeeze()
        test_pred=torch.round(torch.sigmoid(test_logits))
        test_loss=loss_fn(test_logits,y_test)
        test_acc=accuracy_fn(y_true=y_test,y_pred=test_pred)
        if epochs%100==0:
            print(f"Epoch: {epochs} | Loss: {loss:.4f} |Acc: {acc:.2f}% "
                  f"| Test Loss: {test_loss:.4f} | Test Acc: |{test_acc:.2f}%")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
helper_functions.plot_decision_boundary(model_3,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
helper_functions.plot_decision_boundary(model_3,X_test,y_test)
plt.show()