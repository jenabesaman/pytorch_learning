import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
import p03_defenitions
import tabulate
from IPython.display import display
from p03_defenitions import train_data, test_data, train_dataloader, test_dataloader, class_names, train_step, \
    test_step, device, BATCH_SIZE, timer, print_train_time, eval_model
import torch
import torch.nn as nn
import pandas as pd
from helper_functions import accuracy_fn
import random
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

BATCH_SIZE = 32

loss_fn = nn.CrossEntropyLoss()


def eval_model1(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

            loss /= len(data_loader)
            acc /= len(data_loader)
            return {"model_name": model.__class__.__name__,
                    "model_loss": loss.item(),  # single value
                    "model_acc": acc}


test_data1 = datasets.FashionMNIST(root="data", train=False,
                                   download=False,
                                   transform=ToTensor(),
                                   target_transform=None)
test_dataloader1 = DataLoader(dataset=test_data1, batch_size=BATCH_SIZE,
                              shuffle=False)


class FashionMnistModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


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


class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,  # trick: * last layer output shape
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"output shape of classifier: {x.shape}")
        return x


loaded_model1 = FashionMnistModelV0(
    input_shape=28 * 28,
    hidden_units=10,
    output_shape=len(class_names)
)
loaded_model1.load_state_dict(torch.load(f="models/p03_compvis1.pth"))

loaded_model2 = FashionMnistModelV1(input_shape=28 * 28, hidden_units=10,
                                    output_shape=len(class_names)).to(device)
loaded_model2.load_state_dict(torch.load(f="models/p03_compvis2.pth"))

loaded_model3 = FashionMNISTModelV2(input_shape=1,
                                    hidden_units=10,
                                    output_shape=len(class_names)).to(device)
loaded_model3.load_state_dict(torch.load(f="models/p03_compvisionpull4.pth"))
loaded_model3 = loaded_model3.to(device)

loaded_model1_results = eval_model(model=loaded_model1, data_loader=test_dataloader1,
                                   loss_fn=loss_fn, accuracy_fn=accuracy_fn)
loaded_model2_results = eval_model(model=loaded_model2, data_loader=test_dataloader,
                                   loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
loaded_model3_results = eval_model(model=loaded_model3, data_loader=test_dataloader,
                                   loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

compare_results = pd.DataFrame([loaded_model1_results, loaded_model2_results, loaded_model3_results])
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.show()

img, label = test_data[0][:10]
print(img.shape, label)

# random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
print(test_samples[0].shape)

plt.imshow(X=test_samples[0].squeeze(), cmap='gray')
plt.title(class_names[test_labels[0]])
plt.show()

pred_probs = p03_defenitions.make_prediction(model=loaded_model3,
                                             data=test_samples)
print(pred_probs[:2])
pred_classes = torch.argmax(pred_probs, dim=1)
print(f"predictions: {pred_classes} test classes:{test_labels}")

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i + 1)
    plt.imshow(sample.squeeze(), cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f"Pred:{pred_label} | Truth:{truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')
    plt.axis(False);
plt.show()

y_preds = []
loaded_model3.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        X, y = X.to(device), y.to(device)
        y_logits = loaded_model3(X)
        y_pred = torch.softmax(y_logits.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

# print(y_preds)
y_pred_tensor = torch.cat(y_preds)  # turn our list of prediction into a tensor
print(f"one prediction sample per class {y_pred_tensor, len(y_pred_tensor)}")

try:
    import torchmetrics, mlxtend

    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19 or higher"
except:
    # !pip install torchmetrics -u mlxtend
    import torchmetrics, mlxtend

    print(f"mlxtend version: {mlxtend.__version__}")

# confusion matrix
print(f"class_names:{class_names} y_pred_tensor: {y_pred_tensor}\n"
      f" test_data target: {test_data.targets}")

# Make a confusion matrix using torchmetrics.ConfusionMatrix.
# Plot the confusion matrix using mlxtend.plotting.plot_confusion_matrix()
confmat = ConfusionMatrix(num_classes=len(class_names),task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)
print(confmat_tensor)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # matplotlib likes work with numpy,
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()