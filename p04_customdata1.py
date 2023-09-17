import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
from pathlib import Path
import zipfile
import os
import random
from PIL import Image
from typing import Tuple, Dict, List
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer

print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
# how create dataset in site

data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi"
if image_path.is_dir():
    print(f"image path directory already exist.skipping Downloading....")
else:
    print(f"{image_path} does not exist.creating one ....")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Dwonloading pizza steak sushi data ...")
        f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("unzipping pizza steak and sushi data ...")
    zip_ref.extractall(image_path)


def walk_through_dir(dir_path):
    for dirpath, dirnames, filename in os.walk(dir_path):
        print(f"there are {len(dirnames)} directories and {len(filename)} images in '{dirpath}' .")


walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"
print(train_dir, test_dir)

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_pth = random.choice(image_path_list)
image_class = random_image_pth.parent.stem
img = Image.open(random_image_pth)
print(f"Random image path{random_image_pth}")
print(f"Image class{image_class}")
print(f"Image height{img.height}")
print(f"Image width{img.width}")
# img.show()

img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);
plt.show()

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),  # (224,224) is common too
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()  # FIX CHW & compatible with torch model
])
print(f"shape of transformed data{data_transform(img).shape},dtype:{data_transform(img).dtype}")


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)

plt.show()

train_data = datasets.ImageFolder(root=train_dir, transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)
print(train_data, test_data)

class_names = train_data.classes
print(class_names)
class_dict = train_data.class_to_idx  # class map to int
print(class_dict)
print(f"len train data: {len(train_data)} len test data:{len(test_data)}")
print(train_data.targets)  # all targets /labels
print(f"first sample: {train_data.samples[0]}")

# Index on image to find label and samples
img, label = train_data[0][0], train_data[0][1]

print(f"image tensor: {img}")
print(f"image shape: {img.shape}")
print(f"image datatype: {img.dtype}")
print(f"Label: {label}")
print(f"Lable datatype{type(label)}")
print(f"class name of lable: {class_names[label]}")

# plot img with plt
img_permute = img.permute(1, 2, 0)
print(f"Original shape: {img.shape}")  # chw
print(f"Image permuted: {img_permute.shape}")  # hwc
plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis(False)
plt.title(class_names[label], fontsize=14)
plt.show()

# Turning our Dataset's into DataLoader's makes them iterable so a model can go through learn the relationships between samples and targets (features and labels).

print(f"number of cpus: {os.cpu_count()}")
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                             shuffle=False)
print(f"train and test dataloader: {train_dataloader, test_dataloader}")
print(f"len dataloaders: {len(train_dataloader), len(test_dataloader)}")
print(f"len train and test data without batch size: {len(train_data), len(test_data)}")

img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

# loading with custom dataset
print(train_data.classes, train_data.class_to_idx)

target_directory = train_data
print(f"Target directory: {target_directory}")
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"Class name found: {class_names_found}")


# turn to func
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")
    classes_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, classes_to_idx


print(find_classes(directory=train_dir))


# custom dataset class
# class ImageFolderCustom(Dataset):
#     def __init__(self, targer_dir: str, transform: None):
#         self.paths = pathlib.Path(targer_dir).glob("*/*.jpg")
#         self.trasform = transform
#         self.classes, self.class_to_idx = find_classes(targer_dir)
#
#     def load_image(self, index: int) -> Image.Image:
#         image_path = self.paths[index]
#         return Image.open(image_path)
#
#     def __len__(self) -> int:
#         return len(self.paths)
#
#     def __getitem__(self, index:int) -> Tuple[torch.Tensor,int]:
#         img=self.load_image(index)
#         class_name=self.paths[index].parent.name
#         class_idx=self.class_to_idx[class_name]
#         if self.trasform:
#             return self.trasform(img),class_idx
#         else:
#             return img,class_idx

# Write a custom dataset class (inherits from torch.utils.data.Dataset)


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:

        # 3. Create class attributes
        # Get all image paths
        self.paths = list(
            pathlib.Path(targ_dir).glob("*/*.jpg"))  # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

        # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)


trian_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=trian_transform)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transform)

print(f"train and test data custom{train_data_custom, test_data_custom}")
print(f"len train data & custom custom: {len(train_data), len(train_data_custom)}")
print(f"len test data & custom custom: {len(test_data), len(test_data_custom)}")
print(f"train data custom classes & idx:{train_data_custom.classes, train_data_custom.class_to_idx}")

print(f"check equality between original and custom Image folder dataset: "
      f"{train_data_custom.classes == train_data.classes},{test_data_custom.classes == test_data.classes}")


# function display
def display_random_images(dataset=torch.utils.data.dataset,
                          classes: List[str] = None, n: int = 10,
                          display_shape: bool = True, seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print("For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    if seed:
        random.seed = seed

    random_sample_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_sample_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1, 2, 0)  # for matplotlib : chw->hwc
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis(False)
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()


display_random_images(dataset=train_data, classes=class_names, n=5, seed=None)
display_random_images(dataset=train_data_custom, classes=class_names, n=20, seed=42)

# turn to dataloaders
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_dataloader_custom = DataLoader(dataset=train_data_custom, batch_size=BATCH_SIZE,
                                     num_workers=0, shuffle=True)
test_dataloader_custom = DataLoader(dataset=test_data_custom, batch_size=BATCH_SIZE, shuffle=False)
print(f"train and test dataloader custom: {train_dataloader_custom, test_dataloader_custom}")

img_custom, label_custom = next(iter(train_dataloader_custom))
print(f"shape of custom img and label dataloader: {img_custom.shape, label_custom.shape}")

# augmantion
train_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                       transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                       transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                      transforms.ToTensor()])

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
# plot_transformed_images(
#     image_paths=image_path_list,
#     transform=train_transforms,          #####################################eror#################
#     n=3
# )

# create transforms and load data for model0
simple_transform = transforms.Compose([transforms.Resize(size=(64, 64)),
                                       transforms.ToTensor()])
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

train_dataloader_simple = DataLoader(dataset=train_data_simple, batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=0)
test_dataloader_simple = DataLoader(dataset=test_data_simple, batch_size=BATCH_SIZE,
                                    shuffle=False)


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,  # input shape for linear=hiden_unit * h*w of last conv layer
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x)))


# gpu brrrr

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

print(model_0)

image_batch, label_batch = next(iter(train_dataloader_simple))
image_batch = image_batch.to(device)
print(image_batch.shape)
# print(model_0(image_batch))

print(summary(model_0, input_size=([1, 3, 64, 64])))


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device=device):
    y_pred = model.train()
    train_loss, train_acc = 0
    for batch, X, y in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc


def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader,
              loss_fn=nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, X, y in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model_0(y)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5, device=device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader,
                                           loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        print(f"epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results


torch.manual_seed(42)
torch.cuda.manual_seed(42)
NUM_EPOCHS = 5
model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

start_timer = timer()
model_0_results = train(model=model_0, train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple, optimizer=optimizer,
                        loss_fn=loss_fn, epochs=NUM_EPOCHS)
end_time = timer()
print(f"Total training time: {end_time - start_timer:.3f} secends")
print(model_0_results)
