from tqdm.auto import tqdm

import p03_defenitions
from p03_defenitions import train_data,test_data,train_dataloader,test_dataloader,class_names,train_step,test_step,device,BATCH_SIZE,timer,print_train_time,eval_model
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



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
            nn.MaxPool2d(kernel_size=2) # default stride value is same as kernel_size
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

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                              hidden_units=10,
                              output_shape=len(class_names)).to(device)
print(model_2)

torch.manual_seed(42)
image = torch.randn(size=(32, 3, 64, 64))
test_image = image[0]

print(f"imnage shape:{image.shape} test image shape:{test_image.shape}\n"
      f"test image{test_image}")

conv_layer = nn.Conv2d(in_channels=3, out_channels=10,
                       kernel_size=(3, 3), stride=1, padding=0)
conv_output = conv_layer(test_image)
print(f"conv layer ouput:{conv_output}\n conv output shape{conv_output.shape}")

pooling_layer = nn.MaxPool2d(kernel_size=2)
maxpool_out = pooling_layer(test_image)
print(f"max pool shape{maxpool_out.shape}")

rand_image_tensor = torch.rand(size=(1, 28, 28))
pred=model_2(rand_image_tensor.unsqueeze(0).to(device))
print(pred)

from helper_functions import accuracy_fn
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(lr=0.1,params=model_2.parameters())

torch.manual_seed(42)
torch.cuda.manual_seed(42)
train_time_start_model_2=timer()
epochs=3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n---------")
    train_step(model=model_2,data_loader=train_dataloader,
               loss_fn=loss_fn,optimizer=optimizer,
               accuracy_fn=accuracy_fn,device=device)
    test_step(model=model_2,data_loader=test_dataloader,
              loss_fn=loss_fn,accuracy_fn=accuracy_fn,
              device=device)
train_time_end_model_2=timer()
total_train_time_model_2=print_train_time(start=train_time_start_model_2,
                                          end=train_time_end_model_2,device=device)

model_2_results=eval_model(model=model_2,data_loader=test_dataloader,
                           loss_fn=loss_fn,accuracy_fn=accuracy_fn,
                           device=device)
print(model_2_results)

torch.save(obj=model_2.state_dict(),f="models/p03_compvisionpull4.pth")
# import pandas as pd
# copmare_results=pd.DataFrame([[model_0_results, model_1_results, model_2_results]])
# print(compare_results)
# compare_results["training_time"] = [total_train_time_model_0,
#                                     total_train_time_model_1,
#                                     total_train_time_model_2]
# print(compare_results)