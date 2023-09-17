import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader

# Step 2: Load and preprocess the dataset
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('C:/Workarea/File_Analyser/dataset', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Define and train your ResNet model as a one-class classifier
class OneClassResNet(nn.Module):
    def __init__(self, num_classes):
        super(OneClassResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model and set up the loss function
model = OneClassResNet(num_classes=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for images, _ in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.ones_like(outputs))  # Target label is 1 for the target class
        loss.backward()
        optimizer.step()
        print(epoch,loss)

# Save the trained model
torch.save(model.state_dict(), 'one_class_resnet.pth')

# Step 5: Predict on new input images
# Load the saved model
loaded_model = OneClassResNet(num_classes=1)
loaded_model.load_state_dict(torch.load('one_class_resnet.pth'))
loaded_model.eval()

# Preprocess and predict on a new image
new_image_path = 'C:/Workarea/File_Analyser/etc/not_melli/download1.jpg'
new_image = Image.open(new_image_path).convert('RGB')
new_image = data_transform(new_image).unsqueeze(0)  # Add batch dimension
output = loaded_model(new_image)
predicted_class = torch.sigmoid(output).item() > 0.5  # Threshold the output
if predicted_class:
    print(f"The new image belongs to the target class. {torch.sigmoid(output).item()}")
else:
    print("The new image does not belong to the target class.")
