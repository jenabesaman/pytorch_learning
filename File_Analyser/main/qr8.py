import cv2
from pyzbar.pyzbar import decode
import torch
from torchvision import models, transforms
resnet101 = models.resnet152(pretrained=True)
resnet101.eval()
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert to RGB (ResNet-101 expects RGB images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize and normalize the image as required by ResNet-101
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image
def run_resnet101(image):
    with torch.no_grad():
        output = resnet101(image)
    return output


def decode_qr_codes(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Run ResNet-101
    resnet_output = run_resnet101(image)

    # Decode QR codes using pyzbar
    decoded_objects = decode(cv2.imread(image_path))

    return resnet_output, decoded_objects
image_path = "C:/Workarea/File_Analyser/check/qr.jpg"
resnet_output, decoded_objects = decode(image_path)

# print("ResNet-101 Output:")
# print(resnet_output)

print("QR Codes Found:")
for obj in decoded_objects:
    print("Data:", obj.data.decode("utf-8"))
