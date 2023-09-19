import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pyzbar.pyzbar import decode
import cv2

# Load ResNet-101 model
resnet101 = models.resnet152(pretrained=True)
resnet101.eval()

def detect_qr_code(image_path):
    img = cv2.imread(image_path)
    barcodes = decode(img)

    if barcodes:
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            print("QR Code Data:", barcode_data)
    else:
        print("No QR codes found in the image.")


def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Define transformations (resize, normalize, convert to tensor)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of ResNet-101
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

    # Apply transformations to the image
    img = preprocess(img)

    # Add batch dimension (ResNet-101 expects a batch of images)
    img = img.unsqueeze(0)

    return img


def detect_qr_code_in_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Pass the image through the ResNet-101 model
    with torch.no_grad():
        output = resnet101(img)

    # Post-process the output to find QR codes
    # You may need to adapt this part to your specific use case
    # The output from ResNet-101 is a tensor, and you might need to extract features
    # or use additional algorithms to locate QR codes in the image.

    # For example, you can use a QR code detection library like 'pyzbar' on the original image
    detect_qr_code(image_path)


# Provide the path to your image
image_path = 'C:/Workarea/File_Analyser/check/test.jpg'
detect_qr_code_in_image(image_path)

