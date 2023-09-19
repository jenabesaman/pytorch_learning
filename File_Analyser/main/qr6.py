import cv2
from pyzbar.pyzbar import decode
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load a pre-trained model for image processing, for example, a pre-trained object detection model.
# You can use Faster R-CNN, YOLO, or another model trained on a large dataset.

# Replace 'model' with your own pre-trained model.
model = torch.hub.load('pytorch/vision', 'nvidia_resneXt', pretrained=True)

# Set the model to evaluation mode
model.eval()


def detect_qr_code(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale (QR codes are typically black and white)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find QR codes in the image
    decoded_objects = decode(gray)

    # Iterate through the detected objects and draw rectangles around them
    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        print("QR Code Data:", data)

        # You can also draw rectangles around the QR codes if needed
        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the image with detected QR codes (optional)
    cv2.imshow("QR Code Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Specify the path to your image
image_path = "C:/Workarea/File_Analyser/check/okcheck.jpg"

# Call the detect_qr_code function
detect_qr_code(image_path)
