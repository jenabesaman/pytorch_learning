import cv2
from pyzbar.pyzbar import decode
import numpy as np

def find_and_decode_qr_code(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find QR codes in the grayscale image
    decoded_objects = decode(gray)

    if decoded_objects:
        for obj in decoded_objects:
            # Extract the QR code data
            qr_data = obj.data.decode('utf-8')

            # Draw a rectangle around the QR code (optional)
            points = obj.polygon
            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                cv2.polylines(image, [hull], True, (0, 255, 0), 2)

            # Print the QR code data
            print(f"QR Code Data: {qr_data}")

        # Display the image with QR code detection (optional)
        cv2.imshow("QR Code Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No QR code found in the image.")

if __name__ == "__main__":
    image_path = "C:/Workarea/File_Analyser/check/qr.jpg"  # Replace with the path to your image
    find_and_decode_qr_code(image_path)
