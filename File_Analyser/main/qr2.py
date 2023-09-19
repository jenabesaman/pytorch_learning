import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import qrcode


def load_and_process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


def detect_qr_codes(gray_image):
    decoded_objects = decode(gray_image)
    return decoded_objects


def extract_qr_code_text(decoded_objects):
    qr_code_text = []
    for obj in decoded_objects:
        qr_code_text.append(obj.data.decode('utf-8'))
    return qr_code_text


def main(image_path):
    image, gray = load_and_process_image(image_path)
    decoded_objects = detect_qr_codes(gray)
    qr_code_text = extract_qr_code_text(decoded_objects)
    return qr_code_text


# if __name__ == "__main__":
#     image_path = "C:/Workarea/File_Analyser/check/111.jpg"
#     qr_code_text = main(image_path)
#     string_lenght=[len(item) for item in qr_code_text]
#     print(qr_code_text,string_lenght)
#     # print(qr_code_text[2])

if __name__ == "__main__":
    image_path = "C:/Workarea/File_Analyser/check/okcheck.jpg"
    qr_code_text = main(image_path)
    print("QR Code Text:", qr_code_text)


def generate_low_quality_qr_code(text, output_path):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)


generate_low_quality_qr_code("Hello, World!", "low_quality_qr.png")
qr_code_text = main("low_quality_qr.png")
print("QR Code Text:", qr_code_text)

# def generate_low_quality_qr_code(output_path):
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         box_size=10,
#         border=4,
#     )
#     # qr.add_data(text)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
#     img.save(output_path)
#
#
# generate_low_quality_qr_code("low_quality_qr.png")
# qr_code_text = main("low_quality_qr.png")
# print(qr_code_text)
