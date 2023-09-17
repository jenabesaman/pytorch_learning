import cv2
import qrcode
from qrcode import exceptions as qrexceptions

def find_and_decode_qr_code(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the QRCode detector
    detector = cv2.QRCodeDetector()

    # Detect QR codes in the image
    decoded_text = ""
    retval, decoded_info, decoded_points, straight_qrcode = detector.detectAndDecodeMulti(gray_image)

    if retval:
        # Loop through the detected QR codes
        for i, decoded_data in enumerate(decoded_info):
            try:
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=10,
                    border=4,
                )
                qr.add_data(decoded_data)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                qr_img.show()  # Display the QR code image
                decoded_text = decoded_data
            except qrexceptions.DataOverflowError:
                decoded_text = "Data too large for QR code"
            except qrexceptions.VersionError:
                decoded_text = "QR code version error"
            except Exception as e:
                decoded_text = str(e)

    return decoded_text

if __name__ == "__main__":
    image_path = "C:/Workarea/File_Analyser/check/okcheck.jpg"  # Replace with the path to your image
    result = find_and_decode_qr_code(image_path)

    if result:
        print("QR Code Text:", result)
    else:
        print("No QR code found in the image.")