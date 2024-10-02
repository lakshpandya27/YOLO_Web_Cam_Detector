import urllib.request
import cv2
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime

# Load the YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' (Nano model) for faster inference, or 'yolov8s.pt' for more accuracy

url = 'http://192.168.43.51:8080/shot.jpg'

# Define the region of interest (ROI) coordinates (x, y, width, height)
roi_x = 100  # starting x coordinate
roi_y = 200  # starting y coordinate
roi_width = 300  # width of the ROI
roi_height = 200  # height of the ROI

# Folder to save detected car images (updated to the provided path)
save_folder = 'C:/Users/admin/Desktop/PJT/detected_car_img'  # Correct folder path
if not os.path.exists(save_folder):
    os.makedirs(save_folder)  # Create folder if it doesn't exist

# Flag to track if a car has been detected
car_detected = False

while True:
    # Open the URL
    imgRes = urllib.request.urlopen(url)

    # Read and decode the image
    imgNp = np.array(bytearray(imgRes.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    # Define the region of interest (ROI) - crop the image
    roi = img[roi_y:roi_y + roi_height, roi_x: roi_x + roi_width]

    # Perform vehicle detection in the ROI using YOLOv8
    results = model(roi, conf=0.5)  # Adjusted confidence threshold

    # Reset car_detected flag before every detection attempt
    car_detected = False

    # Parse the detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # YOLO returns (x1, y1, x2, y2) coordinates of the detected box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
            label = box.cls  # Class index
            confidence = box.conf  # Confidence score

            # Convert confidence to a float before using it
            confidence_value = float(confidence)

            # If the class detected is a car (YOLOv8 pre-trained model detects cars, trucks, etc.)
            if label == 2:  # 2=car (check class mappings if necessary)
                # Draw a rectangle around the detected vehicle in the ROI
                cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put label and confidence on the bounding box
                cv2.putText(roi, f'Car {confidence_value:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Set the flag for car detected
                car_detected = True

                # Save the detected car image to the folder
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = os.path.join(save_folder, f'car_{timestamp}.jpg')
                cv2.imwrite(save_path, roi)  # Save the ROI containing the detected car

                print(f"Car image saved as {save_path}")

    # If a car has been detected, show an alert
    if car_detected:
        cv2.putText(img, 'Car Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Alert', img)
        cv2.waitKey(1000)  # Show the alert for 1 second

    # Draw a rectangle around the ROI on the full image (optional)
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    # Display the full image with the rectangle and the ROI in separate windows
    cv2.imshow('Full Image', img)  # Show the full image with the rectangle
    cv2.imshow('Region of Interest', roi)  # Show the ROI with detection

    # Exit when 'q' is pressed
    if ord('q') == cv2.waitKey(10):
        break

# Release resources
cv2.destroyAllWindows()
