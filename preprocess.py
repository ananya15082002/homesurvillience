import os

import cv2

# Load YOLOv8
from ultralytics import YOLO

weights_path = "ultralytics/yolov8n.pt"
model = YOLO(weights_path)

# Step 1: Load the dataset
dataset_path = "dataset"
output_path = "preprocessed_image"
os.makedirs(output_path, exist_ok=True)

# Step 2: Preprocess the images
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        desired_width = 416  # Adjust as needed
        desired_height = 416  # Adjust as needed
        resized_face = cv2.resize(face, (desired_width, desired_height))  # Resize the face
        return resized_face
    else:
        return None

# Step 3: Use YOLO for object detection
def detect_objects(image):
    results = model(image)  # Perform YOLO detection
    return results

# Step 4: Save the preprocessed dataset and labels
def save_preprocessed_image(image, filename):
    cv2.imwrite(os.path.join(output_path, filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # Save image with high quality
    print(f"Saved and preprocessed: {filename}")

# Function to extract labels from filenames
def extract_labels(filename):
    return filename.split('.')[0]  # Assuming filenames are in the format "name.jpg"

# Create or open labels.txt file and write labels
with open(os.path.join(output_path, 'labels.txt'), 'w') as label_file:
    # Iterate over images in the dataset folder
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path)
            
            # Step 2: Preprocess the image
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is not None:
                # Step 3: Object detection using YOLO
                results = detect_objects(preprocessed_image)
                # Process YOLO results as needed
                
                # Step 4: Save the preprocessed image
                save_preprocessed_image(preprocessed_image, filename)
                
                # Extract labels and save them
                label = extract_labels(filename)
                label_file.write(f'{filename},{label}\n')
