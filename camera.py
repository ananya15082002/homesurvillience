import os

import cv2

# Create a directory to save the dataset
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default camera index, change if needed

# Set resolution (optional)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Number of images to capture
num_images = 20

# Start capturing images
print("Capturing images...")
for i in range(num_images):
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Display the frame
    cv2.imshow('Frame', frame)

    # Save the frame as an image
    image_path = os.path.join(dataset_dir, f"image_{i}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image {i+1}/{num_images} captured and saved as {image_path}")

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

print("Dataset capture complete.")
