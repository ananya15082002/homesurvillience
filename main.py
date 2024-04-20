import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("trained_model.h5")

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder.npy", allow_pickle=True)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the detected face
def preprocess_face(face):
    face = cv2.resize(face, (100, 100))
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize pixel values
    return face

# Function to predict the name of the person
def predict_name(face):
    face = preprocess_face(face)
    predictions = model.predict(face)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_name = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_name

# Function to display alarm
def raise_alarm():
    print("Person not identified! Raise alarm...")

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Set the video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract detected face
        face = frame[y:y+h, x:x+w]

        # Predict the name of the person
        name = predict_name(face)

        # Draw bounding box and label on the face
        if name == "Unknown":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Draw red rectangle for unknown person
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display "Unknown" in red
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangle for identified person
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Check for alarm condition
    if len(faces) == 0:
        raise_alarm()

    # Break the loop if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
