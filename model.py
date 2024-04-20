import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load and preprocess your small dataset
dataset_path = "preprocessed_image"
X_data = []
y_labels = []

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (100, 100))  # Resize images to a fixed size
        X_data.append(image)
        label = filename.split(".")[0]  # Extract label from filename
        y_labels.append(label)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

# Encode labels into numerical indices
label_encoder = LabelEncoder()
y_labels_encoded = label_encoder.fit_transform(y_labels)

# Convert labels to one-hot encoded vectors
y_labels_onehot = to_categorical(y_labels_encoded)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels_onehot, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Define a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout layer
    Dense(len(np.unique(y_labels_encoded)), activation='softmax')  # Adjust output units based on the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=8), epochs=50, validation_data=(X_val, y_val))

# Save the label encoder
np.save("label_encoder.npy", label_encoder.classes_)

# Save the trained model
model.save("trained_model.h5")
