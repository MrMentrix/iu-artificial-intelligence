import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed = 69420

# Set the directory containing the emotion subdirectories
data_dir = os.path.join(os.getcwd(), "data", "train")

# Get a list of the subdirectories (emotion labels)
emotion_labels = os.listdir(data_dir)

# Load the images and labels
X = []
y = []
for label in emotion_labels:
    label_dir = os.path.join(data_dir, label)
    for filename in os.listdir(label_dir):
        img_path = os.path.join(label_dir, filename)
        img_file = Image.open(img_path)
        img = np.array(img_file)
        X.append(img)
        y.append(emotion_labels.index(label))

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the pixel values
X = X / 255.0

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotion_labels), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

print(history.history)

for key in history.history.keys():
    print(f"{key}: {history.history[key]}")

model.save(f"./model1.h5")