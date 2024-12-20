import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

cancer_dir = "/mnt/c/Users/MONSTER/Documents/BreastCancerPrediction/datset/augmented/Cancer"
non_cancer_dir = "/mnt/c/Users/MONSTER/Documents/BreastCancerPrediction/datset/augmented/Non-Cancer"

def load_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((150, 150))
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            pass
    return np.array(images), np.array(labels)

cancer_images, cancer_labels = load_images(cancer_dir, 1)
non_cancer_images, non_cancer_labels = load_images(non_cancer_dir, 0)

X = np.concatenate((cancer_images, non_cancer_images), axis=0)
y = np.concatenate((cancer_labels, non_cancer_labels), axis=0)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

model.save('breast_cancer_model.keras')
