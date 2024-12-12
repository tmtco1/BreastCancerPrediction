import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

infected_path = 'imgs/infected'
normal_path = 'imgs/NORM'

def load_data():
    images = []
    labels = []

    for label, folder in enumerate([infected_path, normal_path]):
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = image.load_img(img_path, target_size=(224, 224))  # Boyutlandırma
            img_array = image.img_to_array(img) / 255.0  # Normalizasyon
            images.append(img_array)
            labels.append(label)  # Kanserli: 0, Normal: 1

    return np.array(images), np.array(labels)

images, labels = load_data()

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # İkili sınıflandırma için sigmoid aktivasyonu
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

model.save('breast_cancer.h5')