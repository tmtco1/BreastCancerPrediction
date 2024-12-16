import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
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
            img = image.load_img(img_path, target_size=(300, 300))
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)

images, labels = load_data()

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

for layer in base_model.layers:
    layer.trainable = True

model = Sequential([
    base_model,
    BatchNormalization(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=16
)

results = model.evaluate(x_test, y_test, verbose=0)
test_loss, test_acc = results[0], results[1]
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

model.save('breast_cancer.keras')
