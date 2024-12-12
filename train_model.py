import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

infected_path = 'imgs/infected'
normal_path = 'imgs/NORM'

def load_data():
    images = []
    labels = []
    for label, folder in enumerate([infected_path, normal_path]):
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0 
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_data()

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.1
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)

history = model.fit(
    train_generator,
    validation_data=(x_test, y_test),
    epochs=50,
    steps_per_epoch=len(x_train) // 32,
    callbacks=[lr_callback]
)

model.save('breast_cancer.keras')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
