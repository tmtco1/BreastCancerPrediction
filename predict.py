test_image_path = input("PGM file path: ")

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model_path = 'breast_cancer.keras'

model = tf.keras.models.load_model(model_path)

img = image.load_img(test_image_path, target_size=(224, 224), color_mode='rgb')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

print(f"Prediction: {prediction:.4f}")

if prediction > 0.7:
    print("Infected")
else:
    print("Normal")

img.show()
