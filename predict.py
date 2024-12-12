import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model_path = 'breast_cancer.h5'
test_image_path = input("Resmi sürükleyin bırakın veya adresini yazın: ")

model = tf.keras.models.load_model(model_path)

img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0) 

prediction = model.predict(img_array)[0][0]

print("Kanserli" if prediction < 0.5 else "Normal")