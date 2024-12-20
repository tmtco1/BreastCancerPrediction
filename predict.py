import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('breast_cancer_model.keras')

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Cancer' if prediction[0] > 0.5 else 'Non-Cancer'

while True:
    path = input("Dosya konumunu giriniz (lütfen🫸🫷): ")
    if path=="x":
        break
    else:
        print(predict_image(path))
