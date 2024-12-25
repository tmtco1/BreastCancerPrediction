import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

print("Loading model...")
model = tf.keras.models.load_model('breast_cancer_model.keras')
print("Model loaded successfully!")

while True:
    image_path = input("\nEnter image path (or 'x' to exit): ")
    
    if image_path.lower() == 'x':
        print("Exiting program...")
        break
    
    try:
        processed_image = load_and_preprocess_image(image_path)
        prediction = model.predict(processed_image, verbose=0)
        
        probability = prediction[0][0]
        prediction_class = "Cancer" if probability >= 0.5 else "Non-Cancer"
        
        print(f"\nResult:")
        print(f"Prediction: {prediction_class}")
        print(f"Probability: {probability:.2%}")
        
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")