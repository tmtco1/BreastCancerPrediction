from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = tf.keras.models.load_model('final_breast_cancer_model.keras')
print("Model loaded successfully!")


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']

    try:
        temp_path = os.path.join("/tmp", image_file.filename)
        image_file.save(temp_path)

        processed_image = load_and_preprocess_image(temp_path)

        prediction = model.predict(processed_image, verbose=0)
        probability = 1 - prediction[0][0]

        prediction_class = "Cancer" if probability >= 0.5 else "Non-Cancer"

        os.remove(temp_path)

        return jsonify({
            "prediction": prediction_class,
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
