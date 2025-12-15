import io
import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras

# Dynamic path that works on both Windows and Linux
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "best_banknote_model(1).keras")

IMAGE_SIZE = (128, 128)

    
model = keras.models.load_model(MODEL_PATH)
     

def preprocess_image(file_storage):
    image_bytes = file_storage.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, input_array):
    class_mapping = {
        0: 2,
        1: 5,
        2: 10,
        3: 20,
        4: 50,
        5: 100,
        6: 500,
        7: 1000
    }
    
    predictions = model.predict(input_array, verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    predicted_currency = class_mapping.get(predicted_class_idx, predicted_class_idx)
    
    probabilities = {}
    for idx, prob in enumerate(predictions[0]):
        currency = class_mapping.get(idx, idx)
        probabilities[str(currency)] = float(prob)
    
    return str(predicted_currency), confidence, probabilities

app = Flask(__name__)

if model is None:
    print("ERROR: Model failed to load")
else:
    print("SUCCESS")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "koloo tmamm ya basha"
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "Invalid filename"}), 400
    
    try:
        input_array = preprocess_image(file)
        predicted_class, confidence, probabilities = predict(model, input_array)
        
        return jsonify({
            "predicted_class": predicted_class
            ,"confidence_percentage": f"{confidence * 100:.2f}%",
            "probabilities": probabilities
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)