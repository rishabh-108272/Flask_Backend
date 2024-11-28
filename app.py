from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import numpy as np
import torch
from models import flower_model, fruit_model, fruit_transform

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data
        if "image" not in request.files or "model_type" not in request.form:
            return jsonify({"error": "Missing 'image' or 'model_type' field."}), 400

        file = request.files["image"]
        model_type = request.form["model_type"]

        # Load and preprocess image
        image = Image.open(file).convert("RGB")

        if model_type == "flower":
            # Resize and preprocess for TensorFlow model
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = flower_model.predict(image_array)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            return jsonify({
                "model_type": "flower",
                "class_id": int(class_id),
                "confidence": float(confidence)
            })

        elif model_type == "fruit":
            # Resize and preprocess for PyTorch model
            image_tensor = fruit_transform(image).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                prediction = fruit_model(image_tensor)
                probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
                class_id = torch.argmax(probabilities).item()
                confidence = probabilities[class_id].item()

            return jsonify({
                "model_type": "fruit",
                "class_id": int(class_id),
                "confidence": float(confidence)
            })

        else:
            return jsonify({"error": "Invalid model type. Choose 'flower' or 'fruit'."}), 400

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)  # Listen on all network interfaces
