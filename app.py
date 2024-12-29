import os
import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load your saved CNN model in proper format
model_path = "saved_models/mnist_cnn"
model = tf.keras.models.load_model('saved_models/mnist_cnn.keras')
model.summary()  # optional, just to see the CNN layers

@app.route("/")
def home():
    return "CNN is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No base64 image data provided"}), 400

    try:
        # Decode the base64 image
        image_data = base64.b64decode(data)
        img = Image.open(io.BytesIO(image_data)).convert("L")
        img = img.resize((28, 28))
        img_arr = np.array(img).reshape((1, 28, 28, 1)) / 255.0

        # Predict with CNN
        preds = model.predict(img_arr)
        predicted_class = int(np.argmax(preds))

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
