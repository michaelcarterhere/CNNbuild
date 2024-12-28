# predict_cnn.py

import tensorflow as tf
import numpy as np
from PIL import Image

def load_pretrained_model(model_path='saved_models/mnist_cnn'):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    # 1. Load the trained model
    model = load_pretrained_model()

    # 2. Suppose you have an image 'digit.png' (28x28) ...
    img = Image.open('digit.png').convert('L')
    img = img.resize((28, 28))
    img_arr = np.array(img).reshape((1, 28, 28, 1)) / 255.0

    # 3. Get predictions
    preds = model.predict(img_arr)
    predicted_class = np.argmax(preds)
    print(f"Predicted digit: {predicted_class}")
