# train_cnn.py

import tensorflow as tf
import numpy as np
from model_cnn import create_cnn

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape to (num_samples, 28, 28, 1) for channels-last format
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test  = x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create CNN
    model = create_cnn()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,                # Increase to ~10 if you want more training
        batch_size=64
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

    # Save model
    model.save('saved_models/mnist_cnn.keras', save_format='keras')
    print("Model saved to 'saved_models/mnist_cnn'")
