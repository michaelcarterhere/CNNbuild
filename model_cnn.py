# model_cnn.py

import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential()

    # Convolutional layer #1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer #2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Dense (fully connected) layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))  # helps reduce overfitting

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
