import tensorflow as tf
from tensorflow.keras import layers, models

# Input is 42 values (21 landmarks * 2 coordinates: x and y)
model = models.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax') # Assuming 5 different gestures
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])