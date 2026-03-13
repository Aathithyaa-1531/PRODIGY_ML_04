import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# 1. Load the data
df = pd.read_csv('hand_data.csv')

# Separate Features (X) and Labels (y)
X = df.iloc[:, 1:].values.astype('float32') # 42 coordinates
y = df.iloc[:, 0].values.astype('int32')     # Gesture label (0, 1, 2)

# 2. Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the Neural Network
model = models.Sequential([
    layers.Input(shape=(42,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Prevents overfitting
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 classes: Open, Fist, Peace
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train the model
print("Training starting...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 5. Save the trained model
model.save('gesture_recognizer.h5')
print("\nSuccess! 'gesture_recognizer.h5' has been created.")