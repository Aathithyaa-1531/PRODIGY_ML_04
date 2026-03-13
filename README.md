AI Hand Gesture Controller (MediaPipe + TensorFlow)


Project Overview:

This project is an end-to-end Human-Computer Interaction (HCI) system that uses Computer Vision and Deep Learning to control macOS system functions through hand gestures. The system captures hand landmarks in real-time, processes them through a custom-trained Neural Network, and executes system commands like Muting/Unmuting Audio and Taking Screenshots.

System Architecture:

The project follows a classic 3-stage Machine Learning pipeline:


1. Data Engineering (MediaPipe Tasks API)


Using the modern MediaPipe Hand Landmarker, the system extracts 21 key points (x,y coordinates) from the hand.

Input: Raw webcam frames (640×480).

Output: A flattened vector of 42 features representing the skeletal structure of the hand.

Dataset: 300+ custom-captured samples stored in hand_data.csv.


2. Model Training (TensorFlow/Keras)


A custom Multi-Layer Perceptron (MLP) was built to classify the gestures.

Input Layer: 42 neurons (Landmark coordinates).

Hidden Layers: 128 and 64 neurons with ReLU activation and Dropout (0.2) to prevent overfitting.

Output Layer: 3 neurons with Softmax activation (Open Palm, Fist, Peace Sign).

Optimization: Adam optimizer with Sparse Categorical Crossentropy loss.


3. Live Inference & Automation


The final application runs in a real-time loop:

Detection: Identifies hand landmarks at 30+ FPS.

Prediction: Passes landmarks through the .h5 model.

Action: If confidence > 95%, PyAutoGUI triggers system-level hotkeys.


Tech Stack:


Language: Python 3.12

Computer Vision: OpenCV, MediaPipe

Deep Learning: TensorFlow 2.x, Keras

Automation: PyAutoGUI

Data Science: Pandas, NumPy, Scikit-learn

Hardware: Optimized for Apple Silicon (M2 Chip)


OUTPUT: https://drive.google.com/drive/folders/1zI274ftWYFdtL_8jjnPUDpZ4mw-0iABS
