import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time

# 1. Load your trained model
model = tf.keras.models.load_model('gesture_recognizer.h5')
gesture_names = ["Open Palm", "Fist", "Peace Sign"]

# 2. Setup MediaPipe Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# MediaPipe connections for drawing the "skeleton"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

cap = cv2.VideoCapture(0)
last_action_time = 0  # Cooldown timer

print("--- LIVE SMART CONTROLLER STARTED ---")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            input_data = []
            landmark_px = []

            for lm in hand_landmarks:
                input_data.extend([lm.x, lm.y])
                landmark_px.append((int(lm.x * w), int(lm.y * h)))

            # Draw Skeleton
            for connection in HAND_CONNECTIONS:
                cv2.line(image, landmark_px[connection[0]], landmark_px[connection[1]], (0, 255, 0), 2)
            for pt in landmark_px:
                cv2.circle(image, pt, 4, (0, 0, 255), -1)

            # 3. Predict Gesture
            prediction = model.predict(np.array([input_data]), verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            
            # 4. Action Logic (Cooldown of 2 seconds)
            if confidence > 0.95 and (time.time() - last_action_time) > 2:
                current_gesture = gesture_names[class_id]
                
                if current_gesture == "Fist":
                    pyautogui.press('volumemute')
                    print("Action: Muted Audio")
                    last_action_time = time.time()
                
                elif current_gesture == "Peace Sign":
                    pyautogui.hotkey('command', 'shift', '3')
                    print("Action: Screenshot Saved")
                    last_action_time = time.time()

                # Display active action
                cv2.putText(image, "ACTION TRIGGERED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Display Label
            if confidence > 0.8:
                label = f"{gesture_names[class_id]} {int(confidence*100)}%"
                cv2.rectangle(image, (40, 15), (350, 65), (255, 0, 0), -1)
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2)

    cv2.imshow('Smart Gesture Controller', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()