import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# 1. Setup the Hand Landmarker Tasks
if not os.path.exists('hand_landmarker.task'):
    print("Error: 'hand_landmarker.task' not found. Run the curl command first.")
    exit()

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

header = ['label'] + [f'pt_{i}_{axis}' for i in range(21) for axis in ['x', 'y']]
all_data = []

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera. Check Privacy Settings.")
    exit()

print("--- SYSTEM READY ---")
print("Focus on the 'Hand Data Collection' window.")
print("Press '0' for Open, '1' for Fist, '2' for Peace. Press 'q' to save.")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y])
                x_pos = int(lm.x * image.shape[1])
                y_pos = int(lm.y * image.shape[0])
                cv2.circle(image, (x_pos, y_pos), 5, (0, 255, 0), -1)

            # --- KEYBOARD LOGIC ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                label = chr(key)
                all_data.append([label] + landmarks)
                print(f"Captured {label} | Samples: {len(all_data)}")

    cv2.imshow('Hand Data Collection', image)
    # Force window to front on first run
    cv2.setWindowProperty('Hand Data Collection', cv2.WND_PROP_TOPMOST, 1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if all_data:
    with open('hand_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
    print(f"\nSUCCESS: Saved {len(all_data)} samples to 'hand_data.csv'")

cap.release()
cv2.destroyAllWindows()