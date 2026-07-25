import os
# Suppress the oneDNN terminal warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2 
import mediapipe as mp
import csv
import urllib.request

# --- 1. Setup MediaPipe Tasks API ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'hand_landmarker.task')

if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    print("Downloading MediaPipe hand tracking model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# --- 2. Setup CSV File ---
csv_file = os.path.join(script_dir, "asl_dataset.csv") 
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
        writer.writerow(header)

# --- 3. Start Data Collection ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
current_letter = 'A'

window_name = 'ASL Data Collector'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("--- ASL Data Collector ---")
print("1. Press letters A-Z on your keyboard to change the current sign.")
print("2. Press SPACEBAR to TOGGLE recording on/off.") # UPDATED
print("3. Press 'ESC' to quit.")

# OPEN THE FILE ONCE BEFORE THE LOOP TO PREVENT I/O LAG
csv_f = open(csv_file, mode='a', newline='')
csv_writer = csv.writer(csv_f)

# Track recording state
is_recording = False 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = landmarker.detect(mp_image)
    
    cv2.putText(frame, f"Current Sign: {current_letter}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Update on-screen instructions
    instruction_text = "Press SPACE to toggle Recording, 'ESC' to Quit"
    cv2.putText(frame, instruction_text, (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    landmarks_data = []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            
            # --- NORMALIZATION ---
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            wrist_z = hand_landmarks[0].z
            temp_landmarks = []
            for landmark in hand_landmarks:
                temp_landmarks.extend([
                    landmark.x - wrist_x,
                    landmark.y - wrist_y,
                    landmark.z - wrist_z
                ])
            
            # --- SCALE NORMALIZATION ---
            max_val = max(list(map(abs, temp_landmarks))) if temp_landmarks else 0
            if max_val > 0:
                landmarks_data = [val / max_val for val in temp_landmarks]
            else:
                landmarks_data = temp_landmarks

    key = cv2.waitKey(1) & 0xFF

    # --- Controls ---
    if key == 27: 
        break
    elif key == ord(' '): 
        # Toggle recording state instead of requiring a hold
        is_recording = not is_recording 
        
    # Handles both uppercase and lowercase inputs cleanly
    elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
        current_letter = chr(key).upper()

    # If recording is toggled ON and hands are detected, save to CSV
    if is_recording:
        if landmarks_data:
            label_index = alphabet.index(current_letter)
            # Write directly using the already-open file object
            csv_writer.writerow([label_index] + landmarks_data)
        
        # Visual feedback
        cv2.putText(frame, "RECORDING...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow(window_name, frame)

# CLEANUP
cap.release()
cv2.destroyAllWindows()
landmarker.close()
csv_f.close() # Close the CSV file safely
print(f"Data collection complete! Saved to {csv_file}")
