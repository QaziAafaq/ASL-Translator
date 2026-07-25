import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import urllib.request

# --- 1. Setup MediaPipe Model ---
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe hand tracking model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        
    (0, 5), (5, 6), (6, 7), (7, 8),        
    (5, 9), (9, 10), (10, 11), (11, 12),   
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

# --- 2. Setup TensorFlow Model ---
# Provide an alphabet mapping used for Keras models
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Try to load either a pickled sklearn-like model (`ASL_model.p`)
# or a Keras model saved as `asl_model.h5`. If a Keras model is
# found we wrap it to provide a `predict_proba` method and
# `classes_` attribute to keep the rest of the code compatible.
model = None
try:
    import pickle
    model_dict = pickle.load(open('ASL_model.p', 'rb'))
    model = model_dict.get('model', model_dict)
    print("Loaded sklearn-style model from 'ASL_model.p'.")
except Exception:
    try:
        keras_model = tf.keras.models.load_model('asl_model.h5')

        class KerasWrapper:
            def __init__(self, keras_model, classes):
                self._m = keras_model
                self.classes_ = np.array(classes)

            def predict_proba(self, X):
                # Keras `predict` already returns class probabilities
                return self._m.predict(X)

        model = KerasWrapper(keras_model, alphabet)
        print("Loaded Keras model from 'asl_model.h5' and wrapped for compatibility.")
    except Exception as e:
        print(f"Warning: No usable model found ('ASL_model.p' or 'asl_model.h5'). Error: {e}")
        model = None

# --- 3. GUI Application Class ---
class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time ASL Translator")
        self.root.geometry("800x750")
        self.root.configure(bg="#2b2b2b")
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.sentence = ""
        self.recent_predictions = []
        self.frames_to_hold = 15  # Needs 15 consistent frames to type a letter (increased for accuracy)
        self.low_conf_streak = 0  # Track consecutive low-confidence frames
        self.waiting_for_gap = False  # Must lower/move hand before same letter repeats

        # --- EMA smoothing for stable landmark drawing ---
        # alpha: lower = smoother/more lag, higher = snappier/more jitter
        self.ema_alpha = 0.35
        self.smoothed_landmarks = None  # Will hold 21 smoothed (x, y) pixel positions

        self.cap = cv2.VideoCapture(0)
        self.landmarker = HandLandmarker.create_from_options(options)

        # UI Layout
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        self.sign_label = tk.Label(self.root, text="Sign: -- (0.0%)", font=("Helvetica", 18, "bold"), fg="#00ff00", bg="#2b2b2b")

        self.alt_label = tk.Label(self.root, text="Alt: --", font=("Helvetica", 13), fg="#aaaaaa", bg="#2b2b2b")
        self.sign_label.pack()
        self.alt_label.pack()

        self.sentence_label = tk.Label(self.root, text="Text: ", font=("Helvetica", 24, "bold"), fg="white", bg="#2b2b2b")
        self.sentence_label.pack(pady=10)

        self.btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        self.btn_frame.pack(pady=10)

        self.btn_space = tk.Button(self.btn_frame, text="Add Space", font=("Helvetica", 14), command=self.add_space, width=12)
        self.btn_space.grid(row=0, column=0, padx=10)

        self.btn_clear = tk.Button(self.btn_frame, text="Clear Text", font=("Helvetica", 14), command=self.clear_text, width=12)
        self.btn_clear.grid(row=0, column=1, padx=10)

        self.btn_quit = tk.Button(self.btn_frame, text="Quit", font=("Helvetica", 14), command=self.quit_app, width=12, fg="red")
        self.btn_quit.grid(row=0, column=2, padx=10)

        self.update_frame()

    def add_space(self):
        self.sentence += " "
        self.sentence_label.config(text=f"Text: {self.sentence}")

    def clear_text(self):
        self.sentence = ""
        self.sentence_label.config(text="Text: ")

    def quit_app(self):
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'landmarker'):
            self.landmarker.close() 
        self.root.destroy()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = self.landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    h, w = frame.shape[:2]

                    # --- Build raw pixel positions for this frame ---
                    raw_pts = [(landmark.x * w, landmark.y * h)
                               for landmark in hand_landmarks]

                    # --- Apply EMA smoothing ---
                    if self.smoothed_landmarks is None:
                        # First detection: initialise directly
                        self.smoothed_landmarks = raw_pts
                    else:
                        a = self.ema_alpha
                        self.smoothed_landmarks = [
                            (a * rx + (1 - a) * sx,
                             a * ry + (1 - a) * sy)
                            for (rx, ry), (sx, sy)
                            in zip(raw_pts, self.smoothed_landmarks)
                        ]

                    spts = self.smoothed_landmarks  # shorthand

                    # --- Draw skeleton using smoothed points ---
                    for connection in HAND_CONNECTIONS:
                        x1, y1 = int(spts[connection[0]][0]), int(spts[connection[0]][1])
                        x2, y2 = int(spts[connection[1]][0]), int(spts[connection[1]][1])
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)

                    # --- Draw landmark dots using smoothed points ---
                    for sx, sy in spts:
                        cv2.circle(frame, (int(sx), int(sy)), 5, (0, 255, 255), -1)

                    temp_landmarks = []
                    wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z

                    for landmark in hand_landmarks:
                        
                        # Normalize data relative to wrist (use raw values for prediction)
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

                    # Prediction Logic
                    if model is not None:
                        input_data = np.array([landmarks_data], dtype=np.float32)
                        prediction_probs = model.predict_proba(input_data)[0]

                        # Top-2 predictions for disambiguation display
                        top2_idx = np.argsort(prediction_probs)[::-1][:2]
                        class_idx  = top2_idx[0]
                        confidence = prediction_probs[class_idx]
                        alt_idx    = top2_idx[1]
                        alt_conf   = prediction_probs[alt_idx]

                        # Handle classes_ array and fallback if needed
                        if hasattr(model, 'classes_'):
                            predicted_letter = str(model.classes_[class_idx]).upper()
                            alt_letter       = str(model.classes_[alt_idx]).upper()
                        else:
                            predicted_letter = alphabet[class_idx]
                            alt_letter       = alphabet[alt_idx]

                        # Threshold: 70% primary confidence
                        if confidence > 0.70:
                            # predicted_letter is already set above
                            # alt_letter is already set above

                            self.sign_label.config(
                                text=f"Sign: {predicted_letter} ({confidence*100:.1f}%)",
                                fg="#00ff00"
                            )
                            self.alt_label.config(
                                text=f"Alt: {alt_letter} ({alt_conf*100:.1f}%)"
                            )

                            self.recent_predictions.append(predicted_letter)
                            if len(self.recent_predictions) > self.frames_to_hold:
                                self.recent_predictions.pop(0)
                            self.low_conf_streak = 0

                            # If held steady for frames_to_hold, add to sentence
                            if (len(self.recent_predictions) == self.frames_to_hold
                                    and all(p == predicted_letter for p in self.recent_predictions)):
                                if not self.waiting_for_gap:
                                    self.sentence += predicted_letter
                                    self.sentence_label.config(text=f"Text: {self.sentence}")
                                    self.waiting_for_gap = True  # Require a gap before same letter
                                self.recent_predictions.clear()
                        else:
                            self.low_conf_streak += 1
                            # Clear buffer after 5 consecutive low-confidence frames
                            if self.low_conf_streak >= 5:
                                self.recent_predictions.clear()
                                self.low_conf_streak = 0
                                self.waiting_for_gap = False  # Gap achieved — same letter allowed again
                            self.sign_label.config(
                                text="Sign: -- (Low Confidence)",
                                fg="#ff6666"
                            )
                            self.alt_label.config(text="Alt: --")

            else:
                # No hand detected — reset smoothed buffer so markers don't ghost
                self.smoothed_landmarks = None
                self.waiting_for_gap = False   # Gap achieved by removing hand
                self.recent_predictions.clear()

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk 
            self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()