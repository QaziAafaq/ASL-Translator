# SignBridge ASL Translator

This repository contains a real-time American Sign Language (ASL) recognition system that uses MediaPipe hand landmark detection and a TensorFlow model to translate hand signs into text.

## What’s included

- `web.py` — Flask web app with MJPEG streaming, live hand skeleton overlay, and inference state updates.
- `templates/index.html` — front-end UI for the web app.
- `Collect_data.py` — webcam data collection tool for building `asl_dataset.csv` from hand landmarks.
- `train 2.py` — training script that loads `asl_dataset.csv`, augments data, trains a TensorFlow model, and saves `asl_model.h5`.
- `main.py` — local Tkinter desktop ASL translator app using the same MediaPipe + model stack.
- `asl_model.h5` — trained Keras model file used by the web and desktop apps.
- `asl_dataset.csv` — landmark dataset used for training.

## Architecture

1. `Collect_data.py` captures hand landmark frames from the camera.
   - Uses MediaPipe hand landmarker.
   - Normalizes all 21 hand landmarks relative to the wrist.
   - Stores 63 features per frame: `x`, `y`, `z` offsets for 21 landmarks.

2. `train 2.py` loads the dataset, augments it, trains a TensorFlow model, and saves `asl_model.h5`.
   - Uses a fully connected neural network.
   - Applies BatchNorm and Dropout for robustness.
   - Trains on normalized landmark vectors with 26 letter classes.

3. `web.py` runs live inference.
   - Captures webcam frames with OpenCV.
   - Runs MediaPipe on each frame to extract landmarks.
   - Normalizes raw landmarks exactly like the training pipeline.
   - Sends the normalized 63-feature vector to the TensorFlow model.
   - Streams annotated frames to a browser using MJPEG.
   - Keeps camera capture and TensorFlow inference in separate threads to reduce lag.

## Requirements

- Python 3.10+ (3.11 recommended)
- `opencv-python`
- `mediapipe`
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `flask`
- `pillow`

## Setup

Install dependencies in your environment:

```bash
python -m pip install --upgrade pip
python -m pip install numpy opencv-python mediapipe tensorflow pandas scikit-learn flask pillow
```

> On Windows, setting up `mediapipe` and `opencv-python` may require a compatible Python and Visual Studio build tools installation.

## Usage

### 1. Collect training data

Run the data collector and press a letter key to select the label. Press `Space` to toggle recording.

```bash
python Collect_data.py
```

### 2. Train the model

Run the training script. It loads `asl_dataset.csv`, augments the data, and saves `asl_model.h5`.

```bash
python "train 2.py"
```

### 3. Run the web app

Start the Flask app and open the browser.

```bash
python web.py
```

Then open:

```text
http://127.0.0.1:5000
```

### 4. Run the local desktop app

Use the Tkinter-based UI for local inference.

```bash
python main.py
```

## Notes on performance

- `web.py` separates camera capture from model inference in two threads.
- The capture thread handles MediaPipe and JPEG streaming.
- The inference thread processes only the latest normalized landmark vector.
- Using MJPEG and lower JPEG quality helps reduce browser lag.
- The app is optimized for 640×480 resolution at 30 FPS.

## Troubleshooting

- If camera cannot open, verify the webcam is connected and not in use by another app.
- If the model fails to load, make sure `asl_model.h5` exists and is compatible with your TensorFlow version.
- If MediaPipe download fails, check your internet connection and rerun the script.

## GitHub Actions workflow

This project includes a GitHub Actions workflow at `.github/workflows/asl-model.yml`.
It validates Python files, installs required packages, checks imports, and optionally runs training on manual dispatch.

## File references

- `web.py`: web server + camera capture + inference
- `Collect_data.py`: dataset capture and landmark normalization
- `train 2.py`: training pipeline and model saving
- `main.py`: desktop translator UI
- `templates/index.html`: web UI and live prediction display

## License

This repository does not include a license file. Add a license if you plan to publish or share the project.
