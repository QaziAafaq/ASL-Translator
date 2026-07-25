"""
web.py  -  SignBridge Flask server  (performance-optimised)
============================================================
Run with:
    python web.py
Then open http://127.0.0.1:5000 in your browser.

Performance architecture
------------------------
Thread 1 - _capture_loop : Reads camera, runs MediaPipe (fast ~17ms), draws skeleton perfectly in sync,
                           encodes JPEG, and pushes feature arrays to _infer_queue.
Thread 2 - _infer_loop   : Runs TensorFlow predict (slow ~275ms) asynchronously on the latest feature array.
Flask    - _mjpeg_generator: Streams the latest annotated JPEG to the browser.

The two threads are decoupled so the TF model NEVER blocks the camera feed or the MediaPipe tracking.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import threading
import time
import re
import json
import urllib.request
import urllib.parse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ── TensorFlow (graceful fallback) ───────────────────────────────────────────
try:
    import tensorflow as tf
    _model = tf.keras.models.load_model('asl_model.h5')
    print("[SignBridge] CNN-LSTM model loaded.")
except Exception as _e:
    _model = None
    print(f"[SignBridge] Model not loaded ({_e}). Run train.py first.")

# ── Flask ────────────────────────────────────────────────────────────────────
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# ── MediaPipe ────────────────────────────────────────────────────────────────
_MP_MODEL = 'hand_landmarker.task'
if not os.path.exists(_MP_MODEL):
    print("[SignBridge] Downloading MediaPipe model ...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        _MP_MODEL
    )

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MP_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
    )
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Shared state ─────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "current_sign":   "-",
    "confidence":     0.0,
    "alt_sign":       "-",
    "alt_confidence": 0.0,
    "sentence":       "",
}

# Latest annotated JPEG (produced by capture thread, read by MJPEG generator)
_jpeg_lock   = threading.Lock()
_latest_jpeg = None          # bytes | None

# Hand off features to the slow TF model (0-latency queue using Lock/Event)
_infer_lock  = threading.Lock()
_infer_event = threading.Event()
_latest_features  = None     # np.array of shape (1, 10, 63)
_latest_landmarks = None     # raw landmarks (for disambiguation)


# ── Helpers ───────────────────────────────────────────────────────────────────
SUGGESTION_WORDS = [
    'a', 'about', 'above', 'after', 'again', 'all', 'also', 'and', 'any', 'are',
    'as', 'at', 'back', 'be', 'because', 'been', 'before', 'better', 'between',
    'both', 'but', 'by', 'can', 'could', 'day', 'do', 'down', 'even', 'every',
    'few', 'find', 'first', 'for', 'from', 'get', 'give', 'go', 'good', 'great',
    'had', 'has', 'have', 'he', 'help', 'her', 'here', 'him', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'know', 'like', 'little',
    'long', 'look', 'made', 'make', 'man', 'many', 'may', 'me', 'more', 'most',
    'my', 'name', 'new', 'no', 'not', 'now', 'of', 'on', 'one', 'only', 'or',
    'other', 'our', 'out', 'over', 'people', 'said', 'same', 'see', 'she', 'so',
    'some', 'still', 'such', 'take', 'than', 'that', 'the', 'their', 'them',
    'then', 'there', 'these', 'they', 'thing', 'think', 'this', 'time', 'to',
    'two', 'up', 'use', 'very', 'want', 'was', 'way', 'we', 'well', 'what',
    'when', 'which', 'who', 'will', 'with', 'work', 'would', 'write', 'year',
    'you', 'your'
]
NEXT_WORD_SUGGESTIONS = {
    'my': ['name', 'friend', 'idea', 'life', 'best', 'favorite'],
    'your': ['name', 'idea', 'choice', 'life', 'friend', 'time'],
    'the': ['best', 'same', 'next', 'other', 'first', 'last'],
    'name': ['is', 'was', 'should', 'could', 'might', 'will'],
    'i': ['am', 'have', 'need', 'want', 'can', 'will'],
    'you': ['are', 'have', 'can', 'will', 'should', 'must'],
    'it': ['is', 'was', 'can', 'might', 'should', 'could'],
    'this': ['is', 'was', 'can', 'might', 'should', 'could']
}
COMMON_ASL_WORDS = {
    'hello': ['HELLO'],
    'thanks': ['THANK YOU'],
    'thankyou': ['THANK YOU'],
    'please': ['PLEASE'],
    'sorry': ['SORRY'],
    'yes': ['YES'],
    'no': ['NO'],
    'love': ['LOVE'],
    'help': ['HELP'],
    'stop': ['STOP'],
    'more': ['MORE'],
}


def _english_grammar_feedback(sentence):
    sentence = sentence or ''
    sentence = sentence.strip()
    issues = []
    if not sentence:
        return issues

    if re.search(r'\s{2,}', sentence):
        issues.append('Remove extra spaces')

    if sentence[0].islower():
        issues.append('Capitalize the first word')

    # Check if any sentence after a period/!/? starts with lowercase
    parts = re.split(r'[.!?]\s+', sentence)
    if len(parts) > 1 and any(p and p[0].islower() for p in parts[1:]):
        issues.append('Capitalize the first word after punctuation')

    # Only flag all-caps when there are 2+ consecutive all-caps words (single caps word is normal ASL)
    cap_words = re.findall(r'\b[A-Z]{2,}\b', sentence)
    known_acronyms = {'ASL', 'USA', 'NASA', 'FBI', 'CNN', 'UK', 'EU'}
    non_acronym_caps = [w for w in cap_words if w not in known_acronyms]
    if len(non_acronym_caps) >= 2:
        issues.append('Avoid using all caps words')

    if re.search(r'\b(\w+)\s+\1\b', sentence, flags=re.I):
        issues.append('Avoid repeating the same word')

    if re.search(r'\b(must|should|could|would|can|will)\b', sentence, flags=re.I) and re.search(r'\b(need|want|like|love|hate)\s+\b', sentence, flags=re.I):
        issues.append('Check verb usage and sentence flow')

    # Only suggest punctuation when there is at least one complete word (sentence has a space)
    if ' ' in sentence and sentence[-1] not in '.?!':
        issues.append('Consider adding punctuation at the end')

    return issues


def _extract_last_word(sentence):
    matches = re.findall(r"[A-Za-z0-9']+", (sentence or '').strip())
    return matches[-1].lower() if matches else ''


def _fix_english_grammar(sentence):
    sentence = sentence or ''
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    if not sentence:
        return sentence

    allowed_acronyms = {'ASL', 'USA', 'NASA', 'FBI', 'CNN', 'UK', 'EU'}
    words = sentence.split(' ')
    fixed_words = []

    for i, word in enumerate(words):
        cleaned = re.sub(r'[^A-Za-z0-9]', '', word)
        if cleaned.upper() in allowed_acronyms:
            fixed_word = word.upper()
        elif cleaned.isupper() and len(cleaned) > 1:
            fixed_word = word.capitalize()
        elif word.lower() == 'i':
            fixed_word = 'I'
        else:
            fixed_word = word

        if i == 0 and fixed_word:
            fixed_word = fixed_word[0].upper() + fixed_word[1:]

        if fixed_words and fixed_word.lower() == fixed_words[-1].lower():
            continue
        fixed_words.append(fixed_word)

    sentence = ' '.join(fixed_words)
    if len(sentence) > 0 and sentence[-1] not in '.?!':
        sentence += '.'

    return sentence


def _suggest_small_words(prefix, sentence_ended_word=False):
    """Return up to 6 word suggestions.
    - prefix: the current last word (lowercase).
    - sentence_ended_word: True when the sentence ends with a space
      (user just finished a word and wants the NEXT word suggested).
    """
    FALLBACK = ['the', 'is', 'and', 'my', 'you', 'i']

    if not prefix:
        return SUGGESTION_WORDS[:6]

    prefix_lc = prefix.lower()

    # Sentence just ended a word → suggest what comes AFTER that word
    if sentence_ended_word:
        if prefix_lc in NEXT_WORD_SUGGESTIONS:
            return NEXT_WORD_SUGGESTIONS[prefix_lc]
        # Generic: start of-sentence words that could follow anything
        return FALLBACK

    # Mid-word prefix match in our word list
    starts = [w for w in SUGGESTION_WORDS if w.startswith(prefix_lc)]
    if starts:
        # Exact match and only one → offer next-word suggestions
        if len(starts) == 1 and starts[0] == prefix_lc:
            return NEXT_WORD_SUGGESTIONS.get(prefix_lc, FALLBACK)
        return starts[:6]

    # No prefix match → contains match
    contains = [w for w in SUGGESTION_WORDS if prefix_lc in w]
    if contains:
        return contains[:6]

    # Nothing found at all (e.g. a long uncommon word) → fallback
    return FALLBACK


def _convert_sentence_to_asl(sentence):
    words = []
    for raw_word in (sentence or '').strip().split():
        clean_word = re.sub(r'[^A-Za-z0-9]', '', raw_word).lower()
        if not clean_word:
            continue

        if clean_word in COMMON_ASL_WORDS:
            words.append({
                'word': raw_word,
                'asl': COMMON_ASL_WORDS[clean_word],
                'type': 'word'
            })
        else:
            letters = [ch.upper() for ch in clean_word if ch.isalpha()]
            if letters:
                words.append({
                    'word': raw_word,
                    'asl': letters,
                    'type': 'fingerspell'
                })
    return words


def _disambiguate_signs(predicted, alt, landmarks):
    signs = {predicted, alt}
    if signs == {'X', 'J'}:
        pinky_up   = (landmarks[17].y - landmarks[20].y) > 0.04
        index_hook = (landmarks[8].y  - landmarks[6].y)  > -0.02
        if pinky_up and not index_hook:
            return 'J'
        if index_hook and not pinky_up:
            return 'X'
        return predicted
        
    if signs == {'R', 'U'}:
        is_right_hand = landmarks[5].x < landmarks[17].x
        if is_right_hand:
            crossed = landmarks[8].x > landmarks[12].x
        else:
            crossed = landmarks[8].x < landmarks[12].x
        return 'R' if crossed else 'U'

    return predicted


# ─────────────────────────────────────────────────────────────────────────────
# THREAD 1 – Capture & MediaPipe
#   Reads camera, runs MediaPipe, extracts a single normalized frame of landmarks,
#   draws skeleton overlay, and sends the latest feature vector to the inference thread.
# ─────────────────────────────────────────────────────────────────────────────
def _capture_loop():
    global _latest_jpeg, _latest_features, _latest_landmarks

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW avoids WDM latency on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)              # fallback
    if not cap.isOpened():
        print("[SignBridge] ERROR: cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
    ema_alpha      = 0.35
    smoothed_pts   = None
    missed_frames  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = _landmarker.detect(mp_img)

        if result.hand_landmarks:
            missed_frames = 0
            hand_landmarks = result.hand_landmarks[0]
            h, w = frame.shape[:2]

            raw_pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks]
            if smoothed_pts is None:
                smoothed_pts = raw_pts
            else:
                a = ema_alpha
                smoothed_pts = [
                    (a * rx + (1 - a) * sx, a * ry + (1 - a) * sy)
                    for (rx, ry), (sx, sy) in zip(raw_pts, smoothed_pts)
                ]

            for c in HAND_CONNECTIONS:
                x1, y1 = int(smoothed_pts[c[0]][0]), int(smoothed_pts[c[0]][1])
                x2, y2 = int(smoothed_pts[c[1]][0]), int(smoothed_pts[c[1]][1])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)
            for sx, sy in smoothed_pts:
                cv2.circle(frame, (int(sx), int(sy)), 5, (0, 255, 255), -1)

            wrist_x, wrist_y, wrist_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            raw_feat = []
            for lm in hand_landmarks:
                raw_feat.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

            max_v = max(map(abs, raw_feat)) if raw_feat else 0
            feat = [v / max_v for v in raw_feat] if max_v > 0 else raw_feat

            inp = np.array([feat], dtype=np.float32)
            with _infer_lock:
                _latest_features  = inp
                _latest_landmarks = hand_landmarks
            _infer_event.set()
        else:
            missed_frames += 1
            if missed_frames > 3:
                smoothed_pts = None
                with _state_lock:
                    _state["current_sign"]   = "-"
                    _state["confidence"]     = 0.0
                    _state["alt_sign"]       = "-"
                    _state["alt_confidence"] = 0.0
                with _infer_lock:
                    _latest_features  = None
                    _latest_landmarks = None
                _infer_event.set()

        with _state_lock:
            sign = _state["current_sign"]
            conf = _state["confidence"]

        if sign != "-":
            cv2.putText(
                frame,
                f"{sign}  {conf * 100:.0f}%",
                (14, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (0, 217, 255), 2, cv2.LINE_AA
            )

        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# THREAD 2 – Inference
#   Waits for the latest normalized landmark vector, runs the trained model,
#   and updates the app state without blocking camera capture.
# ─────────────────────────────────────────────────────────────────────────────
def _infer_loop():
    frames_to_hold   = 5
    conf_threshold   = 0.70
    low_streak_limit = 4

    recent_preds    = []
    waiting_for_gap = False
    low_conf_streak = 0

    while True:
        _infer_event.wait()
        with _infer_lock:
            inp       = _latest_features
            landmarks = _latest_landmarks
            _infer_event.clear()

        if inp is None or _model is None:
            recent_preds.clear()
            low_conf_streak = 0
            waiting_for_gap = False
            continue

        pred = _model.predict(inp, verbose=0)[0]
        top2 = np.argsort(pred)[::-1][:2]
        ci, cf  = int(top2[0]), float(pred[top2[0]])
        ai, acf = int(top2[1]), float(pred[top2[1]])

        detected_sign = "-"

        if cf > conf_threshold:
            letter = _disambiguate_signs(ALPHABET[ci], ALPHABET[ai], landmarks)
            alt    = ALPHABET[ai]
            detected_sign = letter

            recent_preds.append(letter)
            if len(recent_preds) > frames_to_hold:
                recent_preds.pop(0)
            low_conf_streak = 0

            if len(recent_preds) == frames_to_hold and all(p == letter for p in recent_preds):
                if not waiting_for_gap:
                    with _state_lock:
                        _state["sentence"] += letter
                    waiting_for_gap = True
                recent_preds.clear()
        else:
            low_conf_streak += 1
            if low_conf_streak >= low_streak_limit:
                recent_preds.clear()
                low_conf_streak = 0
                waiting_for_gap = False

        with _state_lock:
            if detected_sign != "-":
                _state["current_sign"]   = detected_sign
                _state["confidence"]     = cf
                _state["alt_sign"]       = alt
                _state["alt_confidence"] = acf
            # if nothing detected, keep the last visible state until the capture thread clears it


# ── Start background threads ──────────────────────────────────────────────────
threading.Thread(target=_capture_loop, daemon=True, name="capture").start()
threading.Thread(target=_infer_loop,   daemon=True, name="infer").start()


# ── MJPEG generator ───────────────────────────────────────────────────────────
def _mjpeg_generator():
    boundary = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    last_jpeg = None
    while True:
        with _jpeg_lock:
            jpeg = _latest_jpeg
        if jpeg and jpeg is not last_jpeg:
            yield boundary + jpeg + b'\r\n'
            last_jpeg = jpeg
        else:
            time.sleep(0.01)   # avoid maxing out CPU and sending duplicate frames


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_state')
def get_state():
    with _state_lock:
        return jsonify(dict(_state))


@app.route('/action', methods=['POST'])
def action():
    data = request.get_json(force=True, silent=True) or {}
    act  = data.get('action', '')
    word = data.get('word', '')

    with _state_lock:
        if act == 'space':
            _state['sentence'] += ' '

        elif act == 'clear':
            if _state['sentence']:
                _state['sentence'] = _state['sentence'][:-1]

        elif act == 'clear_all':
            _state['sentence'] = ''

        elif act == 'apply_suggestion':
            sentence = _state['sentence']
            parts = sentence.rsplit(' ', 1)
            if len(parts) == 1:
                _state['sentence'] = word
            else:
                _state['sentence'] = parts[0].rstrip() + ' ' + word

        elif act == 'apply_correction':
            corrected = (data.get('text') or '').strip()
            if corrected:
                _state['sentence'] = corrected

        sentence = _state['sentence']

    return jsonify({'ok': True, 'sentence': sentence})


# ── External API helpers ─────────────────────────────────────────────────────

def _datamuse_suggest(prefix: str, last_word: str, sentence_ended: bool) -> list | None:
    """Fetch suggestions from Datamuse. Returns list[str] or None on error."""
    try:
        if sentence_ended and last_word:
            # Words that commonly follow `last_word`
            url = ('https://api.datamuse.com/words?'
                   + urllib.parse.urlencode({'lc': last_word, 'max': 6}))
        elif prefix:
            # Autocomplete prefix
            url = ('https://api.datamuse.com/sug?'
                   + urllib.parse.urlencode({'s': prefix, 'max': 6}))
        else:
            url = 'https://api.datamuse.com/words?f=common&max=6'
        req = urllib.request.Request(url,
                                     headers={'User-Agent': 'SignBridge/1.0'})
        with urllib.request.urlopen(req, timeout=2) as resp:
            items = json.loads(resp.read().decode('utf-8'))
            words = [it['word'] for it in items if it.get('word')]
            return words or None
    except Exception:
        return None


def _languagetool_check(sentence: str) -> tuple[list, str]:
    """Check grammar with LanguageTool. Returns (issues, fixed_sentence) or ([], '') on error."""
    try:
        payload = urllib.parse.urlencode({
            'text': sentence,
            'language': 'en-US',
            'enabledOnly': 'false',
        }).encode('utf-8')
        req = urllib.request.Request(
            'https://api.languagetool.org/v2/check',
            data=payload,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'SignBridge/1.0',
            },
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        matches = data.get('matches', [])
        issues = [m['message'] for m in matches]

        # Build fixed sentence by applying replacements right-to-left
        fixed = sentence
        for m in sorted(matches, key=lambda x: x['offset'], reverse=True):
            reps = m.get('replacements', [])
            if not reps:
                continue
            offset = m['offset']
            length = m['length']
            fixed = fixed[:offset] + reps[0]['value'] + fixed[offset + length:]

        return issues, fixed
    except Exception:
        return [], ''


@app.route('/analysis')
def analysis():
    sentence = request.args.get('sentence', '')
    last_word = _extract_last_word(sentence)
    sentence_ended_word = sentence.endswith(' ')

    # Run Datamuse and LanguageTool in parallel to keep latency low
    sug_result: list = []
    lt_issues: list = []
    lt_fixed: str = ''

    def _run_datamuse():
        nonlocal sug_result
        result = _datamuse_suggest(last_word, last_word, sentence_ended_word)
        if result is not None:
            sug_result = result

    def _run_lt():
        nonlocal lt_issues, lt_fixed
        issues, fixed = _languagetool_check(sentence)
        lt_issues = issues
        lt_fixed = fixed

    t1 = threading.Thread(target=_run_datamuse, daemon=True)
    t2 = threading.Thread(target=_run_lt, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=2.5)
    t2.join(timeout=3.5)

    # Fall back to local logic if APIs did not return results
    suggestions = sug_result if sug_result \
        else _suggest_small_words(last_word, sentence_ended_word=sentence_ended_word)

    grammar = lt_issues if lt_issues is not None and (lt_issues or sentence.strip()) \
        else _english_grammar_feedback(sentence)

    # Use LT's auto-corrected sentence when available, otherwise local fixer
    fixed_sentence = lt_fixed if lt_fixed and lt_fixed.strip() != sentence.strip() \
        else _fix_english_grammar(sentence)

    asl_conversion = _convert_sentence_to_asl(sentence)

    return jsonify({
        'grammar': grammar,
        'small_suggestions': suggestions,
        'asl_conversion': asl_conversion,
        'fixed_sentence': fixed_sentence,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print()
    print("  +--------------------------------------+")
    print("  |  SignBridge  -  ASL Web Translator   |")
    print("  +--------------------------------------+")
    print("  |  Open:  http://127.0.0.1:5000        |")
    print("  |  Stop:  Ctrl+C                       |")
    print("  +--------------------------------------+")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False,
            threaded=True)