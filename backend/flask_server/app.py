import os

# Force Keras to use the PyTorch backend. TensorFlow's native DLLs are blocked
# on this host by Windows Application Control, so we avoid importing TF entirely
# by relying on Keras 3 with the torch backend to load the legacy .h5 weights.
os.environ.setdefault('KERAS_BACKEND', 'torch')

import json
import shutil
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import h5py
import keras
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

model = None


def _prepare_h5_for_keras3(src_path):
    """Keras 3 can't read the legacy LSTM `time_major` kwarg — strip it from the
    model config and return a sanitized copy of the checkpoint."""
    dst_path = os.path.join(MODEL_FOLDER, '_clean_' + os.path.basename(src_path))
    if not os.path.exists(dst_path) or os.path.getmtime(dst_path) < os.path.getmtime(src_path):
        shutil.copy(src_path, dst_path)
        with h5py.File(dst_path, 'r+') as f:
            raw = f.attrs.get('model_config')
            if raw is None:
                return dst_path
            cfg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            for layer in cfg.get('config', {}).get('layers', []):
                if layer.get('class_name') == 'LSTM':
                    layer.get('config', {}).pop('time_major', None)
            f.attrs['model_config'] = json.dumps(cfg)
    return dst_path


def load_model():
    global model
    try:
        src_path = os.path.join(MODEL_FOLDER, 'deepfake_detection_model.h5')
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Model file not found at {src_path}")
        clean_path = _prepare_h5_for_keras3(src_path)
        print(f"Loading model from {clean_path}")
        model = keras.models.load_model(clean_path, compile=False)
        print("Model loaded successfully (Keras 3, torch backend)")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


MAX_FRAMES = 8


def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    vid = cv2.VideoCapture(video_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // MAX_FRAMES) if total > 0 else 20
    count = 0
    frames = []
    frames_dirname = os.path.basename(frames_dir)
    while True:
        ok, image = vid.read()
        if not ok:
            break
        if count % step == 0:
            if len(frames) >= MAX_FRAMES:
                break
            frame_path = os.path.join(frames_dir, f'frame{count}.jpg')
            cv2.imwrite(frame_path, image)
            frame_url = f'/uploads/frames/{frames_dirname}/{os.path.basename(frame_path)}'
            frames.append({'frame': os.path.basename(frame_path), 'frame_path': frame_url})
        count += 1
    vid.release()
    return frames


def analyze_frame(frame_path, threshold=0.5):
    global model
    if model is None and not load_model():
        raise RuntimeError('Model not loaded')
    img = cv2.imread(frame_path)
    if img is None:
        raise RuntimeError(f'cv2 could not read {frame_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    arr = img.astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    confidence = float(np.asarray(pred).reshape(-1)[0])
    is_fake = confidence < threshold
    return confidence, is_fake


def analyze_frames_batch(frame_paths, threshold=0.5):
    global model
    if model is None and not load_model():
        raise RuntimeError('Model not loaded')
    batch = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            batch.append(np.zeros((224, 224, 3), dtype='float32'))
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        batch.append(img.astype('float32') / 255.0)
    if not batch:
        return []
    arr = np.stack(batch, axis=0)
    preds = model.predict(arr, verbose=0, batch_size=len(batch))
    flat = np.asarray(preds).reshape(-1)
    return [(float(c), float(c) < threshold) for c in flat]


@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    vf = request.files['video']
    if vf.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        if model is None and not load_model():
            return jsonify({'error': 'Failed to load model'}), 500

        filename = secure_filename(vf.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        vf.save(video_path)

        frames_dirname = f'frames_{os.path.splitext(filename)[0]}'
        frames_dir = os.path.join(FRAMES_FOLDER, frames_dirname)
        frames = extract_frames(video_path, frames_dir)

        paths = [os.path.join(frames_dir, fr['frame']) for fr in frames]
        try:
            results = analyze_frames_batch(paths)
        except Exception as e:
            print('batch inference failed:', e)
            results = []
        frames_analysis = []
        total_conf = 0.0
        for fr, result in zip(frames, results):
            conf, is_fake = result
            total_conf += conf
            frames_analysis.append({
                'frame': fr['frame'],
                'frame_path': fr['frame_path'],
                'confidence': conf,
                'is_fake': is_fake,
            })

        n = len(frames_analysis)
        if n == 0:
            return jsonify({'error': 'No frames were analyzed'}), 500
        avg = total_conf / n
        fake = sum(1 for f in frames_analysis if f['is_fake'])
        return jsonify({
            'frames_analysis': frames_analysis,
            'confidence': avg,
            'is_fake': fake > (n - fake),
            'total_frames': n,
            'fake_frames_count': fake,
            'real_frames_count': n - fake,
        })
    except Exception as e:
        print('Error during analysis:', e)
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/frames/<path:filename>')
def serve_frame(filename):
    try:
        frame_dir = os.path.dirname(filename)
        frame_name = os.path.basename(filename)
        return send_from_directory(os.path.join(FRAMES_FOLDER, frame_dir), frame_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'ok': model is not None})


if __name__ == '__main__':
    if load_model():
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        print(f'Starting CNN-LSTM server on :{port}')
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    else:
        print('Failed to load model.')
