from flask import Flask, request, jsonify, send_from_directory, abort
import cv2
import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)

# Narrow CORS to known frontend/backend origins (override via FLASK_CORS_ORIGINS env).
_default_origins = 'http://localhost:5173,http://localhost:3000'
_allowed_origins = [o.strip() for o in os.environ.get('FLASK_CORS_ORIGINS', _default_origins).split(',') if o.strip()]
CORS(app, resources={r"/*": {"origins": _allowed_origins}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 100 MB upload ceiling matches the Express backend.
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

FRAMES_ROOT = os.path.abspath(FRAMES_FOLDER)
PEER_FRAMES_ROOT = os.path.abspath(os.path.join('..', 'flask_server_2', 'frames'))

# Global variable for model
model = None


def load_model():
    """Load the deepfake detection model"""
    global model
    try:
        model_path = os.path.join(MODEL_FOLDER, 'deepfake_detection_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
}


def _safe_subdir_name(name):
    """Return the basename of *name* only if it is a safe directory name."""
    if not name:
        return None
    cleaned = secure_filename(name)
    if not cleaned or cleaned in ('.', '..'):
        return None
    stem = cleaned.split('.', 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        return None
    return cleaned


def _resolve_within(root, subdir):
    """Join *subdir* onto *root* and confirm the result stays under *root*.

    Uses realpath so symlinks cannot escape the root.
    """
    if not subdir:
        return None
    root_abs = os.path.realpath(os.path.abspath(root))
    candidate = os.path.realpath(os.path.abspath(os.path.join(root, subdir)))
    if candidate == root_abs:
        return None
    if not candidate.startswith(root_abs + os.sep):
        return None
    return candidate


def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        print(f"Created frames directory: {frames_dir}")

    vidObj = cv2.VideoCapture(video_path)
    try:
        count = 0
        success = 1
        frames = []

        frames_dirname = os.path.basename(frames_dir)

        while success:
            # Local name `frame` to avoid shadowing the imported keras `image` module.
            success, frame = vidObj.read()
            if not success:
                break

            frames_num = 20  # Save every 20th frame
            if count % frames_num == 0:
                frame_path = os.path.join(frames_dir, f"frame{count}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_url = f'/uploads/frames/{frames_dirname}/{os.path.basename(frame_path)}'
                frames.append({
                    'frame': os.path.basename(frame_path),
                    'frame_path': frame_url
                })
                print(f"Saved frame: {frame_path}")
                print(f"Frame URL path: {frame_url}")
            count += 1

        print(f"Total frames extracted: {len(frames)}")
        return frames
    finally:
        vidObj.release()


def analyze_frame(frame_path):
    """Analyze a single frame using the model"""
    if model is None:
        raise RuntimeError("Model not loaded")

    try:
        # Prepare image for model
        img = image.load_img(frame_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Get prediction
        confidence = float(model.predict(img_array, verbose=0)[0][0])

        # Validate confidence value
        if np.isnan(confidence):
            raise ValueError("Model returned NaN confidence")

        # If confidence is high (>0.5), it's real; if low (<0.5), it's fake
        is_fake = confidence < 0.5
        print(f"Analyzed {frame_path}: confidence = {confidence}, is_fake = {is_fake}")
        return confidence, is_fake
    except Exception as e:
        print(f"Error analyzing frame {frame_path}: {str(e)}")
        raise


@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(video_file.filename)
    if not filename:
        return jsonify({'error': 'Invalid filename'}), 400
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    # Unique names so concurrent uploads don't collide or clobber frames.
    unique_stem = f"{uuid.uuid4().hex}_{os.path.splitext(filename)[0]}"
    unique_filename = f"{unique_stem}{ext}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    frames_dirname = f"frames_{unique_stem}"
    frames_dir = os.path.join(FRAMES_FOLDER, frames_dirname)

    try:
        video_file.save(video_path)
        print(f"Saved video to: {video_path}")
        print(f"Processing frames in: {frames_dir}")

        frames = extract_frames(video_path, frames_dir)

        # Analyze frames
        total_confidence = 0
        frames_analysis = []
        analysis_errors = 0

        for frame in frames:
            try:
                frame_path = os.path.join(frames_dir, frame['frame'])
                confidence, is_fake = analyze_frame(frame_path)
                total_confidence += confidence

                frames_analysis.append({
                    'frame': frame['frame'],
                    'frame_path': frame['frame_path'],
                    'confidence': confidence,
                    'is_fake': is_fake
                })
            except Exception as e:
                print(f"Error analyzing frame: {str(e)}")
                analysis_errors += 1

        num_frames = len(frames_analysis)
        if num_frames == 0:
            return jsonify({'error': 'No frames were successfully analyzed'}), 500

        average_confidence = total_confidence / num_frames
        fake_frames = sum(1 for frame in frames_analysis if frame['is_fake'])
        real_frames = num_frames - fake_frames
        # Treat ties as fake — safer for a detection system to over-flag than under-flag.
        is_fake = fake_frames >= real_frames

        print(f"Total frames: {num_frames}, fake: {fake_frames}, real: {real_frames}, "
              f"avg confidence: {average_confidence}, is_fake: {is_fake}")

        result = {
            'frames_analysis': frames_analysis,
            'confidence': average_confidence,
            'is_fake': is_fake,
            'total_frames': num_frames,
            'analysis_errors': analysis_errors,
            'fake_frames_count': fake_frames,
            'real_frames_count': real_frames
        }

        return jsonify(result)

    except Exception as e:
        print("Error during analysis:", str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        # Best-effort cleanup of the uploaded video (frames kept for UI links).
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as cleanup_error:
                print(f"Failed to remove uploaded video: {cleanup_error}")


@app.route('/analyze-frames', methods=['POST'])
def analyze_frames():
    try:
        data = request.get_json(silent=True) or {}
        raw_dir = data.get('frames_dir')
        safe_name = _safe_subdir_name(raw_dir)
        if not safe_name:
            return jsonify({'error': 'Invalid frames_dir'}), 400

        full_frames_dir = _resolve_within(PEER_FRAMES_ROOT, safe_name)
        if not full_frames_dir or not os.path.isdir(full_frames_dir):
            return jsonify({'error': 'Frames directory not found'}), 404

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        frames_analysis = []
        total_confidence = 0

        frame_files = sorted([f for f in os.listdir(full_frames_dir) if f.endswith('.jpg')])
        print(f"Found {len(frame_files)} frame files")

        for frame_name in frame_files:
            frame_path = os.path.join(full_frames_dir, frame_name)
            confidence, is_fake = analyze_frame(frame_path)
            total_confidence += confidence

            frame_url = f'/uploads/frames/{safe_name}/{frame_name}'
            frames_analysis.append({
                'frame': frame_name,
                'frame_path': frame_url,
                'confidence': confidence,
                'is_fake': is_fake
            })

        num_frames = len(frames_analysis)
        if num_frames == 0:
            return jsonify({'error': 'Frames directory is empty'}), 400

        average_confidence = total_confidence / num_frames
        fake_frames = sum(1 for frame in frames_analysis if frame['is_fake'])
        real_frames = num_frames - fake_frames
        is_fake = fake_frames >= real_frames

        result = {
            'frames_analysis': frames_analysis,
            'confidence': average_confidence,
            'is_fake': is_fake,
            'total_frames': num_frames,
            'fake_frames_count': fake_frames,
            'real_frames_count': real_frames
        }

        print(f"Analysis complete: {num_frames} frames analyzed, is_fake={is_fake}")
        return jsonify(result)

    except Exception as e:
        print(f"Error during frame analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/frames/<path:filename>')
def serve_frame(filename):
    # Split into subdirectory + filename and confirm both components are safe.
    parts = filename.replace('\\', '/').split('/')
    if len(parts) != 2:
        abort(404)
    safe_subdir = _safe_subdir_name(parts[0])
    safe_file = secure_filename(parts[1])
    if not safe_subdir or not safe_file:
        abort(404)

    frame_dir = _resolve_within(PEER_FRAMES_ROOT, safe_subdir)
    if not frame_dir or not os.path.isdir(frame_dir):
        abort(404)

    return send_from_directory(frame_dir, safe_file)


# Load model when starting the server
if __name__ == '__main__':
    if load_model():
        print("Starting server with CNN+LSTM model loaded")
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    else:
        print("Failed to load model. Please ensure the model file exists.")
