from flask import Flask, request, jsonify, send_from_directory, abort
import torch
from vit_pytorch import ViT
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)

# Narrow CORS to known frontend/backend origins (override via FLASK_CORS_ORIGINS env).
_default_origins = 'http://localhost:5173,http://localhost:3000'
_allowed_origins = [o.strip() for o in os.environ.get('FLASK_CORS_ORIGINS', _default_origins).split(',') if o.strip()]
CORS(app, resources={
    r"/*": {
        "origins": _allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure folders
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

FRAMES_ROOT = os.path.abspath(FRAMES_FOLDER)
UPLOAD_ROOT = os.path.abspath(UPLOAD_FOLDER)

# Global variable for model
model = None


def load_model():
    """Load the ViT model"""
    global model
    try:
        model = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        weights_path = os.path.join(MODEL_FOLDER, "as_model_0.837.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model_state_dict = model.state_dict()

        for key, value in list(state_dict.items()):
            if key in model_state_dict and value.shape != model_state_dict[key].shape:
                while value.dim() > 0 and value.shape[0] == 1 and value.squeeze(0).shape == model_state_dict[key].shape:
                    value = value.squeeze(0)
                state_dict[key] = value

        model.load_state_dict(state_dict)
        model.eval()
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
    if not subdir:
        return None
    root_abs = os.path.realpath(os.path.abspath(root))
    candidate = os.path.realpath(os.path.abspath(os.path.join(root, subdir)))
    if candidate == root_abs:
        return None
    if not candidate.startswith(root_abs + os.sep):
        return None
    return candidate


def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return preprocess(pil_image)


def extract_frames(video_path, frames_dir):
    """Extract frames from video"""
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
            success, image = vidObj.read()
            if not success:
                break

            frames_num = 20  # Save every 20th frame
            if count % frames_num == 0:
                frame_path = os.path.join(frames_dir, f"frame{count}.jpg")
                cv2.imwrite(frame_path, image)
                frame_url = f'/uploads/frames/{frames_dirname}/{os.path.basename(frame_path)}'
                frames.append({
                    'frame': os.path.basename(frame_path),
                    'frame_path': frame_url
                })
                print(f"Saved frame: {frame_path}")
                print(f"Frame URL path: {frame_url}")
            count += 1

        return frames
    finally:
        vidObj.release()


def analyze_frame(frame_path, threshold=0.5):
    """Analyze a single frame using the ViT model"""
    if model is None:
        raise RuntimeError("Model not loaded")

    try:
        frame = cv2.imread(frame_path)
        processed_frame = preprocess_frame(frame)

        with torch.no_grad():
            output = model(processed_frame.unsqueeze(0))
            confidence = float(torch.sigmoid(output).cpu().numpy()[0][0])

        if np.isnan(confidence):
            raise ValueError("Model returned NaN confidence")

        is_fake = confidence < threshold
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

    unique_stem = f"{uuid.uuid4().hex}_{os.path.splitext(filename)[0]}"
    unique_filename = f"{unique_stem}{ext}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    frames_dirname = f"frames_{unique_stem}"
    frames_dir = os.path.join(FRAMES_FOLDER, frames_dirname)

    try:
        video_file.save(video_path)

        frames = extract_frames(video_path, frames_dir)
        frames_analysis = []
        total_confidence = 0

        for frame in frames:
            frame_path = os.path.join(frames_dir, frame['frame'])
            confidence, is_fake = analyze_frame(frame_path)
            total_confidence += confidence

            frames_analysis.append({
                'frame': frame['frame'],
                'frame_path': frame['frame_path'],
                'confidence': confidence,
                'is_fake': is_fake
            })

        num_frames = len(frames_analysis)
        if num_frames == 0:
            return jsonify({'error': 'No frames were analyzed'}), 500

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
        return jsonify(result)

    except Exception as e:
        print("Error during analysis:", str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as cleanup_error:
                print(f"Failed to remove uploaded video: {cleanup_error}")


@app.route('/uploads/frames/<path:filename>')
def serve_frame(filename):
    """Serve frame images"""
    parts = filename.replace('\\', '/').split('/')
    if len(parts) != 2:
        abort(404)
    safe_subdir = _safe_subdir_name(parts[0])
    safe_file = secure_filename(parts[1])
    if not safe_subdir or not safe_file:
        abort(404)

    frame_dir = _resolve_within(FRAMES_ROOT, safe_subdir)
    if not frame_dir or not os.path.isdir(frame_dir):
        abort(404)

    return send_from_directory(frame_dir, safe_file)


@app.route('/extract-frames', methods=['POST'])
def extract_frames_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        raw_video = data.get('video_path')
        raw_frames_dir = data.get('frames_dir')
        if not raw_video or not raw_frames_dir:
            return jsonify({'error': 'Missing video_path or frames_dir'}), 400

        # Only accept a filename inside our uploads dir, not an arbitrary path.
        safe_video_name = secure_filename(os.path.basename(str(raw_video)))
        if not safe_video_name:
            return jsonify({'error': 'Invalid video_path'}), 400
        resolved_video = _resolve_within(UPLOAD_ROOT, safe_video_name)
        if not resolved_video or not os.path.isfile(resolved_video):
            return jsonify({'error': 'Video not found in uploads'}), 404

        safe_dir_name = _safe_subdir_name(raw_frames_dir)
        if not safe_dir_name:
            return jsonify({'error': 'Invalid frames_dir'}), 400
        resolved_frames_dir = _resolve_within(FRAMES_ROOT, safe_dir_name)
        if not resolved_frames_dir:
            return jsonify({'error': 'Invalid frames_dir'}), 400

        frames = extract_frames(resolved_video, resolved_frames_dir)

        return jsonify({
            'success': True,
            'message': 'Frames extracted successfully',
            'frames_count': len(frames)
        })

    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    if load_model():
        print("Starting server with ViT model loaded")
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    else:
        print("Failed to load model. Please ensure the model file exists.")
