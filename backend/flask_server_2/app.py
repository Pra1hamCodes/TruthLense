import os
from flask import Flask, request, jsonify, send_from_directory
import torch
# Saturate CPU cores for faster inference on Windows.
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4)))
except Exception:
    pass
from vit_pytorch import ViT
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from datetime import datetime
import shutil

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure folders
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze-frames', methods=['POST'])
def analyze_frames_endpoint():
    try:
        data = request.get_json()
        if not data or 'frames_dir' not in data:
            return jsonify({'error': 'Missing frames_dir'}), 400

        frames_dir = data['frames_dir']
        frames_dir_path = os.path.join('frames', frames_dir)
        if not os.path.exists(frames_dir_path):
            return jsonify({'error': f'Frames directory not found: {frames_dir_path}'}), 404

        # List all image files in the directory
        frame_files = sorted([f for f in os.listdir(frames_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not frame_files:
            return jsonify({'error': 'No frames found in directory'}), 404

        frames_analysis = []
        total_confidence = 0
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir_path, frame_file)
            try:
                confidence, is_fake = analyze_frame(frame_path)
                total_confidence += confidence
                frames_analysis.append({
                    'frame': frame_file,
                    'frame_path': frame_path,
                    'confidence': confidence,
                    'is_fake': is_fake
                })
            except Exception as e:
                print(f"Error analyzing frame {frame_file}: {str(e)}")

        num_frames = len(frames_analysis)
        if num_frames > 0:
            average_confidence = total_confidence / num_frames
            fake_frames = sum(1 for frame in frames_analysis if frame['is_fake'])
            real_frames = num_frames - fake_frames
            is_fake = fake_frames > real_frames
            result = {
                'frames_analysis': frames_analysis,
                'confidence': average_confidence,
                'is_fake': is_fake,
                'total_frames': num_frames,
                'fake_frames_count': fake_frames,
                'real_frames_count': real_frames
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No frames were analyzed'}), 500
    except Exception as e:
        print(f"Error in /analyze-frames: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

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

        target_sd = model.state_dict()
        for key in ('cls_token', 'pos_embedding'):
            if key in state_dict and key in target_sd:
                src = state_dict[key]
                tgt_shape = target_sd[key].shape
                while src.dim() < len(tgt_shape):
                    src = src.unsqueeze(0)
                while src.dim() > len(tgt_shape) and src.shape[0] == 1:
                    src = src.squeeze(0)
                state_dict[key] = src

        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_frame(frame):
    """Preprocess video frame for model input"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return preprocess(pil_image)

MAX_FRAMES = 8


def extract_frames(video_path, frames_dir):
    """Extract up to MAX_FRAMES frames uniformly sampled from the video."""
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    vidObj = cv2.VideoCapture(video_path)
    total = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // MAX_FRAMES) if total > 0 else 20
    count = 0
    frames = []
    frames_dirname = os.path.basename(frames_dir)

    while True:
        success, image = vidObj.read()
        if not success:
            break
        if count % step == 0:
            if len(frames) >= MAX_FRAMES:
                break
            frame_path = os.path.join(frames_dir, f"frame{count}.jpg")
            cv2.imwrite(frame_path, image)
            frame_url = f'/uploads/frames/{frames_dirname}/{os.path.basename(frame_path)}'
            frames.append({
                'frame': os.path.basename(frame_path),
                'frame_path': frame_url
            })
        count += 1

    vidObj.release()
    return frames

def analyze_frame(frame_path, threshold=0.5):
    """Analyze a single frame using the ViT model"""
    global model
    try:
        if model is None:
            if not load_model():
                raise Exception("Model not loaded")

        # Read and preprocess frame
        frame = cv2.imread(frame_path)
        processed_frame = preprocess_frame(frame)

        # Get prediction
        with torch.no_grad():
            output = model(processed_frame.unsqueeze(0))
            confidence = float(torch.sigmoid(output).cpu().numpy()[0][0])
            is_fake = confidence < threshold

        return confidence, is_fake

    except Exception as e:
        print(f"Error analyzing frame {frame_path}: {str(e)}")
        raise


def analyze_frames_batch(frame_paths, threshold=0.5):
    """Single forward pass over all frames. Much faster than per-frame calls."""
    global model
    if model is None and not load_model():
        raise RuntimeError('Model not loaded')
    tensors = []
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            tensors.append(torch.zeros(3, 224, 224))
            continue
        tensors.append(preprocess_frame(frame))
    if not tensors:
        return []
    batch = torch.stack(tensors, dim=0)
    with torch.no_grad():
        output = model(batch)
        probs = torch.sigmoid(output).cpu().numpy().reshape(-1)
    return [(float(c), float(c) < threshold) for c in probs]

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Ensure model is loaded
        if model is None and not load_model():
            return jsonify({'error': 'Failed to load model'}), 500

        # Save uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        # Create frames directory
        frames_dirname = f"frames_{os.path.splitext(filename)[0]}"
        frames_dir = os.path.join(FRAMES_FOLDER, frames_dirname)

        # Extract and analyze frames
        frames = extract_frames(video_path, frames_dir)
        paths = [os.path.join(frames_dir, f['frame']) for f in frames]
        try:
            results = analyze_frames_batch(paths)
        except Exception as e:
            print('batch inference failed:', e)
            results = []
        frames_analysis = []
        total_confidence = 0
        for frame, result in zip(frames, results):
            confidence, is_fake = result
            total_confidence += confidence
            frames_analysis.append({
                'frame': frame['frame'],
                'frame_path': frame['frame_path'],
                'confidence': confidence,
                'is_fake': is_fake,
            })

        # Calculate overall results
        num_frames = len(frames_analysis)
        if num_frames > 0:
            average_confidence = total_confidence / num_frames
            fake_frames = sum(1 for frame in frames_analysis if frame['is_fake'])
            real_frames = num_frames - fake_frames
            is_fake = fake_frames > real_frames

            result = {
                'frames_analysis': frames_analysis,
                'confidence': average_confidence,
                'is_fake': is_fake,
                'total_frames': num_frames,
                'fake_frames_count': fake_frames,
                'real_frames_count': real_frames
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'No frames were analyzed'}), 500

    except Exception as e:
        print("Error during analysis:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/frames/<path:filename>')
def serve_frame(filename):
    """Serve frame images"""
    try:
        print(f"Serving frame: {filename}")
        frame_dir = os.path.dirname(filename)
        frame_name = os.path.basename(filename)
        frame_path = os.path.join(FRAMES_FOLDER, frame_dir)
        print(f"Full path: {os.path.join(frame_path, frame_name)}")
        return send_from_directory(frame_path, frame_name)
    except Exception as e:
        print(f"Error serving frame {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/extract-frames', methods=['POST'])
def extract_frames_endpoint():
    try:
        data = request.get_json()
        if not data or 'video_path' not in data or 'frames_dir' not in data:
            return jsonify({'error': 'Missing video_path or frames_dir'}), 400

        video_path = data['video_path']
        frames_dir = data['frames_dir']

        # Create frames directory
        frames_dir_path = os.path.join('frames', frames_dir)
        
        # Extract frames
        frames = extract_frames(video_path, frames_dir_path)

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
        # Default to 5001 to match backend FLASK_VIT_URL expectations.
        port = int(os.environ.get('PORT', 5001))
        debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    else:
        print("Failed to load model. Please ensure the model file exists.")