import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import cv2
import torch
from model import (DeepfakeDetector, GenderDetector,
                   EmotionDetector, SkinToneDetector, HairTextureDetector)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from transformers import pipeline
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

print("Loading trained models...")
deepfake_model = DeepfakeDetector().to(DEVICE)
deepfake_model.load_state_dict(torch.load("models/deepfake_model.pth",
                               map_location=DEVICE, weights_only=False))
deepfake_model.eval()

gender_model = GenderDetector().to(DEVICE)
gender_model.load_state_dict(torch.load("models/gender_model.pth",
                             map_location=DEVICE, weights_only=False))
gender_model.eval()

emotion_model = EmotionDetector().to(DEVICE)
emotion_model.load_state_dict(torch.load("models/emotion_model.pth",
                              map_location=DEVICE, weights_only=False))
emotion_model.eval()

skin_model = SkinToneDetector().to(DEVICE)
skin_model.load_state_dict(torch.load("models/skin_model.pth",
                           map_location=DEVICE, weights_only=False))
skin_model.eval()

hair_texture_model = HairTextureDetector().to(DEVICE)
hair_texture_model.load_state_dict(torch.load("models/hair_texture_model.pth",
                                   map_location=DEVICE, weights_only=False))
hair_texture_model.eval()

print("Loading HuggingFace models...")
age_model = pipeline("image-classification",
                     model="nateraw/vit-age-classifier",
                     device=-1)

# GradCAM setup
target_layer = [deepfake_model.backbone.blocks[-1]]
cam = GradCAM(model=deepfake_model, target_layers=target_layer)
os.makedirs('results', exist_ok=True)

detector = MTCNN()

GENDER_LABELS = ['Female', 'Male']
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SKIN_LABELS = ['Fair', 'Light', 'Dark']
HAIR_TEXTURE_LABELS = ['Straight', 'Wavy', 'Curly', 'Kinky', 'Dreadlocks']

def check_hair_visible(face_rgb):
    """Check if hair is visible in the image"""
    face_h, face_w = face_rgb.shape[:2]

    top_region = face_rgb[:int(face_h * 0.2), :]

    hsv = cv2.cvtColor(top_region, cv2.COLOR_RGB2HSV)

    skin_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 70]),
        np.array([20, 150, 255])
    )

    skin_ratio = np.sum(skin_mask > 0) / (
        top_region.shape[0] * top_region.shape[1]
    )

    return skin_ratio < 0.7

def predict_face(face_rgb):
    face_resized = cv2.resize(face_rgb, (224, 224))
    tensor = transform(image=face_resized)["image"].unsqueeze(0).to(DEVICE)
    pil_face = Image.fromarray(face_rgb)

    with torch.no_grad():
        # Deepfake
        fake_out = deepfake_model(tensor)
        fake_prob = torch.softmax(fake_out, dim=1)[0][1].item()
        is_fake = fake_prob > 0.5

        # Gender
        gen_out = gender_model(tensor)
        gender = GENDER_LABELS[gen_out['gender'].argmax(1).item()]

        # Emotion
        emo_out = emotion_model(tensor)
        emotion = EMOTION_LABELS[emo_out.argmax(1).item()]

        # Skin Tone
        skin_out = skin_model(tensor)
        skin_tone = SKIN_LABELS[skin_out.argmax(1).item()]

        # Hair Texture - check if hair visible first
                # Hair Texture - check if hair is visible first
        hair_visible = check_hair_visible(face_rgb)

        if not hair_visible:
            hair_texture = "Not Detectable"
        else:
            ht_out = hair_texture_model(tensor)
            ht_probs = torch.softmax(ht_out, dim=1)
            ht_confidence = ht_probs.max().item()

            if ht_confidence < 0.20:
                hair_texture = "Not Detectable"
            else:
                hair_texture = HAIR_TEXTURE_LABELS[ht_out.argmax(1).item()]
    # Age from HuggingFace
    age = age_model(pil_face)[0]['label']

    return {
        'fake_prob': fake_prob,
        'is_fake': is_fake,
        'gender': gender,
        'hair_texture': hair_texture,
        'emotion': emotion,
        'skin_tone': skin_tone,
        'age': age,
        'face_resized': face_resized
    }

def save_gradcam(face_resized, frame_count):
    face_float = face_resized / 255.0
    tensor_cam = transform(image=face_resized)["image"].unsqueeze(0).to(DEVICE)
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=tensor_cam, targets=targets)[0]
    visualization = show_cam_on_image(
        face_float.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    gradcam_path = f'results/gradcam_fake_frame_{frame_count}.jpg'
    cv2.imwrite(gradcam_path,
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return gradcam_path

def detect_image(image_path):
    print(f"\nAnalyzing image: {image_path}")
    print("="*60)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if not faces:
        print("No face detected in image!")
        return

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face = rgb[y:y+h, x:x+w]

    if face.size == 0:
        print("Face too small!")
        return

    result = predict_face(face)
    status = "FAKE ❌" if result['is_fake'] else "REAL ✅"

    print(f"\nIMAGE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Status      : {status} ({result['fake_prob']:.2%})")
    print(f"  Gender      : {result['gender']}")
    print(f"  Age         : {result['age']}")
    print(f"  Emotion     : {result['emotion']}")
    print(f"  Skin Tone   : {result['skin_tone']}")
    print(f"  Hair Texture: {result['hair_texture']}")

    if result['is_fake']:
        gradcam_path = save_gradcam(result['face_resized'], 'image')
        print(f"  GradCAM     : {gradcam_path}")
    else:
        print(f"  GradCAM     : Not generated (image is REAL)")

    print(f"{'='*60}")

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_count = 0
    fake_frames = []
    checked = 0

    print(f"\nAnalyzing video: {video_path}")
    print(f"Total Frames  : {total_frames}")
    print(f"FPS           : {fps:.2f}")
    print(f"Duration      : {int(duration//60):02d}:{int(duration%60):02d}")
    print(f"Frames to check: {total_frames//3}")
    print("="*60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 != 0:
            continue

        timestamp_sec = frame_count / fps
        minutes = int(timestamp_sec // 60)
        seconds = int(timestamp_sec % 60)
        milliseconds = int((timestamp_sec % 1) * 1000)
        timestamp_str = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]

            if face.size == 0:
                continue

            result = predict_face(face)
            status = "FAKE ❌" if result['is_fake'] else "REAL ✅"
            checked += 1

            print(f"\nFrame {frame_count} | Time: {timestamp_str}")
            print(f"  Status      : {status} ({result['fake_prob']:.2%})")
            print(f"  Gender      : {result['gender']}")
            print(f"  Age         : {result['age']}")
            print(f"  Emotion     : {result['emotion']}")
            print(f"  Skin Tone   : {result['skin_tone']}")
            print(f"  Hair Texture: {result['hair_texture']}")

            if result['is_fake']:
                gradcam_path = save_gradcam(result['face_resized'], frame_count)
                print(f"  GradCAM     : {gradcam_path}")

                fake_frames.append({
                    'frame': frame_count,
                    'timestamp': timestamp_str,
                    'probability': result['fake_prob'],
                    'gradcam': gradcam_path
                })

    cap.release()

    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)

    if fake_frames:
        print(f"\nVERDICT: FAKE VIDEO DETECTED! ❌")
        print(f"\nDeepfake detected at:")
        print(f"{'Frame':<10} {'Timestamp':<15} {'Probability':<15} {'GradCAM'}")
        print("-"*65)
        for f in fake_frames:
            prob_str = f"{f['probability']:.2%}"
            print(f"{f['frame']:<10} {f['timestamp']:<15} {prob_str:<15} {f['gradcam']}")
        print(f"\nTotal fake frames : {len(fake_frames)}")
        print(f"Total frames checked: {checked}")
        print(f"GradCAM heatmaps saved in results folder!")
        print(f"Red areas = WHERE deepfake manipulation detected!")
    else:
        print(f"\nVERDICT: REAL VIDEO ✅")
        print(f"Total frames checked: {checked}")
        print(f"No GradCAM generated - video is REAL!")

    print("="*60)

import yt_dlp
import requests
import tempfile
def download_video(url):
    print(f"Downloading video from: {url}")

    ydl_opts = {
        'format': 'bv*[ext=mp4]/bv*/b',
        'outtmpl': 'temp_downloaded.%(ext)s',
        'js_runtimes': {'deno': {}},
        'remote_components': ['ejs:github'],
        'noplaylist': True,
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

    print("Download complete!")
    return downloaded_file

def download_image(url):
    print(f"Downloading image from: {url}")
    response = requests.get(url, timeout=10)
    temp_path = 'temp_downloaded.jpg'
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    print("Download complete!")
    return temp_path

def is_url(path):
    return path.startswith('http://') or path.startswith('https://')

def is_image_url(url):
    return any(url.lower().endswith(ext) 
               for ext in ['.jpg', '.jpeg', '.png', '.webp'])

# =====================
# CHANGE INPUT HERE
# =====================
input_path = "555.jpeg"
# OR
# input_path = "https://example.com/image.jpg"
# OR
# input_path = "1.mp4"  # local file still works!

# Auto detect URL or local file
if is_url(input_path):
    if is_image_url(input_path):
        local_path = download_image(input_path)
        detect_image(local_path)
        os.remove(local_path)
    else:
        local_path = download_video(input_path)
        detect_video(local_path)
        os.remove(local_path)
else:
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        detect_video(input_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        detect_image(input_path)
    else:
        print("Unsupported file type!")