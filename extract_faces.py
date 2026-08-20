import cv2
import os
import glob
from mtcnn import MTCNN
from tqdm import tqdm

detector = MTCNN()

def extract_faces(video_path, save_folder, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    os.makedirs(save_folder, exist_ok=True)

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 5 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
        else:
            face = cv2.resize(frame, (224, 224))

        filename = f"{save_folder}/{os.path.basename(video_path)}_{saved}.jpg"
        cv2.imwrite(filename, face)
        saved += 1

    cap.release()

BASE = "dataset/FaceForensics++_C23"
REAL_OUT = "dataset/real_faces"
FAKE_OUT = "dataset/fake_faces"

os.makedirs(REAL_OUT, exist_ok=True)
os.makedirs(FAKE_OUT, exist_ok=True)

# Extract real videos
print("Extracting real faces...")
real_videos = glob.glob(f"{BASE}/original/*.mp4")
print(f"Found {len(real_videos)} real videos!")
for video in tqdm(real_videos):
    extract_faces(video, REAL_OUT)

# Extract fake videos
print("Extracting fake faces...")
fake_folders = ['Deepfakes', 'Face2Face', 'FaceShifter', 
                'FaceSwap', 'NeuralTextures', 'DeepFakeDetection']
for folder in fake_folders:
    videos = glob.glob(f"{BASE}/{folder}/*.mp4")
    print(f"Found {len(videos)} videos in {folder}!")
    for video in tqdm(videos):
        extract_faces(video, FAKE_OUT)

print("Done! All faces extracted!")