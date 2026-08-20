# Deepfake Detection System - EfficientNetV2

## Overview
A multi-task deepfake detection system using EfficientNetV2-S that detects:
- ✅ Real/Fake (Deepfake Detection) - 83.40%
- ✅ Gender (Male/Female) - 99.79%
- ✅ Emotion (7 types) - 69.31%
- ✅ Skin Tone (3 types) - 79.52%
- ✅ Hair Texture (5 types) - 98.48%
- ✅ Overall Accuracy - 86.10%

## How to Run

### Step 1 - Clone Project
git clone https://github.com/sharif902/deepfake-detection.git
cd deepfake-detection

### Step 2 - Install Libraries
pip install -r requirements.txt

### Step 3 - Run Detection
python detect_video.py

## Datasets Used
- FaceForensics++ (Deepfake detection)
- CelebA (Gender, Hair)
- FER Dataset (Emotions)
- Skin Tone Dataset
- Hair Texture Dataset

## Results
| Task | Accuracy |
|---|---|
| Deepfake Detection | 83.40% |
| Gender | 99.79% |
| Emotion | 69.31% |
| Skin Tone | 79.52% |
| Hair Texture | 98.48% |
| **Overall** | **86.10%** |

## Project Structure
- model.py - Model architecture
- train.py - Training code
- detect_video.py - Video detection
- dataset.py - Dataset loading
- models/ - Trained model files