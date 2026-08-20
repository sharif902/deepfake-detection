echo "# Deepfake Detection System - EfficientNetV2

## ⚠️ Requirements
- Python 3.12 (NOT 3.13 or 3.14)
- CUDA 12.1 (for GPU support)

## Step 1 - Check Python Version
python --version
# Must show Python 3.12.x

## Step 2 - Clone Project
git clone https://github.com/sharif902/deepfake-detection.git
cd deepfake-detection

## Step 3 - Install PyTorch (GPU) - Run This First!
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

## Step 4 - Install Other Libraries
pip install -r requirements.txt

## Step 5 - Run Detection
python detect_video.py

## Results
| Task | Accuracy |
|---|---|
| Deepfake Detection | 83.40% |
| Gender | 99.79% |
| Emotion | 69.31% |
| Skin Tone | 79.52% |
| Hair Texture | 98.48% |
| Overall | 86.10% |

## Datasets Used
- FaceForensics++ (Deepfake detection)
- CelebA (Gender, Hair)
- FER Dataset (Emotions)
- Skin Tone Dataset
- Hair Texture Dataset

## Project Structure
- model.py - Model architecture
- train.py - Training code
- detect_video.py - Video detection
- dataset.py - Dataset loading
- models/ - Trained model files" > README.md

