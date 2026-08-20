import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import (DeepfakeDetector, GenderDetector,
                   EmotionDetector, SkinToneDetector, HairTextureDetector)
from dataset import (DeepfakeDataset, CelebATestDataset, EmotionDataset,
                     SkinToneDataset, HairTextureDataset, val_transform)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
os.makedirs('results', exist_ok=True)

print(f"Using: {DEVICE}")

def evaluate_model(model, loader, class_names, model_name, is_multi=False, task=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            images = images.to(DEVICE)
            outputs = model(images)

            if is_multi:
                preds = outputs[task].argmax(1).cpu().numpy()
                targets = labels[task].numpy()
            else:
                preds = outputs.argmax(1).cpu().numpy()
                targets = labels.numpy()

            all_preds.extend(preds)
            all_labels.extend(targets)

    acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Accuracy: {acc:.2f}%")
    print(classification_report(all_labels, all_preds,
                               target_names=class_names))

    return acc, cm, class_names

# Load all models
print("\nLoading all models...")

deepfake_model = DeepfakeDetector().to(DEVICE)
deepfake_model.load_state_dict(torch.load('models/deepfake_model.pth',
                               map_location=DEVICE, weights_only=False))

gender_model = GenderDetector().to(DEVICE)
gender_model.load_state_dict(torch.load('models/gender_model.pth',
                             map_location=DEVICE, weights_only=False))

emotion_model = EmotionDetector().to(DEVICE)
emotion_model.load_state_dict(torch.load('models/emotion_model.pth',
                              map_location=DEVICE, weights_only=False))

skin_model = SkinToneDetector().to(DEVICE)
skin_model.load_state_dict(torch.load('models/skin_model.pth',
                           map_location=DEVICE, weights_only=False))

hair_texture_model = HairTextureDetector().to(DEVICE)
hair_texture_model.load_state_dict(torch.load('models/hair_texture_model.pth',
                                   map_location=DEVICE, weights_only=False))

# Evaluate all models
print("\n1. Evaluating Deepfake Model...")
deepfake_dataset = DeepfakeDataset(
    'test/celebdf_real_faces',
    'test/celebdf_fake_faces',
    val_transform
)
loader = DataLoader(deepfake_dataset, batch_size=BATCH_SIZE, num_workers=0)
deepfake_acc, deepfake_cm, deepfake_classes = evaluate_model(
    deepfake_model, loader, ['Real', 'Fake'], 'Deepfake')

print("\n2. Evaluating Gender Model...")
gender_dataset = CelebATestDataset(
    'dataset/img_align_celeba/img_align_celeba',
    'dataset/img_align_celeba/list_attr_celeba.csv',
    'dataset/img_align_celeba/list_eval_partition.csv',
    val_transform
)
loader = DataLoader(gender_dataset, batch_size=BATCH_SIZE, num_workers=0)
gender_acc, gender_cm, gender_classes = evaluate_model(
    gender_model, loader, ['Female', 'Male'], 'Gender',
    is_multi=True, task='gender')

print("\n3. Evaluating Emotion Model...")
emotion_dataset = EmotionDataset('test/emotion_test', val_transform)
loader = DataLoader(emotion_dataset, batch_size=BATCH_SIZE, num_workers=0)
emotion_acc, emotion_cm, emotion_classes = evaluate_model(
    emotion_model, loader,
    ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    'Emotion')

print("\n4. Evaluating Skin Tone Model...")
skin_dataset = SkinToneDataset('test/skin_test', val_transform)
loader = DataLoader(skin_dataset, batch_size=BATCH_SIZE, num_workers=0)
skin_acc, skin_cm, skin_classes = evaluate_model(
    skin_model, loader, ['Fair', 'Light', 'Dark'], 'Skin Tone')

print("\n5. Evaluating Hair Texture Model...")
hair_dataset = HairTextureDataset('test/hairtexture_test', val_transform)
loader = DataLoader(hair_dataset, batch_size=BATCH_SIZE, num_workers=0)
hair_acc, hair_cm, hair_classes = evaluate_model(
    hair_texture_model, loader,
    ['Straight', 'Wavy', 'Curly', 'Kinky', 'Dreadlocks'],
    'Hair Texture')

# Overall Accuracy
overall_acc = (deepfake_acc + gender_acc + emotion_acc +
               skin_acc + hair_acc) / 5

print(f"\n{'='*60}")
print(f"OVERALL SYSTEM ACCURACY: {overall_acc:.2f}%")
print(f"{'='*60}")
print(f"Deepfake Detection : {deepfake_acc:.2f}%")
print(f"Gender Detection   : {gender_acc:.2f}%")
print(f"Emotion Detection  : {emotion_acc:.2f}%")
print(f"Skin Tone Detection: {skin_acc:.2f}%")
print(f"Hair Texture       : {hair_acc:.2f}%")
print(f"{'='*60}")

# Create complete image - NO BAR GRAPH
fig = plt.figure(figsize=(24, 16))

# Overall accuracy highlighted at top
fig.text(0.5, 0.98,
         'Complete Model Evaluation Results - Unseen Data',
         ha='center', fontsize=22, fontweight='bold')
fig.text(0.5, 0.94,
         f'⭐ OVERALL SYSTEM ACCURACY: {overall_acc:.2f}% ⭐',
         ha='center', fontsize=20, fontweight='bold',
         color='white',
         bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#27ae60', edgecolor='#1a8a4a', linewidth=3))

# 5 Confusion Matrices
def plot_cm(ax, cm, classes, title, acc):
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes,
                yticklabels=classes,
                cmap='Blues', ax=ax)
    ax.set_title(f'{title}\nAccuracy: {acc:.2f}%',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

ax1 = fig.add_subplot(2, 3, 1)
plot_cm(ax1, deepfake_cm, deepfake_classes,
        'Deepfake Detection', deepfake_acc)

ax2 = fig.add_subplot(2, 3, 2)
plot_cm(ax2, gender_cm, gender_classes,
        'Gender Detection', gender_acc)

ax3 = fig.add_subplot(2, 3, 3)
plot_cm(ax3, emotion_cm, emotion_classes,
        'Emotion Detection', emotion_acc)

ax4 = fig.add_subplot(2, 3, 4)
plot_cm(ax4, skin_cm, skin_classes,
        'Skin Tone Detection', skin_acc)

ax5 = fig.add_subplot(2, 3, 5)
plot_cm(ax5, hair_cm, hair_classes,
        'Hair Texture Detection', hair_acc)

# Summary Table
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
table_data = [
    ['Model', 'Test Dataset', 'Accuracy'],
    ['Deepfake', 'CelebDF v2', f'{deepfake_acc:.2f}%'],
    ['Gender', 'CelebA Test', f'{gender_acc:.2f}%'],
    ['Emotion', 'FER2013 Test', f'{emotion_acc:.2f}%'],
    ['Skin Tone', 'Skin Test', f'{skin_acc:.2f}%'],
    ['Hair Texture', 'Hair Test', f'{hair_acc:.2f}%'],
    ['⭐ OVERALL', 'All Tests', f'{overall_acc:.2f}%'],
]
table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 2.8)

# Highlight overall row
for j in range(3):
    table[6, j].set_facecolor('#27ae60')
    table[6, j].set_text_props(color='white', fontweight='bold')

# Highlight header row
for j in range(3):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax6.set_title('Summary Table', fontweight='bold', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('results/complete_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nComplete evaluation image saved!")

# Save text results
with open('results/evaluation_results.txt', 'w') as f:
    f.write("FINAL EVALUATION SUMMARY - UNSEEN DATA\n")
    f.write("="*60 + "\n")
    f.write(f"OVERALL SYSTEM ACCURACY    : {overall_acc:.2f}%\n")
    f.write("="*60 + "\n")
    f.write(f"Deepfake Detection (CelebDF v2) : {deepfake_acc:.2f}%\n")
    f.write(f"Gender Detection (CelebA Test)  : {gender_acc:.2f}%\n")
    f.write(f"Emotion Detection (Test)        : {emotion_acc:.2f}%\n")
    f.write(f"Skin Tone Detection (Test)      : {skin_acc:.2f}%\n")
    f.write(f"Hair Texture Detection (Test)   : {hair_acc:.2f}%\n")
    f.write("="*60 + "\n")

print("Results saved to results/evaluation_results.txt!")