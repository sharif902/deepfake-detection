import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results', exist_ok=True)

# =====================
# Graph 1 — Deepfake Detection
# =====================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [83.40, 89, 93, 91]
colors = ['#1f77b4', '#2ca02c', '#d62728', '#bcbd22']

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Deepfake Detection Test Metrics\n(EfficientNetV2 - CelebDF v2 Test Set)',
          fontsize=13, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/deepfake_metrics.png', dpi=150)
plt.close()
plt.show()

# =====================
# Graph 2 — Gender Detection
# =====================
values = [99.79, 100, 100, 100]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Gender Detection Test Metrics\n(EfficientNetV2 - CelebA Test Set)',
          fontsize=13, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(90, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/gender_metrics.png', dpi=150)
plt.close()
plt.show()

# =====================
# Graph 3 — Emotion Detection
# =====================
values = [69.31, 68, 69, 68]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Emotion Detection Test Metrics\n(EfficientNetV2 - FER2013 Test Set)',
          fontsize=13, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/emotion_metrics.png', dpi=150)
plt.close()
plt.show()

# =====================
# Graph 4 — Skin Tone Detection
# =====================
values = [79.52, 80, 80, 80]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Skin Tone Detection Test Metrics\n(EfficientNetV2 - Skin Test Set)',
          fontsize=13, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/skintone_metrics.png', dpi=150)
plt.close()
plt.show()

# =====================
# Graph 5 — Hair Texture Detection
# =====================
values = [98.48, 99, 98, 98]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, width=0.5)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Hair Texture Detection Test Metrics\n(EfficientNetV2 - Hair Test Set)',
          fontsize=13, fontweight='bold')
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(90, 105)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/hair_metrics.png', dpi=150)
plt.close()
plt.show()

# =====================
# Graph 6 — Overall System
# =====================
models = ['Deepfake\nDetection', 'Gender\nDetection',
          'Emotion\nDetection', 'Skin Tone\nDetection',
          'Hair Texture\nDetection', 'OVERALL']
overall = [83.40, 99.79, 69.31, 79.52, 98.48, 86.10]
colors2 = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728', '#bcbd22', '#17becf']

plt.figure(figsize=(12, 6))
bars = plt.bar(models, overall, color=colors2, width=0.5)
for bar, val in zip(bars, overall):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Overall System Performance\n(EfficientNetV2 Multi-Task Detection System)',
          fontsize=13, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/overall_metrics.png', dpi=150)
plt.show()

print("All 6 graphs saved to results/ folder!")
print("1. deepfake_metrics.png")
print("2. gender_metrics.png")
print("3. emotion_metrics.png")
print("4. skintone_metrics.png")
print("5. hair_metrics.png")
print("6. overall_metrics.png")