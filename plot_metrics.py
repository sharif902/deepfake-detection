import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.makedirs('results/metric_plots', exist_ok=True)

# =====================
# Function to plot square metric matrix
# =====================
def plot_metric_matrix(model_name, class_names, cm, filename, subtitle=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax, linewidths=0.5,
                annot_kws={"size": 13, "weight": "bold"})
    
    ax.set_title(f'{model_name}\n{subtitle}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plt.show()
    print(f"Saved: {filename}")

# =====================
# Function to plot metrics summary box
# =====================
def plot_metrics_summary(model_name, metrics_dict, filename, subtitle=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Create table data
    table_data = [[f'{v:.2f}%'] for v in values]
    
    colors_list = []
    for v in values:
        if v >= 90:
            colors_list.append(['#27AE60'])
        elif v >= 75:
            colors_list.append(['#F39C12'])
        else:
            colors_list.append(['#E74C3C'])
    
    table = ax.table(
        cellText=table_data,
        rowLabels=metrics,
        colLabels=['Score'],
        cellLoc='center',
        rowLoc='center',
        loc='center',
        cellColours=colors_list
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(2.5, 3.0)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        if col == -1:
            cell.set_facecolor('#2980B9')
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    ax.set_title(f'{model_name}\n{subtitle}',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    green = mpatches.Patch(color='#27AE60', label='≥ 90% (Excellent)')
    orange = mpatches.Patch(color='#F39C12', label='75-89% (Good)')
    red = mpatches.Patch(color='#E74C3C', label='< 75% (Moderate)')
    ax.legend(handles=[green, orange, red],
             loc='lower center',
             bbox_to_anchor=(0.5, -0.05),
             ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plt.show()
    print(f"Saved: {filename}")

# =====================
# 1. Deepfake Detection
# =====================
cm_deepfake = np.array([
    [4304, 13476],
    [8197, 104583]
])
plot_metric_matrix(
    'Deepfake Detection — Confusion Matrix',
    ['Real', 'Fake'],
    cm_deepfake,
    'results/metric_plots/1_deepfake_confusion.png',
    subtitle='EfficientNetV2-S | CelebDF v2 Test Set | 130,560 Images'
)

plot_metrics_summary(
    'Deepfake Detection — Evaluation Metrics',
    {
        'Accuracy': 83.40,
        'Precision (Fake)': 89.00,
        'Recall (Fake)': 93.00,
        'F1-Score (Fake)': 91.00,
        'Precision (Real)': 34.00,
        'Recall (Real)': 24.00,
    },
    'results/metric_plots/2_deepfake_metrics.png',
    subtitle='EfficientNetV2-S | CelebDF v2 Test Set'
)

# =====================
# 2. Gender Detection
# =====================
cm_gender = np.array([
    [12230, 17],
    [24, 7691]
])
plot_metric_matrix(
    'Gender Detection — Confusion Matrix',
    ['Female', 'Male'],
    cm_gender,
    'results/metric_plots/3_gender_confusion.png',
    subtitle='EfficientNetV2-S | CelebA Test Set | 19,962 Images'
)

plot_metrics_summary(
    'Gender Detection — Evaluation Metrics',
    {
        'Accuracy': 99.79,
        'Precision (Female)': 100.00,
        'Recall (Female)': 100.00,
        'F1-Score (Female)': 100.00,
        'Precision (Male)': 100.00,
        'Recall (Male)': 100.00,
    },
    'results/metric_plots/4_gender_metrics.png',
    subtitle='EfficientNetV2-S | CelebA Test Set'
)

# =====================
# 3. Emotion Detection
# =====================
cm_emotion = np.array([
    [635, 17, 107, 26, 94, 16, 63],
    [22, 78, 2, 4, 1, 1, 3],
    [126, 6, 550, 15, 164, 77, 86],
    [38, 1, 23, 1544, 32, 36, 100],
    [139, 8, 157, 44, 659, 20, 220],
    [22, 1, 67, 24, 9, 694, 14],
    [98, 0, 73, 73, 159, 15, 815],
])
plot_metric_matrix(
    'Emotion Detection — Confusion Matrix',
    ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
    cm_emotion,
    'results/metric_plots/5_emotion_confusion.png',
    subtitle='EfficientNetV2-S | FER2013 Test Set | 7,178 Images'
)

plot_metrics_summary(
    'Emotion Detection — Evaluation Metrics',
    {
        'Accuracy': 69.31,
        'Precision (Avg)': 68.00,
        'Recall (Avg)': 69.00,
        'F1-Score (Avg)': 68.00,
        'Happy F1': 88.00,
        'Surprise F1': 82.00,
    },
    'results/metric_plots/6_emotion_metrics.png',
    subtitle='EfficientNetV2-S | FER2013 Test Set'
)

# =====================
# 4. Skin Tone Detection
# =====================
cm_skin = np.array([
    [51, 19, 0],
    [20, 48, 2],
    [0, 2, 68]
])
plot_metric_matrix(
    'Skin Tone Detection — Confusion Matrix',
    ['Fair', 'Light', 'Dark'],
    cm_skin,
    'results/metric_plots/7_skin_confusion.png',
    subtitle='EfficientNetV2-S | Skin Test Set | 210 Images'
)

plot_metrics_summary(
    'Skin Tone Detection — Evaluation Metrics',
    {
        'Accuracy': 79.52,
        'Precision (Avg)': 80.00,
        'Recall (Avg)': 80.00,
        'F1-Score (Avg)': 80.00,
        'Dark Precision': 97.00,
        'Dark Recall': 97.00,
    },
    'results/metric_plots/8_skin_metrics.png',
    subtitle='EfficientNetV2-S | Skin Test Set'
)

# =====================
# 5. Hair Texture Detection
# =====================
cm_hair = np.array([
    [96, 0, 0, 0, 0],
    [0, 62, 4, 0, 0],
    [0, 0, 102, 0, 0],
    [0, 0, 0, 42, 1],
    [0, 0, 1, 0, 87]
])
plot_metric_matrix(
    'Hair Texture Detection — Confusion Matrix',
    ['Straight', 'Wavy', 'Curly', 'Kinky', 'Dreadlocks'],
    cm_hair,
    'results/metric_plots/9_hair_confusion.png',
    subtitle='EfficientNetV2-S | Hair Test Set | 395 Images'
)

plot_metrics_summary(
    'Hair Texture Detection — Evaluation Metrics',
    {
        'Accuracy': 98.48,
        'Precision (Avg)': 99.00,
        'Recall (Avg)': 98.00,
        'F1-Score (Avg)': 98.00,
        'Straight F1': 100.00,
        'Curly F1': 98.00,
    },
    'results/metric_plots/10_hair_metrics.png',
    subtitle='EfficientNetV2-S | Hair Test Set'
)

# =====================
# 6. Overall System Summary
# =====================
def plot_overall_summary(filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table_data = [
        ['Deepfake Detection', 'CelebDF v2',    '130,560', '83.40%', '89%', '93%', '91%'],
        ['Gender Detection',   'CelebA Test',   '19,962',  '99.79%', '100%','100%','100%'],
        ['Emotion Detection',  'FER2013 Test',  '7,178',   '69.31%', '68%', '69%', '68%'],
        ['Skin Tone Detection','Skin Test Set', '210',     '79.52%', '80%', '80%', '80%'],
        ['Hair Texture',       'Hair Test Set', '395',     '98.48%', '99%', '98%', '98%'],
        ['OVERALL AVERAGE',    'All Test Sets', '158,305', '86.10%', '—',   '—',   '—'],
    ]

    col_labels = ['Model', 'Test Dataset', 'Test Images',
                  'Accuracy', 'Precision', 'Recall', 'F1-Score']

    row_colors = [
        ['#D6EAF8']*7,
        ['#D5F5E3']*7,
        ['#FDEBD0']*7,
        ['#FADBD8']*7,
        ['#E8DAEF']*7,
        ['#1A5276']*7,
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=row_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    # Style header
    for col in range(len(col_labels)):
        table[0, col].set_facecolor('#2C3E50')
        table[0, col].set_text_props(color='white', fontweight='bold')

    # Style overall row
    for col in range(len(col_labels)):
        table[6, col].set_text_props(color='white', fontweight='bold')

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(2)

    ax.set_title('Overall System Evaluation — Multi-Task EfficientNetV2-S\nUnseen Test Data Performance Summary',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plt.show()
    print(f"Saved: {filename}")

plot_overall_summary('results/metric_plots/11_overall_summary.png')
from IPython.display import display, Image
import os

folder = 'results/metric_plots'

for file in sorted(os.listdir(folder)):
    if file.endswith('.png'):
        print(file)
        display(Image(filename=os.path.join(folder, file)))

print("\n✅ All 11 metric images saved to results/metric_plots/")

print("\n✅ All 11 metric images saved to results/metric_plots/")
print("1.  deepfake_confusion.png")
print("2.  deepfake_metrics.png")
print("3.  gender_confusion.png")
print("4.  gender_metrics.png")
print("5.  emotion_confusion.png")
print("6.  emotion_metrics.png")
print("7.  skin_confusion.png")
print("8.  skin_metrics.png")
print("9.  hair_confusion.png")
print("10. hair_metrics.png")
print("11. overall_summary.png")