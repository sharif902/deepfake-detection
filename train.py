import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import (DeepfakeDetector, GenderDetector,
                   EmotionDetector, SkinToneDetector, HairTextureDetector)
from dataset import (DeepfakeDataset, CelebADataset, EmotionDataset,
                     SkinToneDataset, HairTextureDataset,
                     train_transform, val_transform)
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4

print(f"Using: {DEVICE}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

criterion = FocalLoss(gamma=2)

def train_model(model, train_loader, val_loader,
                model_name, is_multi=False, tasks=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS)
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            images, labels = batch
            images = images.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)

            if is_multi:
                loss = torch.tensor(0.0).to(DEVICE)
                for task in tasks:
                    targets = labels[task].to(DEVICE)
                    loss = loss + criterion(outputs[task], targets)
                first_task = tasks[0]
                correct += (outputs[first_task].argmax(1) ==
                           labels[first_task].to(DEVICE)).sum().item()
            else:
                targets = labels.to(DEVICE)
                loss = criterion(outputs, targets)
                correct += (outputs.argmax(1) == targets).sum().item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += images.size(0)

        train_acc = correct / total * 100

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(DEVICE)
                outputs = model(images)

                if is_multi:
                    first_task = tasks[0]
                    targets = labels[first_task].to(DEVICE)
                    preds = outputs[first_task].argmax(1)
                else:
                    targets = labels.to(DEVICE)
                    preds = outputs.argmax(1)

                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total * 100
        scheduler.step()

        print(f"{model_name} | Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {total_loss:.3f} | "
              f"Train: {train_acc:.2f}% | "
              f"Val: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                      f'models/{model_name}.pth')
            print(f"  -> Best model saved! ({best_acc:.2f}%)")

    print(f"\n{model_name} Training Complete! Best Val: {best_acc:.2f}%\n")

os.makedirs('models', exist_ok=True)

# 1. Deepfake Model
print("\n" + "="*50)
print("1. Training Deepfake Detector...")
print("="*50)
deepfake_dataset = DeepfakeDataset(
    'dataset/real_faces',
    'dataset/fake_faces',
    train_transform
)
train_size = int(0.8 * len(deepfake_dataset))
val_size = len(deepfake_dataset) - train_size
train_data, val_data = random_split(deepfake_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}")
deepfake_model = DeepfakeDetector().to(DEVICE)
train_model(deepfake_model, train_loader, val_loader, 'deepfake_model')

# 2. Gender Model
print("\n" + "="*50)
print("2. Training Gender Detector...")
print("="*50)
celeba_dataset = CelebADataset(
    'dataset/img_align_celeba/img_align_celeba',
    'dataset/img_align_celeba/list_attr_celeba.csv',
    train_transform
)
train_size = int(0.8 * len(celeba_dataset))
val_size = len(celeba_dataset) - train_size
train_data, val_data = random_split(celeba_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}")
gender_model = GenderDetector().to(DEVICE)
train_model(gender_model, train_loader, val_loader,
            'gender_model', is_multi=True, tasks=['gender'])

# 3. Emotion Model
print("\n" + "="*50)
print("3. Training Emotion Detector...")
print("="*50)
emotion_dataset = EmotionDataset('dataset/emotions', train_transform)
train_size = int(0.8 * len(emotion_dataset))
val_size = len(emotion_dataset) - train_size
train_data, val_data = random_split(emotion_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}")
emotion_model = EmotionDetector().to(DEVICE)
train_model(emotion_model, train_loader, val_loader, 'emotion_model')

# 4. Skin Tone Model
print("\n" + "="*50)
print("4. Training Skin Tone Detector...")
print("="*50)
skin_dataset = SkinToneDataset('dataset/skin', train_transform)
train_size = int(0.8 * len(skin_dataset))
val_size = len(skin_dataset) - train_size
train_data, val_data = random_split(skin_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}")
skin_model = SkinToneDetector().to(DEVICE)
train_model(skin_model, train_loader, val_loader, 'skin_model')

# 5. Hair Texture Model
print("\n" + "="*50)
print("5. Training Hair Texture Detector...")
print("="*50)
hair_dataset = HairTextureDataset('dataset/hairtexture', train_transform)
train_size = int(0.8 * len(hair_dataset))
val_size = len(hair_dataset) - train_size
train_data, val_data = random_split(hair_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
print(f"Train: {train_size} | Val: {val_size}")
hair_texture_model = HairTextureDetector().to(DEVICE)
train_model(hair_texture_model, train_loader, val_loader, 'hair_texture_model')

print("\n" + "="*50)
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("="*50)