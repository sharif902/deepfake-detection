import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ImageCompression(p=0.4),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.data = []
        self.transform = transform

        for img in os.listdir(real_dir):
            if img.endswith(('.jpg','.png','.jpeg')):
                self.data.append((os.path.join(real_dir, img), 0))

        for img in os.listdir(fake_dir):
            if img.endswith(('.jpg','.png','.jpeg')):
                self.data.append((os.path.join(fake_dir, img), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(attr_file)
        df = df.replace(-1, 0)

        self.images = df['image_id'].tolist()
        self.gender = df['Male'].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']

        labels = {
            'gender': torch.tensor(self.gender[idx], dtype=torch.long),
        }
        return image, labels


class CelebATestDataset(Dataset):
    def __init__(self, img_dir, attr_file, partition_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(attr_file)
        df = df.replace(-1, 0)

        part = pd.read_csv(partition_file)
        test_images = part[part['partition'] == 2]['image_id'].tolist()

        df = df[df['image_id'].isin(test_images)]

        self.images = df['image_id'].tolist()
        self.gender = df['Male'].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']

        labels = {
            'gender': torch.tensor(self.gender[idx], dtype=torch.long),
        }
        return image, labels


class EmotionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform

        emotion_labels = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6
        }

        for emotion, label in emotion_labels.items():
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img in os.listdir(emotion_dir):
                    if img.endswith(('.jpg','.png','.jpeg')):
                        self.data.append((os.path.join(emotion_dir, img), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


class SkinToneDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform

        skin_labels = {
            'fair': 0, 'light': 1, 'dark': 2
        }

        for skin, label in skin_labels.items():
            skin_dir = os.path.join(data_dir, skin)
            if os.path.exists(skin_dir):
                for img in os.listdir(skin_dir):
                    if img.endswith(('.jpg','.png','.jpeg')):
                        self.data.append((os.path.join(skin_dir, img), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


class HairTextureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform

        hair_labels = {
            'Straight': 0, 'Wavy': 1,
            'curly': 2, 'kinky': 3, 'dreadlocks': 4
        }

        for hair, label in hair_labels.items():
            hair_dir = os.path.join(data_dir, hair)
            if os.path.exists(hair_dir):
                for img in os.listdir(hair_dir):
                    if img.endswith(('.jpg','.png','.jpeg')):
                        self.data.append((os.path.join(hair_dir, img), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label
    
    
if __name__ == "__main__":

    print("Testing Deepfake Dataset...")
    deepfake = DeepfakeDataset(
        "dataset/real",
        "dataset/fake",
        transform=train_transform
    )
    print(f"Deepfake Images: {len(deepfake)}")

    print("\nTesting CelebA Dataset...")
    celeba = CelebADataset(
        "dataset/img_align_celeba",
        "dataset/list_attr_celeba.csv",
        transform=train_transform
    )
    print(f"CelebA Images: {len(celeba)}")

    print("\nTesting Emotion Dataset...")
    emotion = EmotionDataset(
        "dataset/emotions",
        transform=train_transform
    )
    print(f"Emotion Images: {len(emotion)}")

    print("\nTesting Skin Tone Dataset...")
    skin = SkinToneDataset(
        "dataset/skin_tone",
        transform=train_transform
    )
    print(f"Skin Tone Images: {len(skin)}")

    print("\nTesting Hair Texture Dataset...")
    hair = HairTextureDataset(
        "dataset/hair_texture",
        transform=train_transform
    )
    print(f"Hair Texture Images: {len(hair)}")

    print("\n✅ All datasets loaded successfully!")