import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


class GenderDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.gender = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return {'gender': self.gender(features)}


class EmotionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


class SkinToneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


class HairTextureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
if __name__ == '__main__':
    print("Testing EfficientNetV2 Models...")     
    x = torch.randn(1, 3, 224, 224)

    deepfake = DeepfakeDetector()
    print(f"DeepfakeDetector: {deepfake(x).shape}")

    gender = GenderDetector()
    print(f"GenderDetector: {gender(x)['gender'].shape}")

    emotion = EmotionDetector()
    print(f"EmotionDetector: {emotion(x).shape}")

    skin = SkinToneDetector()
    print(f"SkinToneDetector: {skin(x).shape}")

    hair = HairTextureDetector()
    print(f"HairTextureDetector: {hair(x).shape}")

    print("\nAll EfficientNetV2 models created successfully!")