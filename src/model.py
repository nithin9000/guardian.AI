import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image
import os

class AIImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'ai_generated']

        self.images = []
        self.labels = []

        real_dir = os.path.join(root_dir, 'real')
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)

        ai_dir = os.path.join(root_dir, 'ai_generated')
        for img_name in os.listdir(ai_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(ai_dir, img_name))
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class AIImageDetector(nn.Module):
    def __init__(self):
        super(AIImageDetector, self).__init__()
        self.densenet = models.densenet121(pretrained=True)

        # Freeze early layers
        for param in list(self.densenet.parameters())[:-12]:
            param.requires_grad = False

        # Modify the classifier for binary classification
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.densenet(x)