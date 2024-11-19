import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import DenseNet121_Weights
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
        ai_dir = os.path.join(root_dir, 'ai_generated')

        # Load real images
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)

        # Load AI-generated images
        for img_name in os.listdir(ai_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(ai_dir, img_name))
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, SyntaxError, UnidentifiedImageError):
            print(f"Skipping invalid image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))  # Skip invalid image and return the next

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class AIImageDetector(nn.Module):
    def __init__(self):
        super(AIImageDetector, self).__init__()
        # Use the latest pretrained weights
        self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

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

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Dataset
train_dataset = AIImageDataset(root_dir='X:/Data/train', transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Model and Loss Function
model = AIImageDetector()
criterion = nn.CrossEntropyLoss()

# Example Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
