'''
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import Dataset
from PIL import Image
import os
import logging
from typing import Tuple, List

class AIImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'ai_generated']
        self.images: List[str] = []
        self.labels: List[int] = []
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

        try:
            # Load real images
            real_dir = os.path.join(root_dir, 'real')
            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(self.valid_extensions):
                        self.images.append(os.path.join(real_dir, img_name))
                        self.labels.append(0)

            # Load AI-generated images
            ai_dir = os.path.join(root_dir, 'ai_generated')
            if os.path.exists(ai_dir):
                for img_name in os.listdir(ai_dir):
                    if img_name.lower().endswith(self.valid_extensions):
                        self.images.append(os.path.join(ai_dir, img_name))
                        self.labels.append(1)

            if not self.images:
                raise ValueError("No valid images found in the dataset directories")

        except Exception as e:
            logging.error(f"Error initializing dataset: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path = self.images[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except (IOError, OSError) as e:
                logging.error(f"Error loading image {img_path}: {str(e)}")
                idx = (idx + 1) % len(self.images)
                return self.__getitem__(idx)

            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise

class AIImageDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(AIImageDetector, self).__init__()

        try:
            self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze early layers
            for param in list(self.densenet.parameters())[:-12]:
                param.requires_grad = False

            # Enhanced classifier
            num_features = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

            # Initialize weights
            for m in self.densenet.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.densenet(x)
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        try:
            original_classifier = self.densenet.classifier
            self.densenet.classifier = nn.Identity()

            with torch.no_grad():
                features = self.densenet(x)

            self.densenet.classifier = original_classifier
            return features

        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            raise

'''
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import Dataset
from PIL import Image
import os
import logging
from typing import Tuple, List

class AIImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'ai_generated']
        self.images: List[str] = []
        self.labels: List[int] = []
        self.valid_extensions = ('.png', '.jpg', '.jpeg')

        try:
            # Load real images
            real_dir = os.path.join(root_dir, 'real')
            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(self.valid_extensions):
                        self.images.append(os.path.join(real_dir, img_name))
                        self.labels.append(0)

            # Load AI-generated images
            ai_dir = os.path.join(root_dir, 'ai_generated')
            if os.path.exists(ai_dir):
                for img_name in os.listdir(ai_dir):
                    if img_name.lower().endswith(self.valid_extensions):
                        self.images.append(os.path.join(ai_dir, img_name))
                        self.labels.append(1)

            if not self.images:
                raise ValueError("No valid images found in the dataset directories")

        except Exception as e:
            logging.error(f"Error initializing dataset: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path = self.images[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except (IOError, OSError) as e:
                logging.error(f"Error loading image {img_path}: {str(e)}")
                idx = (idx + 1) % len(self.images)
                return self.__getitem__(idx)

            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise

class AIImageDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(AIImageDetector, self).__init__()

        try:
            self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # Freeze early layers
            for param in list(self.densenet.parameters())[:-12]:
                param.requires_grad = False

            # Enhanced classifier
            num_features = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, num_classes)
            )

            # Initialize weights
            for m in self.densenet.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.densenet(x)
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        try:
            original_classifier = self.densenet.classifier
            self.densenet.classifier = nn.Identity()

            with torch.no_grad():
                features = self.densenet(x)

            self.densenet.classifier = original_classifier
            return features

        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            raise
