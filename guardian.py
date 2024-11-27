'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import logging
from datetime import datetime
from pathlib import Path
import gc
from tqdm import tqdm
from model import AIImageDetector, AIImageDataset

class AIDetectorTrainer:
    def __init__(self, train_dir, val_dir, batch_size=32, num_workers=4):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.setup_logging()
            logging.info(f"Using device: {self.device}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

            self.train_dataset = AIImageDataset(root_dir=train_dir, transform=self.transform)
            self.val_dataset = AIImageDataset(root_dir=val_dir, transform=self.val_transform)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

            self.model = AIImageDetector().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.01
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=2,
                factor=0.1,
                verbose=True
            )

        except Exception as e:
            logging.error(f"Error in initialization: {str(e)}")
            raise

    def setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        handlers = [
            logging.FileHandler(log_dir / f'training_{timestamp}.log'),
            logging.StreamHandler()
        ]

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            try:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })

            except Exception as e:
                logging.error(f"Error in training batch: {str(e)}")
                continue

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validating'):
                try:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                except Exception as e:
                    logging.error(f"Error in validation batch: {str(e)}")
                    continue

        return val_loss / len(self.val_loader), 100 * correct / total

    def save_model(self, filename):
        try:
            save_dir = Path('models')
            save_dir.mkdir(exist_ok=True)

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }, save_dir / filename)

        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")

    def train(self, num_epochs=10):
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5

        for epoch in range(num_epochs):
            try:
                logging.info(f"\nEpoch {epoch+1}/{num_epochs}")

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()

                log_message = (
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n'
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                )

                logging.info(log_message)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    logging.info("Early stopping triggered")
                    break

                if (epoch + 1) % 5 == 0:
                    self.save_model(f'checkpoint_epoch_{epoch+1}.pth')

            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {str(e)}")
                self.save_model(f'emergency_checkpoint_epoch_{epoch+1}.pth')
                continue

def main():
    try:
        train_dir = '/Users/nithin/Documents/guardianAI/dataset/train'
        val_dir = '/Users/nithin/Documents/guardianAI/dataset/val'

        trainer = AIDetectorTrainer(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=32,
            num_workers=4
        )

        trainer.train(num_epochs=10)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import logging
from datetime import datetime
from src.model import AIImageDetector, AIImageDataset

class AIDetectorTrainer:
    def __init__(self, train_dir, val_dir, batch_size=32, num_workers=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = AIImageDataset(root_dir=train_dir, transform=self.transform)
        self.val_dataset = AIImageDataset(root_dir=val_dir, transform=self.transform)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # Initialize model and move to device
        self.model = AIImageDetector().to(self.device)

        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=2
        )

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Setup logging configuration
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            filename=f'logs/training_{timestamp}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return val_loss / len(self.val_loader), 100 * correct / total

    def train(self, num_epochs=10):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Log metrics
            log_message = (f'Epoch {epoch+1}/{num_epochs}\n'
                         f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n'
                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            print(log_message)
            logging.info(log_message)

            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                logging.info(f'Saved new best model with validation loss: {val_loss:.4f}')

    def save_model(self, filename):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join('models', filename))

def main():
    train_dir = '/Users/nithin/Documents/guardianAI/dataset/train'
    val_dir = '/Users/nithin/Documents/guardianAI/dataset/val'

    # Initialize trainer
    trainer = AIDetectorTrainer(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=32,
        num_workers=4
    )

    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main()