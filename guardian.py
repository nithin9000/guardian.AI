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
    train_dir = r'X:\Data\train'
    val_dir = r'X:\Data\val'

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