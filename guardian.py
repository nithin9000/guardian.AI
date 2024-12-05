import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.efficientNet import EfficientNetB7
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Check for MPS (Apple Silicon) availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def setup_data_loaders(train_dir, val_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Reduced from 600x600
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((400, 400)),  # Reduced from 600x600
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader

def train(config):
    writer = SummaryWriter(f'runs/face_detection_{config["experiment_name"]}')

    train_loader, val_loader = setup_data_loaders(
        config['train_dir'],
        config['val_dir'],
        config['batch_size']
    )

    model = EfficientNetB7().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['scheduler_patience'],
        factor=config['scheduler_factor']
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                val_loss += criterion(outputs, labels.float()).item()

                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels.float()).sum().item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f'checkpoints/model_{config["experiment_name"]}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print("Early stopping triggered")
                break

    writer.close()
    return model

if __name__ == "__main__":
    config = {
        'experiment_name': 'exp1',
        'train_dir': '/Users/nithin/documents/guardianai/dataset/data/train',
        'val_dir': '/Users/nithin/documents/guardianai/dataset/data/val',
        'batch_size': 4,       # Reduced for M3
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'scheduler_patience': 3,
        'scheduler_factor': 0.2,
        'early_stopping_patience': 5
    }

    os.makedirs('checkpoints', exist_ok=True)
    model = train(config)