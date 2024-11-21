#Custom denseNet121

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseLayer, self).__init__()

        # BN -> ReLU -> 1x1 Conv
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        )

        # BN -> ReLU -> 3x3 Conv
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        )

    def forward(self, x):
        new_features = self.layer1(x)
        new_features = self.layer2(new_features)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size
            ))

    def forward(self, x):
        features = x
        for layer in self.layers:
            features = layer(features)
        return features

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)

class CustomDenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000, bn_size=4):
        super(CustomDenseNet121, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # DenseNet specific parameters
        num_layers = [6, 12, 24, 16]  # Number of dense layers in each block
        in_channels = 64

        # Dense blocks and transition layers
        blocks = []
        for i, num_layers_in_block in enumerate(num_layers):
            # Add dense block
            block = DenseBlock(
                num_layers=num_layers_in_block,
                in_channels=in_channels,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            blocks.append(block)
            in_channels += num_layers_in_block * growth_rate

            # Add transition layer between dense blocks (except after last block)
            if i != len(num_layers) - 1:
                out_channels = in_channels // 2  # Compression factor of 0.5
                trans = TransitionLayer(in_channels, out_channels)
                blocks.append(trans)
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(in_channels)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features = self.blocks(features)
        features = self.final_bn(features)
        out = self.classifier(features)
        return out

class CustomAIDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomAIDetector, self).__init__()

        # Initialize base DenseNet121
        self.densenet = CustomDenseNet121(growth_rate=32, num_classes=1000)

        # Replace the classifier for binary classification
        in_features = self.densenet.classifier[-1].in_features
        self.densenet.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)

# Training utilities
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')

    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100. * correct / total

# Usage example:
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomAIDetector().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop would go here
    print("Model architecture created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == '__main__':
    main()