import torch
import torch.nn as nn
import math

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, se_ratio=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion_factor = expansion_factor
        expanded_channels = in_channels * expansion_factor

        self.expand = nn.Identity() if expansion_factor == 1 else nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeezed_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeezed_channels, expanded_channels, 1),
            nn.Sigmoid()
        )

        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)
        if self.skip:
            x = x + identity
        return x

class EfficientNetB7(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        layers = [
            [1, 32, 4, 1],
            [6, 48, 7, 2],
            [6, 80, 7, 2],
            [6, 160, 10, 2],
            [6, 224, 10, 1],
            [6, 384, 13, 2],
            [6, 640, 4, 1]
        ]

        self.layers = []
        in_channels = 64

        for expansion_factor, channels, n_layers, stride in layers:
            for i in range(n_layers):
                s = stride if i == 0 else 1
                self.layers.append(
                    MBConvBlock(in_channels, channels,
                              kernel_size=3 if expansion_factor == 1 else 5,
                              stride=s, expansion_factor=expansion_factor)
                )
                in_channels = channels

        self.layers = nn.Sequential(*self.layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(640, 2560, kernel_size=1, bias=False),
            nn.BatchNorm2d(2560),
            nn.SiLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x