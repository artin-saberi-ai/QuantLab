import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, image_size=28, widths=(32, 64)):
        super().__init__()
        w1, w2 = widths
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(w1, w2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        spatial = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w2 * spatial * spatial, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10, widths=(16, 32, 64), blocks_per_stage=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        stages = []
        in_channels = widths[0]
        for stage, out_channels in enumerate(widths):
            stride = 1 if stage == 0 else 2
            layers = [BasicBlock(in_channels, out_channels, stride=stride)]
            layers += [BasicBlock(out_channels, out_channels) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*layers))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_model(name, in_channels, num_classes, image_size, **kwargs):
    if name == "simple_cnn":
        return SimpleCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
            widths=tuple(kwargs.get("widths", (32, 64))),
        )
    if name == "resnet20":
        return ResNet20(num_classes=num_classes)
    raise ValueError(f"unknown model '{name}' (expected simple_cnn or resnet20)")
