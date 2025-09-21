import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Basic CNN for hotdog classification with batch normalization"""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers (assuming 128x128 input -> 16x16x128 after 3 pooling)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv layers with BatchNorm, ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32x64
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16x16x128

        # Flatten
        x = x.view(-1, 128 * 16 * 16)

        # Fully connected layers with BatchNorm
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNetBlock(nn.Module):
    """Custom ResNet-style block with skip connections"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Skip connection
        out = F.relu(out)
        
        return out


class CustomResNet(nn.Module):
    """Custom ResNet-inspired architecture using ResNetBlock"""

    def __init__(self, num_classes=2, num_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], dropout_rate=0.5):
        super(CustomResNet, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], num_blocks[3], stride=2)
        
        # Global average pooling, dropout, and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper method to create a layer with multiple ResNet blocks"""
        layers = []
        # First block might have stride > 1 for downsampling
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling, dropout, and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def get_model(model_type="simple", **kwargs):
    """Factory function to get different model types"""
    models_dict = {
        "cnn": ConvNet,
        "custom_resnet": CustomResNet,
    }

    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}")

    return models_dict[model_type](**kwargs)


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
