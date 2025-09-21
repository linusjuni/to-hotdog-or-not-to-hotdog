import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Basic CNN for hotdog classification"""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers (assuming 128x128 input -> 16x16x128 after 3 pooling)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 64x64x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32x64
        x = self.pool(F.relu(self.conv3(x)))  # 16x16x128

        # Flatten
        x = x.view(-1, 128 * 16 * 16)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNetBlock(nn.Module):
    """Custom ResNet-style block with skip connections"""

    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        # TODO: Implement ResNet block
        raise NotImplementedError("ResNetBlock.__init__ not implemented")

    def forward(self, x):
        # TODO: Implement forward pass with skip connection
        raise NotImplementedError("ResNetBlock.forward not implemented")


class CustomResNet(nn.Module):
    """Custom ResNet-inspired architecture using ResNetBlock"""

    def __init__(self, num_classes, num_blocks, channels):
        super(CustomResNet, self).__init__()
        # TODO: Initial convolution layer
        # TODO: Create layers using ResNetBlock with num_blocks configuration
        # TODO: Global average pooling and final classifier
        raise NotImplementedError("CustomResNet.__init__ not implemented")

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper method to create a layer with multiple ResNet blocks"""
        # TODO: Create a sequence of ResNetBlocks
        raise NotImplementedError("CustomResNet._make_layer not implemented")

    def forward(self, x):
        # TODO: Implement forward pass through all layers
        raise NotImplementedError("CustomResNet.forward not implemented")


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
