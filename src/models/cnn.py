import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvNet(nn.Module):
    """Basic CNN for hotdog classification"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ConvNet, self).__init__()
        # TODO: Implement basic CNN architecture
        pass
        
    def forward(self, x):
        # TODO: Implement forward pass
        pass


class ResNetBlock(nn.Module):
    """Custom ResNet-style block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        # TODO: Implement ResNet block
        pass
        
    def forward(self, x):
        # TODO: Implement forward pass with skip connection
        pass


class CustomResNet(nn.Module):
    """Custom ResNet-inspired architecture using ResNetBlock"""
    def __init__(self, num_classes=2, num_blocks=[2, 2, 2], channels=[64, 128, 256]):
        super(CustomResNet, self).__init__()
        # TODO: Initial convolution layer
        # TODO: Create layers using ResNetBlock with num_blocks configuration
        # TODO: Global average pooling and final classifier
        pass
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """Helper method to create a layer with multiple ResNet blocks"""
        # TODO: Create a sequence of ResNetBlocks
        pass
        
    def forward(self, x):
        # TODO: Implement forward pass through all layers
        pass


def get_model(model_type='simple', **kwargs):
    """Factory function to get different model types"""
    models_dict = {
        'simple': ConvNet,
        'custom_resnet': CustomResNet,
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models_dict[model_type](**kwargs)


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)