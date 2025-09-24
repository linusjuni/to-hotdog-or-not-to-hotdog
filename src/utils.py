import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
import os
from datetime import datetime

from models.models import count_parameters


def get_device():
    if torch.cuda.is_available():
        print("The code will run on GPU (CUDA).")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("The code will run on GPU (MPS).")
        device = torch.device("mps")
    else:
        print("The code will run on CPU.")
        device = torch.device("cpu")
    return device

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                           Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                              Default: 0.001
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Call this method after each epoch with the validation loss and model.
        
        Args:
            val_loss (float): Current validation loss
            model: The model being trained
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Save the current best weights
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            # Validation loss didn't improve
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epoch(s)")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"Restored model weights from best epoch (val_loss: {self.best_loss:.6f})")
                
        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print("Training...")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def test_epoch(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    print("Testing...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def split_train_dataset(dataset, train_ratio=0.85, val_ratio=0.15, seed=42):
    """
    Split a training dataset into train and validation sets.

    Args:
        dataset: PyTorch training dataset to split
        train_ratio: Proportion for training set (default: 0.85)
        val_ratio: Proportion for validation set (default: 0.15)
        seed: Random seed for reproducible splits (default: 42)

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Check that ratios sum to 1
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("Train and validation ratios must sum to 1.0")

    # Set seed for reproducible splits
    torch.manual_seed(seed)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size  # Ensure all samples are used

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("Training dataset split:")
    print(f"  Total training samples: {total_size}")
    print(f"  Training: {len(train_dataset)} ({len(train_dataset) / total_size:.1%})")
    print(f"  Validation: {len(val_dataset)} ({len(val_dataset) / total_size:.1%})")

    return train_dataset, val_dataset


def load_model(model_path, device=None):
    """
    Load a trained model from a .pth file.
    """
    if device is None:
        device = get_device()

    # Load the saved data
    checkpoint = torch.load(model_path, map_location=device)

    # Get model info from checkpoint
    model_type = checkpoint["model_type"]

    # Recreate the model architecture
    from models.models import get_model

    model = get_model(model_type=model_type, num_classes=2, dropout_rate=0.5)

    # Load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device
    model.to(device)

    print(f"Loaded {model_type} model from {model_path}")
    print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")

    return model
    


def get_predictions(model, data_loader, device):
    """
    Get predictions and true labels for a dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for the dataset to evaluate
        device: Device to run the model on
    
    Returns:
        tuple: (predictions, true_labels) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return all_predictions, all_targets


def get_model_image_size(model_type):
    """
    Get the appropriate image size for different model types.
    """
    size_mapping = {
        'cnn': 128,
        'custom_resnet': 128,
        'efficientnet': 224,
    }
    
    return size_mapping.get(model_type, 128)


def get_transforms(model_type, image_size=None):
    """
    Get appropriate transforms based on model type.
    
    Args:
        model_type (str): Type of model
        image_size (int, optional): Image size. If None, uses model-specific default.
    """
    # Use model-specific image size if not provided
    if image_size is None:
        image_size = get_model_image_size(model_type)
    
    print(f"Using image size: {image_size}x{image_size}")
    
    # ImageNet normalization for transfer learning models
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Custom normalization for models trained from scratch
    custom_mean = [0.5, 0.5, 0.5]
    custom_std = [0.5, 0.5, 0.5]
    
    # Choose normalization based on model type
    if model_type in ['efficientnet']:  # Add other transfer learning models here
        mean, std = imagenet_mean, imagenet_std
    else:
        mean, std = custom_mean, custom_std

    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, test_transform

def save_model(model, model_type, test_acc, save_dir="models/trained_models"):
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained PyTorch model
        model_type (str): Type of model (e.g., 'cnn', 'custom_resnet', 'efficientnet')
        test_acc (float): Test accuracy achieved by the model
        save_dir (str): Directory to save the model
    
    Returns:
        str: Path where the model was saved
    """
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_acc{test_acc:.1f}_{timestamp}.pth"
    save_path = os.path.join(save_dir, filename)
    
    # Save model state dict along with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'test_accuracy': test_acc,
        'timestamp': timestamp,
        'num_parameters': count_parameters(model)
    }, save_path)
    
    return save_path


def prompt_save_model():
    """Prompt user whether to save the model."""
    while True:
        try:
            choice = input("\nWould you like to save this trained model? (y/n): ").strip().lower()
            
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return False
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def prompt_saliency_maps():
    """Prompt user whether to generate saliency maps."""
    while True:
        try:
            choice = (
                input(
                    "\nWould you like to generate saliency maps to see what the model focuses on? (y/n): "
                )
                .strip()
                .lower()
            )

            if choice in ["y", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return False
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def get_model_choice():
    """Prompt user to select which model to train."""
    print("Available models:")
    print("1. CNN")
    print("2. Custom ResNet")
    print("3. EfficientNet-B0 (Transfer Learning)")
    
    while True:
        try:
            choice = input("\nPlease select a model (1, 2, or 3): ").strip()
            
            if choice == "1":
                return "cnn"
            elif choice == "2":
                return "custom_resnet"
            elif choice == "3":
                return "efficientnet"
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")