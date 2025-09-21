import torch
from torch.utils.data import random_split


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