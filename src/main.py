import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.dataloader import Hotdog_NotHotdog
from models.models import get_model, count_parameters
from visualizations.training_plots import (
    plot_training_curves,
    plot_confusion_matrix,
    save_training_history,
)

from visualizations.saliency_maps import plot_saliency_maps

from utils import (
    get_device,
    train_epoch,
    test_epoch,
    split_train_dataset,
    get_predictions,
    EarlyStopping,
    get_model_choice,
    get_transforms,
    get_model_image_size,
    prompt_save_model,
    save_model,
    prompt_saliency_maps
)

device = get_device()

def main():
    # Model to train (cnn or custom_resnet)
    model_type = get_model_choice()

    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0001
    image_size = get_model_image_size(model_type)
    weight_decay = 1e-4
    early_stopping_patience = 5
    early_stopping_min_delta = 0.001

    # Get model-specific transforms
    train_transform, test_transform = get_transforms(model_type, image_size)

    # Set data path relative to script location
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # Create results directory for saving plots
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load training and test datasets separately
    print("Loading training dataset...")
    full_train_dataset = Hotdog_NotHotdog(
        train=True, transform=train_transform, data_path=data_path
    )

    print("Loading test dataset...")
    test_dataset = Hotdog_NotHotdog(
        train=False, transform=test_transform, data_path=data_path
    )

    print(f"Training samples: {len(full_train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if len(full_train_dataset) == 0:
        print("Error: Training dataset is empty!")
        return

    # Split only the training dataset into train and validation
    train_dataset, val_dataset = split_train_dataset(
        full_train_dataset, train_ratio=0.85, val_ratio=0.15, seed=42
    )

    # Apply test transforms to validation set
    val_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = get_model(model_type=model_type, num_classes=2, dropout_rate=0.5)
    model.to(device)

    print(f"Model has {count_parameters(model)} trainable parameters")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        restore_best_weights=True,
        verbose=True,
    )

    # Lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = test_epoch(model, val_loader, criterion, device)

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Check early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    print(f"\nTraining completed after {len(train_losses)} epochs")

    # Final test on test set
    print("\nFinal evaluation on test set:")
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Get predictions
    test_predictions, test_labels = get_predictions(model, test_loader, device)

    # Generate and save visualizations
    print("\nGenerating visualizations...")

    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        save_path=os.path.join(results_dir, "training_curves.png"),
    )

    # Plot confusion matrix for test set
    print("\nConfusion Matrix - Test Set:")
    plot_confusion_matrix(
        test_predictions,
        test_labels,
        class_names=["Not Hotdog", "Hotdog"],
        save_path=os.path.join(results_dir, "confusion_matrix_test.png"),
    )

    # Save training history
    save_training_history(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        save_path=os.path.join(results_dir, "training_history.csv"),
    )

    # Generate saliency maps for transfer learning models
    if model_type == "efficientnet":
        if prompt_saliency_maps():
            print("\nGenerating saliency maps...")
            plot_saliency_maps(
                model,
                test_loader,
                device,
                num_samples=6,
                save_path=os.path.join(results_dir, "saliency_maps.png"),
            )

    print(f"\nAll results saved to: {results_dir}")

    if prompt_save_model():
        try:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", "trained_models"
            )
            save_path = save_model(model, model_type, test_acc, save_dir)
            print(f"Model saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("Model not saved.")


if __name__ == "__main__":
    main()