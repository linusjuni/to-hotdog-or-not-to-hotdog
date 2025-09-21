import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

# Set seaborn style and palette
sns.set_style("whitegrid")
sns.set_palette("pastel")


def plot_training_curves(
    train_losses, val_losses, train_accuracies, val_accuracies, save_path=None
):
    """
    Plot training and validation loss and accuracy curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        save_path: Optional path to save the plots
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss curves
    sns.lineplot(
        x=epochs,
        y=train_losses,
        label="Training Loss",
        linewidth=2.5,
        ax=ax1,
        color=sns.color_palette("pastel")[0],
    )
    sns.lineplot(
        x=epochs,
        y=val_losses,
        label="Validation Loss",
        linewidth=2.5,
        ax=ax1,
        color=sns.color_palette("pastel")[1],
    )
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracy curves
    sns.lineplot(
        x=epochs,
        y=train_accuracies,
        label="Training Accuracy",
        linewidth=2.5,
        ax=ax2,
        color=sns.color_palette("pastel")[0],
    )
    sns.lineplot(
        x=epochs,
        y=val_accuracies,
        label="Validation Accuracy",
        linewidth=2.5,
        ax=ax2,
        color=sns.color_palette("pastel")[1],
    )
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_confusion_matrix(predictions, true_labels, class_names=None, save_path=None):
    """
    Generate and plot confusion matrix from predictions and true labels.

    Args:
        predictions: Array of predicted labels
        true_labels: Array of true labels
        class_names: List of class names for labeling (default: ['Not Hotdog', 'Hotdog'])
        save_path: Optional path to save the confusion matrix plot

    Returns:
        confusion matrix as numpy array
    """
    if class_names is None:
        class_names = ["Not Hotdog", "Hotdog"]

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Create simple confusion matrix plot
    plt.figure(figsize=(6, 5))

    # Use simple continuous colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    return cm


def save_training_history(
    train_losses, val_losses, train_accuracies, val_accuracies, save_path
):
    """
    Save training history to a text file.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        save_path: Path to save the history file
    """
    with open(save_path, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_Acc,Val_Acc\n")
        for i in range(len(train_losses)):
            f.write(
                f"{i + 1},{train_losses[i]:.4f},{val_losses[i]:.4f},"
                f"{train_accuracies[i]:.2f},{val_accuracies[i]:.2f}\n"
            )

    print(f"Training history saved to {save_path}")
