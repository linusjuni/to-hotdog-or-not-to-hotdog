import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms


def generate_saliency_map(model, image, target_class, device):
    """
    Generate a saliency map for a single image using gradients.

    Args:
        model: Trained PyTorch model
        image: Input image tensor (C, H, W)
        target_class: Target class for gradient computation
        device: Device to run computation on

    Returns:
        numpy array: Saliency map
    """
    model.eval()

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    image.requires_grad_(True)

    # Forward pass
    output = model(image)

    # Get the score for the target class
    class_score = output[0, target_class]

    # Backward pass to compute gradients
    model.zero_grad()
    class_score.backward()

    # Get gradients and convert to saliency map
    gradients = image.grad.data.squeeze(0)

    # Take absolute value and max across color channels
    saliency = torch.max(torch.abs(gradients), dim=0)[0]

    return saliency.cpu().numpy()


def plot_saliency_maps(model, test_loader, device, num_samples=6, save_path=None):
    """
    Generate and plot saliency maps for a few test samples.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run computation on
        num_samples: Number of samples to show (default: 6)
        save_path: Optional path to save the plot
    """
    model.eval()

    # Get a batch of test samples
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Take only the requested number of samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))
    if num_samples == 1:
        axes = axes.reshape(3, 1)

    class_names = ["Not Hotdog", "Hotdog"]

    # Reverse normalization for display (assuming ImageNet normalization)
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for i in range(min(num_samples, len(images))):
        image = images[i]
        true_label = labels[i].item()

        # Get model prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred_prob = F.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = pred_prob[0, pred_class].item()

        # Generate saliency map for the predicted class
        saliency = generate_saliency_map(model, image, pred_class, device)

        # Unnormalize image for display
        display_image = unnormalize(image)
        display_image = torch.clamp(display_image, 0, 1)
        display_image = display_image.permute(1, 2, 0).numpy()

        # Plot original image
        axes[0, i].imshow(display_image)
        axes[0, i].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=10)
        axes[0, i].axis("off")

        # Plot saliency map
        axes[1, i].imshow(saliency, cmap="hot", interpolation="bilinear")
        axes[1, i].set_title(
            f"Saliency Map\nPred: {class_names[pred_class]}", fontsize=10
        )
        axes[1, i].axis("off")

        # Plot overlay
        axes[2, i].imshow(display_image)
        axes[2, i].imshow(saliency, cmap="hot", alpha=0.4, interpolation="bilinear")
        axes[2, i].set_title(f"Overlay\nConf: {confidence:.2f}", fontsize=10)
        axes[2, i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saliency maps saved to {save_path}")

    plt.show()

