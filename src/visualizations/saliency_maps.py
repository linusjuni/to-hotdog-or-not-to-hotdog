import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import random
from scipy import ndimage

NUM_SAMPLES = 3

def generate_saliency_map(
    model, image, target_class, device, num_samples=NUM_SAMPLES, noise_level=0.15
):
    """
    Generate a saliency map using SmoothGrad method.

    Args:
        model: Trained PyTorch model
        image: Input image tensor (C, H, W)
        target_class: Target class for gradient computation
        device: Device to run computation on
        num_samples: Number of noisy samples for averaging
        noise_level: Standard deviation of noise to add

    Returns:
        numpy array: Saliency map
    """
    model.eval()
    image = image.to(device)

    saliencies = []

    for _ in range(num_samples):
        # Add noise to input
        noise = torch.randn_like(image).to(device) * noise_level
        noisy_image = image + noise
        noisy_image = noisy_image.unsqueeze(0)
        noisy_image.requires_grad_(True)

        # Forward pass
        output = model(noisy_image)
        class_score = output[0, target_class]

        # Backward pass
        model.zero_grad()
        class_score.backward()

        # Get gradients and compute saliency
        gradients = noisy_image.grad.data.squeeze(0)
        saliency = torch.norm(gradients, dim=0)
        saliencies.append(saliency.cpu().numpy())

    # Average all saliencies
    avg_saliency = np.mean(saliencies, axis=0)
    return avg_saliency


def normalize_saliency(saliency):
    """Normalize saliency map using percentile normalization for better contrast"""
    p2, p98 = np.percentile(saliency, [2, 98])
    saliency = np.clip((saliency - p2) / (p98 - p2 + 1e-8), 0, 1)
    return saliency


def apply_gaussian_blur(saliency, sigma=1.2):
    """Apply Gaussian blur to smooth the saliency map"""
    return ndimage.gaussian_filter(saliency, sigma=sigma)


def plot_saliency_maps(model, test_loader, device, num_samples=NUM_SAMPLES, save_path=None):
    """
    Generate and plot saliency maps using SmoothGrad.
    """
    model.eval()

    # Sample random hotdog images
    dataset = getattr(test_loader, "dataset", None)
    dataset_len = len(dataset)

    if dataset_len == 0:
        raise ValueError("The dataset is empty")

    # Filter for hotdog images
    hotdog_indices = []
    for idx in range(dataset_len):
        _, label = dataset[idx]
        if label == 0:
            hotdog_indices.append(idx)

    if len(hotdog_indices) == 0:
        raise ValueError("No hotdog images found in the dataset")

    print(f"Found {len(hotdog_indices)} hotdog images in test dataset")

    # Sample images
    num = min(num_samples, len(hotdog_indices))
    indices = random.sample(hotdog_indices, k=num)

    images = []
    labels = []
    image_paths = []
    for idx in indices:
        img, lbl = dataset[idx]
        images.append(img)
        labels.append(lbl)
        try:
            image_paths.append(dataset.image_paths[idx])
        except Exception:
            image_paths.append(None)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Create figure with 3 columns: Original, Saliency Map, Overlay
    fig, axes = plt.subplots(num, 3, figsize=(12, 4 * num))
    if num == 1:
        axes = axes.reshape(1, 3)

    class_names = ["Hotdog", "Not Hotdog"]

    # Reverse normalization for display
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    for i in range(min(num, len(images))):
        image = images[i]
        true_label = labels[i].item()

        # Print image path
        img_path = image_paths[i] if i < len(image_paths) else None
        if img_path:
            print(f"Image {i + 1} path: {img_path}")

        # Get model prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred_prob = F.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = pred_prob[0, pred_class].item()

        # Generate saliency map
        print(f"Generating SmoothGrad saliency for image {i + 1}...")
        saliency = generate_saliency_map(model, image, pred_class, device)

        # Apply enhancements
        saliency = normalize_saliency(saliency)
        saliency = apply_gaussian_blur(saliency)

        # Unnormalize image for display
        display_image = unnormalize(image)
        display_image = torch.clamp(display_image, 0, 1)
        display_image = display_image.permute(1, 2, 0).numpy()

        # Plot original image
        axes[i, 0].imshow(display_image)
        axes[i, 0].set_title(f"Original\nTrue: {class_names[true_label]}", fontsize=10)
        axes[i, 0].axis("off")

        # Plot saliency map
        axes[i, 1].imshow(saliency, cmap="jet", interpolation="bilinear")
        axes[i, 1].set_title(
            f"Saliency Map\nPred: {class_names[pred_class]}", fontsize=10
        )
        axes[i, 1].axis("off")

        # Plot overlay
        axes[i, 2].imshow(display_image)
        axes[i, 2].imshow(saliency, cmap="jet", alpha=0.5, interpolation="bilinear")
        axes[i, 2].set_title(f"Overlay\nConf: {confidence:.2f}", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saliency maps saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from models.dataloader import Hotdog_NotHotdog
    from utils import load_model, get_device, get_transforms, get_model_image_size

    device = get_device()

    # Get model path from user
    trained_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "trained_models"
    )

    if not os.path.exists(trained_models_dir):
        print(f"No trained models directory found at {trained_models_dir}")
        exit()

    model_files = [f for f in os.listdir(trained_models_dir) if f.endswith(".pth")]

    if not model_files:
        print("No trained models found!")
        exit()

    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i + 1}. {model_file}")

    while True:
        try:
            choice = int(input("\nSelect a model (number): ")) - 1
            if 0 <= choice < len(model_files):
                model_path = os.path.join(trained_models_dir, model_files[choice])
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")

    # Load the model
    model = load_model(model_path, device)

    # Get model type from filename or checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint["model_type"]

    # Setup data
    image_size = get_model_image_size(model_type)
    _, test_transform = get_transforms(model_type, image_size)

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    test_dataset = Hotdog_NotHotdog(
        train=False, transform=test_transform, data_path=data_path
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if len(test_dataset) == 0:
        print("No test data found!")
        exit()

    # Generate saliency maps
    num_samples = NUM_SAMPLES
    print(
        f"\nGenerating SmoothGrad saliency maps for {num_samples} random hotdog samples..."
    )
    plot_saliency_maps(model, test_loader, device, num_samples=num_samples)
