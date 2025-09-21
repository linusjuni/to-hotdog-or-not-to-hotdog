import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plot_class_imbalance(dataset, class_names=None, title="Class Distribution"):
    """
    Plot the class distribution of a dataset to visualize class imbalance.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset to analyze
    class_names : list, optional
        List of class names. If None, uses indices
    title : str, optional
        Title for the plot
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Extract all labels from the dataset
    labels = []
    print(f"Analyzing dataset with {len(dataset)} samples...")

    for i in range(len(dataset)):
        try:
            _, label = dataset[i]
            labels.append(label)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue

    if not labels:
        print("No labels could be extracted from the dataset!")
        return

    # Count occurrences of each class
    label_counts = Counter(labels)

    # Get class names
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(label_counts.keys())]

    # Prepare data for plotting
    classes = sorted(label_counts.keys())
    counts = [label_counts[cls] for cls in classes]
    class_labels = [class_names[i] for i in classes]

    # Define pastel colors: green for hotdog, red for not hotdog
    colors = []
    for label in class_labels:
        if "hotdog" in label.lower() and "not" not in label.lower():
            colors.append("#90EE90")  # Light green (pastel)
        else:
            colors.append("#FFB6C1")  # Light pink/red (pastel)

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_labels, counts, color=colors)

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_samples = sum(counts)
    print("\nDataset Summary:")
    print(f"Total samples: {total_samples}")
    for i, (cls_name, count) in enumerate(zip(class_labels, counts)):
        percentage = (count / total_samples) * 100
        print(f"{cls_name}: {count} samples ({percentage:.1f}%)")
    if len(counts) > 1:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
