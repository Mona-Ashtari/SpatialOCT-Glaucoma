import torch


def one_hot_encoding(labels, num_classes):
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return labels_one_hot.float()


def calculate_sample_weights(dataset):
    """
    Calculates weights for balancing classes in a dataset.

    Parameters:
    - dataset (Dataset): The dataset object containing class labels.

    Returns:
    - list[float]: Weights for each sample in the dataset.
    """

    # Count the number of occurrences of each class
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Total number of samples
    num_samples = len(dataset)

    # Calculate weight for each class
    weights = {cls: num_samples / class_counts[cls] for cls in class_counts}

    # Create a list of weights for each sample in the dataset
    sample_weights = [weights[label] for _, label in dataset]

    return sample_weights


