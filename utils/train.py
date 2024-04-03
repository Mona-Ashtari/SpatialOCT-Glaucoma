import torch
from utils.utils import one_hot_encoding


def train_model(model, data_loader, optimizer, criterion, device, alpha, gamma):
    """
    Trains the GRU model for one epoch.

    Parameters:
    - model (torch.nn.Module): The GRU model to train.
    - data_loader (DataLoader): DataLoader for training data.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - criterion: Loss function.
    - device: Torch device.
    - alpha (float): Alpha parameter for focal loss.
    - gamma (float): Gamma parameter for focal loss.

    Returns:
    - Tuple of average loss and accuracy for the training set.
    """
    model.train()

    total_loss = 0
    train_acc = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        labels_one_hot = one_hot_encoding(labels, num_classes=model.num_classes).to(device)

        # Compute the loss
        loss = criterion(outputs, labels_one_hot, alpha=alpha, gamma=gamma)
        loss = loss.sum()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        train_acc += (preds == labels).sum().item()

        total += labels.size(0)

    total_loss = total_loss / total
    train_acc = train_acc / total

    return total_loss, train_acc
