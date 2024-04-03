import torch
from utils.utils import one_hot_encoding
from torchmetrics import ConfusionMatrix


def evaluate_model(model, data_loader, criterion, device, alpha, gamma):
    """
    Evaluates the GRU model.

    Parameters:
    - model (torch.nn.Module): The GRU model to evaluate.
    - data_loader (DataLoader): DataLoader for validation or test data.
    - criterion: Loss function.
    - device: Torch device.
    - alpha (float): Alpha parameter for focal loss.
    - gamma (float): Gamma parameter for focal loss.

    Returns:
    - Tuple of average loss, accuracy, confusion matrix, probabilities, and labels for the validation set.
    """

    model.eval()

    total_loss = 0
    val_acc = 0
    confmat_val = 0
    confmat = ConfusionMatrix(task='binary', num_classes=model.num_classes).to(device)
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            labels_one_hot = one_hot_encoding(labels, num_classes=model.num_classes).to(device)

            loss = criterion(outputs, labels_one_hot, alpha=alpha, gamma=gamma)
            loss = loss.sum()
            total_loss += loss.item()

            # Convert outputs to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())

            # Performance measures
            _, preds = torch.max(outputs.data, 1)
            ConfMatrix = confmat(preds, labels)

            val_acc += (preds == labels).sum().item()
            confmat_val += ConfMatrix

            total += labels.size(0)

    all_probs = [prob[1] for prob in all_probs]
    return total_loss/total, val_acc/total, confmat_val, all_probs, all_labels


def montecarlo_uncertainty(model, data_loader, device, num_runs=100):
    """
    Estimates model uncertainty using Monte Carlo Dropout.

    Parameters:
    - model (torch.nn.Module): The trained model with dropout layers.
    - data_loader (DataLoader): DataLoader for validation or test data.
    - device: Torch device.
    - num_runs (int): Number of forward passes to simulate.

    Returns:
    - Mean and standard deviation of model predictions across runs.
    """
    predicts = []

    # Run MC-Dropout inference
    with torch.no_grad():
        model.train()  # Set the model to training mode to enable dropout
        for _ in range(num_runs):
            val_acc = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                # Performance measures
                _, preds = torch.max(outputs.data, 1)

                val_acc += (preds == labels).sum().item()
            predicts.append(val_acc)

    # Stack predictions along the batch dimension
    predicts = np.stack(predicts)

    # Compute statistics (mean and standard deviation) across MC-Dropout runs
    mean_prediction = np.mean(predicts)
    std_prediction = np.std(predicts)

    return predicts, mean_prediction, std_prediction
