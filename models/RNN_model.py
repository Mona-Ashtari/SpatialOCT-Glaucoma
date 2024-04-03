import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_model(nn.Module):
    """
    RNN model with GRUs for sequence data processing.

    Parameters:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dim (int): Dimensionality of the hidden layer.
    - num_classes (int): Number of output classes.

    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GRU_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # GRU layers
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.3)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.gru1(x)

        x, _ = self.gru2(x)

        x = self.dropout1(x)

        # Adaptive pooling and transformation to latent space
        x = self.adaptive_pool(x.transpose(1, 2)).squeeze(2)

        x = self.fc(x)

        return x
