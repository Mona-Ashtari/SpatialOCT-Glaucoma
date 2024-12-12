import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_model(nn.Module):
    """
    RNN model with GRUs for sequence data processing.

    Parameters:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dim1 (int): Dimensionality of the hidden layer in GRU1.
    - hidden_dim2 (int): Dimensionality of the hidden layer in GRU2.
    - num_classes (int): Number of output classes.
    - drop_rate (float): Dropout rate for regularization. Default: 0.3.
    - feature (bool): If True, the model outputs intermediate features instead of the final classification. Default: False.

    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, drop_rate=0.3, feature=False):
        super(GRU_model, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.feature = feature

        # GRU layers
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=drop_rate)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim2 * 2, num_classes)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout1(x)

        # Adaptive pooling and transformation to latent space
        x = self.adaptive_pool(x.transpose(1, 2)).squeeze(2)

        if self.feature:
            return x

        x = self.fc(x)

        return x
