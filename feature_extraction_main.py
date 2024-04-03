"""
Author: Mona Ashtari
Description: This script extracts features from OCT images using a Vision Transformer (ViT) model.

Project: SpatialOCT-Glaucoma
"""

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
from utils.pos_embed import interpolate_pos_embed
from vit_model import models_vit

from utils.OCT_dataset import OCT_Dataset
import os
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn


# Global variable to store model outputs during forward pass
outputs = {}


def hook(model, data, output):
    """
    Hook function to capture the output of the model's final layer.

    Parameters:
    model (torch.nn.Module): The model being hooked.
    data (torch.Tensor): The input data to the model.
    output (torch.Tensor): The output from the model's hooked layer.
    """
    outputs["fc_norm"] = output


def save_volume_path(output_path, label):
    """
    Prepares the directory for saving extracted features based on the label.

    Parameters:
    output_path (str): The base path for saving extracted features.
    label (int): The label of the data being processed.

    Returns:
    str: The path to the directory where features should be saved.
    """
    saving_path = os.path.join(os.getcwd(), output_path, f'class{label}')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)
    return saving_path


def volume_index(volumes, label):
    """
    Manages and updates the index for each volume based on its label.

    Parameters:
    volumes (dict): A dictionary tracking the current index of volumes for each label.
    label (int): The label of the current volume.

    Returns:
    tuple: A tuple containing the updated volume index and the volumes dictionary.
    """
    volumes[label] += 1
    return volumes[label], volumes


def main():

    # Configuration and model parameters
    model_name = 'vit_large_patch16'
    data_path = 'OCT_data'
    finetune = 'RETFound_oct_weights.pth'
    output_path = 'OCT_features'
    batch_size = 64
    input_size = 128
    drop_path = 0.1
    nb_classes = 2

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    cudnn.benchmark = True

    # Data loading
    dataset = OCT_Dataset(data_path, input_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

    # Model loading and configuration
    model = models_vit.__dict__[model_name](
        img_size=input_size,
        num_classes=nb_classes,
        drop_path_rate=drop_path,
        global_pool=True,
    )

    # Load checkpoint model weights
    checkpoint = torch.load(finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Remove head weights if they don't match the model's head dimensions.
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # Interpolate position embedding if the checkpoint model's image size is different.
    # This ensures the positional embeddings are compatible with the input image size.
    interpolate_pos_embed(model, checkpoint_model)

    # Load pre-trained model weights, now possibly adjusted for the model's architecture.
    model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    model.eval()

    # Registering the hook
    hook_handle = model.fc_norm.register_forward_hook(hook)

    volumes = {0: -1, 1: -1}  # Initial volume indices

    # Feature extraction loop
    for data, label in data_loader:
        data = data.to(device)
        label = label[0].item()

        model(data)  # Forward pass to trigger the hook

        extracted_features = outputs["fc_norm"].cpu().detach()

        saving_path = save_volume_path(output_path, label)

        volume_idx, volumes = volume_index(volumes, label)
        feature_file_path = os.path.join(saving_path, f'volume{volume_idx}.pickle')

        with open(feature_file_path, 'wb') as handle:
            pickle.dump(extracted_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    hook_handle.remove()  # Clean up the hook


if __name__ == '__main__':
    main()
