"""
Author: Mona Ashtari
Description: This script extracts features from OCT images using a Vision Transformer (ViT) model.

Project: SpatialOCT-Glaucoma
"""


import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import timm
from utils.pos_embed import interpolate_pos_embed
from models.vit_model import models_vit
from utils.OCT_dataset import OCT_Dataset
import pickle


def save_volume_path(output_path, class_id):
    """
    Prepares the directory for saving extracted features based on the label.

    Parameters:
    output_path (str): The base path for saving extracted features.
    class_id (int): The label of the data being processed.

    Returns:
    str: The path to the directory where features should be saved.
    """
    saving_path = os.path.join(os.getcwd(), output_path, class_id)
    if not os.path.exists(saving_path):
        os.makedirs(saving_path, exist_ok=True)
    return saving_path


def main():
    # Configuration and model parameters
    model_name = 'vit_large_patch16'
    data_path = 'OCT_data'
    finetune = 'RETFound_oct_weights.pth'
    output_path = 'OCT_features'
    batch_size = 32  # 64 slices per volume
    input_size = 128
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
        global_pool='avg',
    )

    # Load checkpoint model weights
    checkpoint = torch.load(finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Create a new checkpoint_model with renamed keys
    new_checkpoint_model = {}
    for key, value in checkpoint_model.items():
        # Rename 'norm.weight' and 'norm.bias' to 'fc_norm.weight' and 'fc_norm.bias'
        if key == 'norm.weight':
            print(key)
            new_checkpoint_model['fc_norm.weight'] = value
        elif key == 'norm.bias':
            print(key)
            new_checkpoint_model['fc_norm.bias'] = value
        else:
            new_checkpoint_model[key] = value

    # Remove head weights if they don't match the model's head dimensions.
    for k in ['head.weight', 'head.bias']:
        if k in new_checkpoint_model and new_checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del new_checkpoint_model[k]

    del checkpoint
    del checkpoint_model

    # Interpolate position embedding if the checkpoint model's image size is different.
    # This ensures the positional embeddings are compatible with the input image size.
    interpolate_pos_embed(model, new_checkpoint_model)

    # Load pre-trained model weights, now possibly adjusted for the model's architecture.
    model.load_state_dict(new_checkpoint_model, strict=False)

    model.to(device)
    model.eval()

    extracted_features = []

    for b, batch in enumerate(data_loader):
        slice_img = batch[0]
        img_id = batch[1]
        slice_img = slice_img.to(device)
        img_id = img_id[0]

        output = model(slice_img)
        extracted_features.append(output.cpu().detach())

        if 'Normal' in img_id:
            class_id = 'class0'
        else:
            class_id = 'class1'

        if b % 2 != 0:
            extracted_features = np.vstack(extracted_features)
            with open(os.path.join(output_path, class_id, f'{img_id[:img_id.rfind(".npy")]}.pickle'), 'wb') as handle:
                pickle.dump(extracted_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            extracted_features = []


if __name__ == '__main__':
    main()
    
