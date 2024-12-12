from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2


class GrayscaleToRGB:
    def __call__(self, image):
        image_rgb = image.convert('RGB')
        return image_rgb  # Convert back to PIL image


def load_volume(file_path):
    """
    Load a 3D volume from a file.
    This function needs to be defined based on your file format.
    """
    img_3D = np.load(file_path)
    return img_3D


class OCT_Dataset(Dataset):
    """
    A PyTorch Dataset class for OCT images. This class handles loading of OCT images,
    applying necessary transformations, and returning the image and its label.

    Attributes:
    input_size (int): The target size for each image (height, width).
    data_transform (torchvision.transforms.Compose): A composition of transformations to apply to the images.
    file_paths (list): A list of tuples, each containing the file path and label for each image.
    """
    def __init__(self, data_path, input_size):
        """
        Initializes the OCT_Dataset instance.

        Parameters:
        data_path (str): The directory where the data is stored.
        input_size (int): The height and width to which the images should be resized.
        """

        self.input_size = input_size
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            GrayscaleToRGB(),
            transforms.Resize((self.input_size, self.input_size)),  # Resize to 128x128
            transforms.ToTensor(), # Convert PIL Image to Tensor
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        self.file_paths = [[os.path.join(data_path, file), file] for file in os.listdir(data_path) if file.endswith('.npy')]

    def __len__(self):
        return len(self.file_paths) * 64  # Total number of slices

    def __getitem__(self, idx):

        # Determine which volume and which slice within that volume to load
        volume_idx = idx // 64  # Integer division to get the volume index
        slice_idx = idx % 64

        volume_data = load_volume(self.file_paths[volume_idx][0])
        volume_name = self.file_paths[volume_idx][1]

        slice_img = volume_data[slice_idx]
        slice_img = self.data_transform(slice_img)

        return slice_img, volume_name
