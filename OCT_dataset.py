from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_volume(file_path):
    """
    Load a 3D OCT volume from a file.

    Parameters:
    file_path (str): The file path to the .npy file containing the 3D volume.

    Returns:
    numpy.ndarray: The loaded 3D OCT volume.
    """
    img_3D = np.load(file_path)
    return img_3D


class GrayscaleToRGB:
    """
    A transformation class to convert a grayscale image to RGB by duplicating the single channel.
    """
    def __call__(self, image):
        image_rgb = image.convert('RGB')
        return image_rgb


class OCT_Dataset(Dataset):
    """
    A PyTorch Dataset class for OCT images. This class handles loading of OCT images,
    applying necessary transformations, and returning the image and its label.

    Attributes:
    data_path (str): Path to the directory containing the data.
    input_size (int): The target size for each image (height, width).
    data_transform (torchvision.transforms.Compose): A composition of transformations to apply to the images.
    samples (list): A list of tuples, each containing the file path and label for each image.
    slice_counts (list): A list of the number of slices in each volume.
    """
    def __init__(self, data_path, input_size):
        """
        Initializes the OCT_Dataset instance.

        Parameters:
        data_path (str): The directory where the data is stored.
        input_size (int): The height and width to which the images should be resized.
        """
        self.data_path = data_path
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            GrayscaleToRGB(),
            transforms.Resize((input_size, input_size)),  # Resize to 128x128
            transforms.ToTensor(), # Convert PIL Image to Tensor
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        self.samples = self._load_samples()
        self.slice_counts = [load_volume(file_path[0]).shape[0] for file_path in self.samples]

    def _load_samples(self):
        """
        Load sample paths and labels from the dataset directory.

        Returns:
        list: A list of tuples, each containing the path to a sample and its label.
        """
        samples = []
        for label in ["class0", "class1"]:
            class_dir = os.path.join(self.data_path, label)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                samples.append((file_path, int(label[-1])))
        return samples

    def __len__(self):
        """
        Returns the total number of slices across all volumes in the dataset.

        Returns:
        int: The total number of slices.
        """
        return sum(self.slice_counts)  # Total number of slices

    def __getitem__(self, idx):
        """
        Retrieve a specific item from the dataset.

        Parameters:
        idx (int): The index of the item.

        Returns:
        tuple: A tuple containing the transformed image and its label.
        """

        # Determine which volume and which slice within that volume to load
        volume_idx, slice_idx = 0, idx
        while slice_idx >= self.slice_counts[volume_idx]:
            slice_idx -= self.slice_counts[volume_idx]
            volume_idx += 1

        volume_data = load_volume(self.samples[volume_idx][0])

        slice_img = volume_data[slice_idx]
        label = self.samples[volume_idx][1]

        slice_img = self.data_transform(slice_img)

        return slice_img, label

