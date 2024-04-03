import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, root_dir, indexes, transform=None):
        """
        Args:
            root_dir (string): Directory with all the pickle files.
            indexes:
            [0] class0_index_split
            [1] class1_index_split
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load data paths and labels
        for l, label in enumerate(["class0", "class1"]):
            class_dir = os.path.join(root_dir, label)
            class_data = np.array(os.listdir(class_dir))[indexes[l]]
            for file in class_data:
                file_path = os.path.join(class_dir, file)
                self.samples.append([file_path, l])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with open(file_path, 'rb') as f:
            features = pickle.load(f)

        # Apply transform if any
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float), label
