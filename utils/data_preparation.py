import os
from sklearn.model_selection import StratifiedKFold, train_test_split
import re
import numpy as np


def cross_validation_indices(data_path, num_folds):
    """
    Generates indices for cross-validation folds.

    Parameters:
    - data_path (str): Path to the dataset.
    - num_folds (int): Number of folds for cross-validation.

    Returns:
    - A tuple of indices for training, validation, and testing splits for each class.
    """
    img_list = os.listdir(data_path)

    cls_healthy = set()
    cls_gl = set()

    for file in img_list:
        if file.endswith(".npy"):
            match = re.search(r'-(\d+)-', file)
            img_id = match.group(1)
            if 'Normal' in file:
                cls_healthy.add(img_id)
            elif 'POAG' in file:
                cls_gl.add(img_id)

    cls_healthy = list(cls_healthy)
    cls_gl = list(cls_gl)

    img_ids = np.array(cls_healthy + cls_gl)
    labels = np.array([0] * len(cls_healthy) + [1] * len(cls_gl))  # 0 for healthy, 1 for glaucoma

    val_fraction = 0.1

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    index_split_train = {}
    index_split_test = {}
    index_split_val = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(img_ids, labels)):
        print(f'Preparing subset, fold {fold}')
        train_img_ids, test_img_ids = img_ids[train_idx], img_ids[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # Split the training set to get 10% validation set
        train_img_ids, val_img_ids, train_labels, val_labels = train_test_split(
            train_img_ids, train_labels, test_size=val_fraction, stratify=train_labels, random_state=42)

        index_split_train[f'fold{fold}'] = [train_img_ids, train_labels]
        index_split_test[f'fold{fold}'] = [test_img_ids, test_labels]
        index_split_val[f'fold{fold}'] = [val_img_ids, val_labels]

    return index_split_train, index_split_test, index_split_val
