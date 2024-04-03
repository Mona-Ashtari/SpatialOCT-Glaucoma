import os
from sklearn.model_selection import KFold


def cross_validation_indices(data_path, num_folds):
    """
    Generates indices for cross-validation folds.

    Parameters:
    - data_path (str): Path to the dataset.
    - num_folds (int): Number of folds for cross-validation.

    Returns:
    - A tuple of indices for training, validation, and testing splits for each class.
    """

    class0_index_split_train = []
    class0_index_split_test = []
    class0_index_split_val = []

    class1_index_split_train = []
    class1_index_split_test = []
    class1_index_split_val = []

    val_fraction = 0.1

    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in kf.split(os.listdir(os.path.join(data_path, 'class0'))):
        val_size = int(len(train_index) * val_fraction)
        val_indices = np.random.choice(train_index, size=val_size, replace=False)
        train_indices = [i for i in train_index if i not in val_indices]

        class0_index_split_train.append(train_indices)
        class0_index_split_val.append(val_indices)
        class0_index_split_test.append(test_index)

    for train_index, test_index in kf.split(os.listdir(os.path.join(data_path, 'class1'))):
        val_size = int(len(train_index) * val_fraction)
        val_indices = np.random.choice(train_index, size=val_size, replace=False)
        train_indices = [i for i in train_index if i not in val_indices]

        class1_index_split_train.append(train_indices)
        class1_index_split_val.append(val_indices)
        class1_index_split_test.append(test_index)

    return class0_index_split_train, class0_index_split_val, class0_index_split_test\
        , class1_index_split_train, class1_index_split_val, class1_index_split_test