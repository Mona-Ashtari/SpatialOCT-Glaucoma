"""
Author: Mona Ashtari
Description: This script conducts cross-validation using a GRU-based RNN model on OCT image features
for Glaucoma detection.

Project: SpatialOCT-Glaucoma
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
from torch.optim import lr_scheduler
import numpy as np
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import roc_curve, auc
import pickle
import pandas as pd

from models.RNN_model import GRU_model
from utils.feature_dataset import FeatureDataset
from utils.train import train_model
from utils.evaluate import evaluate_model, montecarlo_uncertainty
from utils.utils import calculate_sample_weights
from utils.data_preparation import cross_validation_indices


def main():

    # Set hyperparameters
    num_epochs = 2
    batch_size = 32
    num_classes = 2
    learning_rate = 0.0001
    alpha = 0.3
    gamma = 2
    num_folds = 5
    hidden_dim = 256

    data_path = 'OCT_features'
    ckpn_path = ''
    mont_uncertainty = False

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split data for cross-validation
    # indexes = cross_validation_indices(data_path, num_folds)
    with open('split_index.pickle', 'rb') as f:
        indexes = pickle.load(f)

    # Initialize results dictionary
    results = {}

    # Cross-validation loop
    for fold in range(num_folds):

        train_losses = []
        train_accs = []

        val_losses = []
        val_accs = []
        val_confmat_log = []
        val_auc = []

        min_loss = float('inf')
        patience = 6  # Number of epochs to wait for improvement
        counter = 0

        # Instantiate the model, loss function, and optimizer
        gru_model = GRU_model(input_dim=1024,
                              hidden_dim=hidden_dim,
                              num_classes=num_classes
                              ).to(device)

        optimizer = optim.Adam(gru_model.parameters(), lr=learning_rate)

        if len(ckpn_path) > 0:
            model_checkpoint = torch.load(ckpn_path, map_location=torch.device(device))
            gru_model.load_state_dict(model_checkpoint['model_state_dict'])

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # Decay every 5 epochs

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = sigmoid_focal_loss

        # Dataset folding
        dataset_train = FeatureDataset(data_path, [indexes[0][fold], indexes[3][fold]])
        dataset_val = FeatureDataset(data_path, [indexes[1][fold], indexes[4][fold]])
        dataset_test = FeatureDataset(data_path, [indexes[2][fold], indexes[5][fold]])

        # train data loader and sampler
        sample_weights = calculate_sample_weights(dataset_train)
        # weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        weighted_sampler = WeightedRandomSampler(sample_weights,
                                                 num_samples=int(sum(np.hstack(sample_weights) >= 2) * 2),
                                                 replacement=False)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=weighted_sampler
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=True
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True
        )

        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(gru_model, data_loader_train,
                                                optimizer, criterion, device, alpha, gamma)

            print(f"Fold: {fold} \\ Epoch {epoch + 1}/{num_epochs}:\\ "
                  f"Train Accuracy: {train_acc:.4f}\\ "
                  f"Train Loss: {train_loss:.4f}")

            val_loss, val_acc, confmat_val, vep_probs, vep_labels = evaluate_model(gru_model, data_loader_val,
                                                                                   criterion, device, alpha, gamma)

            vfpr, vtpr, _ = roc_curve(vep_labels, vep_probs)
            vroc_auc = auc(vfpr, vtpr)

            print(f"Fold: {fold} \\ Epoch {epoch + 1}/{num_epochs}:\\ "
                  f"Val Accuracy: {val_acc:.4f}\\ "
                  f"Val Loss: {val_loss:.4f}\\"
                  f"Val AUC: {vroc_auc:.4f}")

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_confmat_log.append(confmat_val.cpu())
            val_auc.append(vroc_auc)

            scheduler.step()

            # Early stopping logic
            if val_losses[-1] < min_loss:
                min_loss = val_losses[-1]
                counter = 0  # Reset counter when a new minimum is found
            else:
                counter += 1  # Increment counter when no improvement is found

            if counter >= patience:
                print('Early stopping triggered.')
                break  # Exit the loop

        torch.save({
            'model_state_dict': gru_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f'gru_model_checkpoint_fold{fold}.ckpt')

        test_loss, test_acc, confmat_test, ep_probs, ep_labels = evaluate_model(gru_model,
                                                                                data_loader_test,
                                                                                criterion, device, alpha, gamma)

        fpr, tpr, _ = roc_curve(ep_labels, ep_probs)
        roc_auc = auc(fpr, tpr)

        print(f"Test Accuracy: {test_acc:.4f}\\ "
              f"Test Loss: {test_loss:.4f}.\\"
              f"Test Confusion matrix: {confmat_test}")

        print(f'train_losses = {train_losses}')
        print(f'val_losses = {val_losses}\n')

        print(f'fpr= {list(fpr)}\ntpr= {list(tpr)}\nroc_auc= {roc_auc}\n')

        if mont_uncertainty:
            mont_result = montecarlo_uncertainty(gru_model, data_loader_test, device, num_runs=100)
            results[f'montecarlo_fold{fold}'] = mont_result[0]

            print(f'montecarlo_accs = {mont_result[0] / len(dataset_test)}\n'
                  f'montecarlo_mean = {mont_result[1] / len(dataset_test)}\n'
                  f'montecarlo_std = {mont_result[2] / len(dataset_test)}\n')

        results[f'train_acc_fold{fold}'] = train_accs
        results[f'train_loss_fold{fold}'] = train_losses

        results[f'val_acc_fold{fold}'] = val_accs
        results[f'val_loss_fold{fold}'] = val_losses
        results[f'val_conf_fold{fold}'] = val_confmat_log
        results[f'val_auc_fold{fold}'] = val_auc

        results[f'test_acc_fold{fold}'] = test_acc
        results[f'test_loss_fold{fold}'] = test_loss
        results[f'test_conf_fold{fold}'] = confmat_test.cpu()
        results[f'test_auc_fold{fold}'] = roc_auc
        results[f'test_auc_data_fold{fold}'] = [fpr, tpr]

        # plt.figure()
        # plt.plot(train_losses, label='Train')
        # plt.plot(val_losses, label='Validation')
        # plt.legend()
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss')
        # plt.show()

        # # Plot ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # plt.show()

    with open('results5folds.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

