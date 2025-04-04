"""
Python file to define the pytorch Dataset subclass for the DREBIN dataset.
"""
import logging

import numpy as np
import torch
import scipy
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.utils import *

class PtDrebinDataset(Dataset):
    """
    Pytorch Drebin Dataset class for the DREBIN dataset.
    Used for training the nn models with batch methods

    Parameters
    ----------
    X : scipy.sparse.csc
        The feature matrix.
    y : np.array
        The ground truth labels.
    """

    def __init__(self, X:scipy.sparse.csc, y:np.array = None, distillation: float = None):
        self.X = X
        self.y = y
        self.hard_labels = y

        if distillation > 0.0:

            print("Distillation mode activated. Fitting the RF model to smooth the labels.")

            hyperparameters = {
                'n_estimators': 95, 'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2,
                'min_samples_leaf': 1, 'class_weight': None, 'max_samples': None,
                "min_weight_fraction_leaf": 0.0, "max_features": "sqrt", "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0, "bootstrap": False, "oob_score": False, "n_jobs": None,
                "random_state": 0, "verbose": 0, "warm_start": False,
                "ccp_alpha": 0.0, "monotonic_cst": None
            }

            rf = RandomForestClassifier(**hyperparameters)
            rf.fit(X, y)

            # Get the probabilities
            y = rf.predict_proba(X)[:,1]
            # Set the new labels as a soft version of the hard labels
            self.y = distillation * y + (1 - distillation) * self.hard_labels


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple : (features:torch.Tensor, gt_label:torch.Tensor)
            The features and the ground truth label.
        """
        # Convert the sparse row to a dense numpy array
        features = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()
        if self.y is not None:
            if hasattr(self, "hard_labels"):
                gt_label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
                hard_label = torch.tensor(self.hard_labels[idx], dtype=torch.float32).unsqueeze(0)
                return features, gt_label, hard_label
            else:
                gt_label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
                return features, gt_label, None
        else:
            return features