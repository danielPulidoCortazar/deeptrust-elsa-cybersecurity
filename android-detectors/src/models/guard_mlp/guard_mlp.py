"""
Python file containing the base iiia_mlp model built within the IIIA.
"""
import logging
import os
import random
import sys
import torch
from scipy.sparse import csr_matrix
from torch.nn.functional import sigmoid
from torcheval.metrics import BinaryConfusionMatrix
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from iiia_data import PtDrebinDataset
from models.base.base_drebin import BaseDREBIN
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils._array_api import get_namespace
from sklearn.model_selection import train_test_split
import dill as pkl


class GuardMLP(nn.Module, BaseDREBIN):
    """
    Base class for a multi-layer perceptron model.

    Parameters
    ----------
    """

    def __init__(self):
        """
        Initializes the model.
        """

        # Set seeds
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        nn.Module.__init__(self)
        BaseDREBIN.__init__(self)

        # Set device automatically (cuda, if available, else mps, if available, else cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        # Get the model hyperparameters
        self.batch_size = 32
        self.patience = 3
        self.min_epochs = 3
        self.max_epochs = 10
        self.in_dim = 1461078
        self.out_dim = 1
        self.hidden_sizes = [256, 32, 256]
        self.activation = "leaky_relu"
        self.dropout = 0.7000000000000001
        self.lr_rate= 0.001
        self.optimizer = 'adam'
        self.optimizer_params = {'weight_decay': 0.002463768595899745}
        self.loss = 'bce'
        self.loss_params = {'pos_weight': 8.5}

        # Get the training hyperparameters
        self.train_mode = {
            "adv_train": True,
            "perturbation_scheme": "accumulate",
            "m": 10,
            "classes_to_perturb": [1],
            "delta_type": "discrete",
            "delta_bound": 100,
            "feat_selection": "topk"
        }

        # Get if distillation is enabled
        self.distillation = 0.5

        # Create the linear layers
        ## Create the input layer
        self.input_layer = nn.Linear(self.in_dim, self.hidden_sizes[0]).to(self.device)
        # Print the weights of the input layer
        print(f"Reproducibility check: \n"
                 f"Input layer weights - {self.input_layer.weight}")
        ## Create the hidden layers with dropout
        self.layers = nn.ModuleList()
        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.to(self.device)
        ## Create the output layer
        self.output_layer = nn.Linear(self.hidden_sizes[-1], 1)
        self.output_layer.to(self.device)

        # Create activation function
        if self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif self.activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # Create the optimizer
        if self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_rate, **self.optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        # Create the loss function
        if self.loss == 'bce':

            if self.loss_params.get('pos_weight', None):
                self.loss_params['pos_weight'] = torch.tensor(self.loss_params['pos_weight']).to(self.device)

            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum",
                **self.loss_params)

        else:
            raise ValueError(f"Unknown loss function: {self.loss}")


    def forward(self, x, return_embedding=False):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        return_embedding : bool, optional
            Whether to return the output of the last hidden layer.

        Returns
        -------
        torch.Tensor
            Output tensor.
        torch.Tensor, optional
            Embedding tensor if return_embedding is True.
        """
        x = self.input_layer(x)

        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_fn(x)

        # The embedding is the final activation before the output layer
        embedding = x

        x = self.output_layer(x)

        return (x, embedding) if return_embedding else x


    def load_pt_dataset(self, X: csr_matrix, y = None, distillation=0.0):
        """
        Loads the dataset into a PyTorch dataset.

        Parameters
        ----------
        X : CArray
            Features.
        y : CArray
            Labels.

        Returns
        -------
        DataLoader
            The PyTorch DataLoader.
        DataLoader (optional)
            The PyTorch DataLoader.
        """

        # Create the PyTorch dataset
        self.set = PtDrebinDataset(X, y, distillation)

        # If the labels are provided, split the dataset for training
        if self.set.y is not None:
            # Split in a stratified way the dataset into train and validation (80-20)
            train_idx, val_idx = train_test_split(
                np.arange(len(self.set)), test_size=0.2, stratify=self.set.hard_labels)
            self.valset = torch.utils.data.Subset(self.set, val_idx)
            self.trainset = torch.utils.data.Subset(self.set, train_idx)
            print(f"Reproducibility check: First 10 train split idx -  {train_idx[:10]}")
            print(f"Reproducibility check: First 10 val split idx - {val_idx[:10]}")

            # Create the dataloaders
            self.trainloader = DataLoader(
                self.trainset, batch_size=self.batch_size, shuffle=True,num_workers=0)
            self.valloader = DataLoader(
                self.valset, batch_size=self.batch_size, shuffle=False, num_workers=0)

            return self.trainloader, self.valloader
        else:
            self.dataloader = DataLoader(
                self.set, batch_size=self.batch_size, shuffle=False)

            return self.dataloader, None


    def _compute_metrics(self, loss, tp, fp, tn, fn, i):
        """
        Computes the metrics.

        Parameters
        ----------
        loss : torch.Tensor
            The loss.
        tp : int
            True positives.
        fp : int
            False positives.
        tn : int
            True negatives.
        fn : int
            False negatives.
        i : int
            The index of the batch for the mean loss computation.

        Returns
        -------
        dict
            The metrics.
        """

        acc = np.round((tp + tn) / (tp + fp + tn + fn), 4).item()
        prec = np.round(tp / (tp + fp), 4).item()
        rec = np.round(tp / (tp + fn), 4).item()
        spec = np.round(tn / (tn + fp), 4).item()
        f1_score = np.round(2 * (prec * rec) / (prec + rec), 4).item() if prec + rec > 0 else torch.nan
        mean_loss = np.round(loss.detach().cpu() / (self.batch_size * (i + 1)), 6).item()
        fp_rate = np.round(fp / (fp + tn), 4).item() if fp + tn > 0 else torch.nan
        metrics_dict = {'L': mean_loss, 'A': acc, 'P': prec, 'R': rec, 'S': spec,
                        'F1': f1_score, 'FPR': fp_rate}

        return metrics_dict


    def _fit(self, X, y):
        """
        Fits the model with 'Free' adversarial training algorithm (Shafahi et al., 2019).

        Parameters
        ----------
        X : CArray
            Features.
        y : CArray
            Labels.

        Returns
        -------
        dict
            The training metrics.
        """

        # Load the PyTorch dataset
        self.load_pt_dataset(X, y, self.distillation)

        # Show num of params
        print(f"Number of trainable parameters: "
                 f"{sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

        # Set train mode hyperparameters
        self.perturbation_scheme = self.train_mode['perturbation_scheme']
        self.m = self.train_mode['m']
        self.classes_to_perturb = self.train_mode['classes_to_perturb']
        self.delta_bound = self.train_mode['delta_bound']
        self.delta_type = self.train_mode['delta_type']
        self.feat_selection = self.train_mode['feat_selection']

        # Initialize the confusion matrix and dictionaries
        cm = BinaryConfusionMatrix(threshold=0.5)
        metrics_dict = {}
        train_metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_spec': [], 'val_spec': [],
            'train_f1': [], 'val_f1': [],
            'train_fpr': [], 'val_fpr': []
        }

        # Initialize delta to vector of zeros with values randomly sampled from a uniform distribution
        if self.perturbation_scheme == 'accumulate':
            delta_global = torch.zeros(
                self.batch_size, self.in_dim, requires_grad=False, device='mps')

        for epoch in range(1, (self.max_epochs // self.m) + 1):
            with tqdm(iterable=self.trainloader, total=len(self.trainloader),
                      desc= f'Epoch {epoch}, train', postfix=metrics_dict) as pbar:

                train_loss = 0.0
                self.train()
                for i, (features, labels, hard_labels) in enumerate(self.trainloader):

                    # Get the features and labels
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    if self.perturbation_scheme == 'reset':
                        delta_global = torch.zeros(
                            self.batch_size, self.in_dim, requires_grad=False, device='mps')

                    for _ in range(self.m):

                        perturbed_features = features.detach().clone()

                        indices_to_perturb = torch.tensor(
                            [i for i in range(self.batch_size)
                             if labels[i].item() in self.classes_to_perturb])

                        if len(indices_to_perturb) > 0:

                            if self.delta_type == 'discrete':

                                perturbed_features[indices_to_perturb] = torch.clamp(
                                    perturbed_features[indices_to_perturb] + delta_global[indices_to_perturb],
                                    0, 1)

                            if self.delta_type == 'continuous':
                                perturbed_features[indices_to_perturb] += delta_global[indices_to_perturb]

                        # Forward pass
                        outputs = self.forward(perturbed_features)

                        if self.distillation > 0.0:
                            # Create weighted binary cross entropy loss based on the hard labels
                            self.loss_fn = nn.BCEWithLogitsLoss(
                                reduction="sum",
                                weight=hard_labels.to(self.device) *
                                       self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                        # Compute the loss
                        loss = self.loss_fn(outputs, labels)
                        train_loss += loss

                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()

                        # Update delta
                        if len(indices_to_perturb) > 0:

                            delta = delta_global.detach().clone().requires_grad_(True)
                            perturbed_features = features.detach().clone()

                            if self.delta_type == 'discrete':

                                perturbed_features[indices_to_perturb] = torch.clamp(
                                    perturbed_features[indices_to_perturb] + delta[indices_to_perturb],
                                    0, 1)

                            if self.delta_type == 'continuous':
                                perturbed_features[indices_to_perturb] += delta[indices_to_perturb]


                            # Compute the gradient of the loss with respect to delta
                            outputs = self.forward(perturbed_features)

                            if self.distillation > 0.0:
                                # Create weighted binary cross entropy loss based on the hard labels
                                self.loss_fn = nn.BCEWithLogitsLoss(
                                    reduction="sum",
                                    weight=hard_labels.to(self.device) *
                                           self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                            loss = self.loss_fn(outputs, labels)
                            grad = torch.autograd.grad(loss, delta)[0].detach().cpu()
                            sign_indices = torch.sign(grad)

                            if self.delta_type == 'discrete':

                                if self.feat_selection == 'topk':
                                    # Get k features with the highest gradients
                                    feat_indices = torch.topk(torch.abs(grad), self.delta_bound, dim=1)[1]

                                elif self.feat_selection == 'random':
                                    # Get random k features
                                    feat_indices = torch.randint(0, self.in_dim, (self.batch_size, self.delta_bound))

                                sign_indices = torch.gather(sign_indices, 1, feat_indices)

                                # Update delta_global
                                delta_temp = torch.zeros(self.batch_size, self.in_dim, requires_grad=False,
                                                         device='mps')
                                delta_temp.scatter_(
                                    1, feat_indices.to(self.device), sign_indices.to(self.device))

                                delta_global[indices_to_perturb] += delta_temp[indices_to_perturb]
                                delta_global[indices_to_perturb] = torch.clamp(
                                    delta_global[indices_to_perturb], -1, 1)

                                # For each sample in the batch, scale the number of changes to epsilon
                                for j in indices_to_perturb:
                                    bin_diff = torch.abs(delta_global[j])
                                    diff_indices = torch.nonzero(bin_diff).squeeze()
                                    n_changes = int(bin_diff.sum().item())
                                    if n_changes > self.delta_bound:
                                        indices_perm = torch.randperm(n_changes)
                                        restore_indices = indices_perm[:n_changes - self.delta_bound]
                                        restore_feats = diff_indices[restore_indices]
                                        delta_global[j, restore_feats] = 0

                            elif self.delta_type == 'continuous':
                                # Update delta_global
                                delta_global[indices_to_perturb] += self.delta_bound * sign_indices[indices_to_perturb].to(self.device)

                                # Clip delta_global
                                delta_global[indices_to_perturb] = torch.clamp(
                                    delta_global[indices_to_perturb], -self.delta_bound, self.delta_bound)

                        # Optimize
                        self.optimizer.step()

                        if self.distillation > 0.0:
                            # Compute cm and metrics
                            cm.update(input=sigmoid(outputs).squeeze(),
                                      target=hard_labels.squeeze().to(torch.int64)
                                      )
                        else:
                            # Compute cm and metrics
                            cm.update(input=sigmoid(outputs).squeeze(),
                                      target=labels.squeeze().to(torch.int64)
                                      )

                        tn, fp, fn, tp = cm.compute().reshape(-1)
                        metrics_dict = self._compute_metrics(train_loss, tp, fp, tn, fn, i)

                        # Update the metrics of the progress bar
                        pbar.set_postfix(metrics_dict)

                    # Update the progress bar
                    pbar.update(1)

                # Store the metrics and reset the cm
                train_metrics['train_loss'].append(metrics_dict['L'])
                train_metrics['train_acc'].append(metrics_dict['A'])
                train_metrics['train_prec'].append(metrics_dict['P'])
                train_metrics['train_rec'].append(metrics_dict['R'])
                train_metrics['train_spec'].append(metrics_dict['S'])
                train_metrics['train_f1'].append(metrics_dict['F1'])
                train_metrics['train_fpr'].append(metrics_dict['FPR'])

                print(f"Epoch {epoch}, train metrics: {metrics_dict}")

                # Reset the confusion matrix
                cm.reset()

            with tqdm(iterable=self.valloader, total=len(self.valloader),
                      desc=f'Epoch {epoch}, val', postfix=metrics_dict) as pbar:

                self.eval()
                val_loss = 0.0

                for i, (features, labels, hard_labels) in enumerate(self.valloader):
                    with torch.no_grad():
                        # Get the features and labels
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        # Forward pass
                        outputs = self.forward(features)

                        if self.distillation > 0.0:
                            # Create weighted binary cross entropy loss based on the hard labels
                            self.loss_fn = nn.BCEWithLogitsLoss(
                                reduction="sum",
                                weight=hard_labels.to(self.device) *
                                       self.loss_params['pos_weight'] + (1 - hard_labels.to(self.device)))

                        # Compute the loss
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss

                    if self.distillation > 0.0:
                        # Compute cm and metrics
                        cm.update(input=sigmoid(outputs).squeeze(),
                                  target=hard_labels.squeeze().to(torch.int64)
                                  )
                    else:
                        # Compute cm and metrics
                        cm.update(input=sigmoid(outputs).squeeze(),
                                  target=labels.squeeze().to(torch.int64)
                                  )
                    tn, fp, fn, tp = cm.compute().reshape(-1)
                    metrics_dict = self._compute_metrics(val_loss, tp, fp, tn, fn, i)

                    # Update the progress bar
                    pbar.set_postfix(metrics_dict)
                    pbar.update(1)

                # Store the metrics and reset the cm
                train_metrics['val_loss'].append(metrics_dict['L'])
                train_metrics['val_acc'].append(metrics_dict['A'])
                train_metrics['val_prec'].append(metrics_dict['P'])
                train_metrics['val_rec'].append(metrics_dict['R'])
                train_metrics['val_spec'].append(metrics_dict['S'])
                train_metrics['val_f1'].append(metrics_dict['F1'])
                train_metrics['val_fpr'].append(metrics_dict['FPR'])

                print(f"Epoch {epoch}, val metrics: {metrics_dict}")

            # Reset the confusion matrix
            cm.reset()

        # To avoid conflicts when loading the model
        if self.distillation > 0.0:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum",
                                                **self.loss_params)

        return train_metrics


    def predict(self, features):
        """
        Predicts the labels for the given features.

        Parameters
        ----------
        features : CArray
            Features.

        Returns
        -------
        CArray
            Predicted labels.
        """

        X = self._vectorizer.transform(features)

        if (hasattr(self, 'used_features') and
                self.used_features is not None):
            X = X[:, self.used_features]

        dataloader, _ = self.load_pt_dataset(X)
        xp, _ = get_namespace(X)

        scores = np.zeros(len(self.set))
        self.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                features = batch.to(self.device)
                outputs = self.forward(features)
                probs = sigmoid(outputs).squeeze().cpu().numpy().tolist()
                scores[i * self.batch_size: (i + 1) * self.batch_size] = probs

        indices = xp.astype(scores >= 0.5, dtype=int)

        return indices, scores


    def save(self, vectorizer_path, classifier_path):
        """
        Saves the model.

        Parameters
        ----------
        vectorizer_path : str
            Path to save the vectorizer.
        classifier_path : str
            Path to save the classifier.
        """
        with open(vectorizer_path, "wb") as f:
            pkl.dump(self._vectorizer, f)

        vectorizer = self._vectorizer
        self._vectorizer = None
        torch.save(self.state_dict(), classifier_path)
        self._vectorizer = vectorizer


    @staticmethod
    def load(vectorizer_path, classifier_path):
        """
        Loads the model.

        Parameters
        ----------
        vectorizer_path : str
            Path to load the vectorizer.
        classifier_path : str
            Path to load the classifier.

        Returns
        -------
        MLP
            The loaded model.
        """
        with open(vectorizer_path, "rb") as f:
            vectorizer = pkl.load(f)

        model = GuardMLP()
        model._vectorizer = vectorizer
        model._input_features = (model._vectorizer.get_feature_names_out()
                                .tolist())

        if classifier_path is not None:
            # Set device automatically
            # (cuda, if available, else mps, if available, else cpu)
            device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else
                                  "cpu")
            weights = torch.load(
                classifier_path, map_location=device, weights_only=True)
            model.load_state_dict(weights)

        return model


    def fit(self, features, y, feat_selection=np.empty(0)):
        """
        Parameters
        ----------
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
        y : np.ndarray
            Array of shape (n_samples,) containing the class labels.
        """

        X = self._vectorizer.fit_transform(features)
        self._input_features = (self._vectorizer.get_feature_names_out()
                                .tolist())

        self.used_features = None
        if feat_selection.any():
            self.used_features = feat_selection
            self._input_features = [self._input_features[idx] for idx in self.used_features]
            X = X[:, self.used_features]

        train_metrics = self._fit(X, y)

        return train_metrics