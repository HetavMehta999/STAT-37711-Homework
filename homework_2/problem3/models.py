"""
Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models for Problem 3.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn


class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0):
        """
        Hyper-parameters:
            D: total number of random features, D = 2h (must be even)
            sigma: kernel scale; projections are drawn as w_j ~ N(0, I / sigma^2)
            ridge_lambda: ridge regularization strength
            seed: seed for drawing the random projection (w_j, b_j)
        """
        assert D % 2 == 0, "D = 2h must be even (one sin and one cos feature per projection)"
        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.W = None
        self.b = None
        self.coef_ = None

    def _init_projection(self, d: int):
        rng = np.random.default_rng(self.seed)
        h = self.D // 2
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(d, h))
        self.b = rng.uniform(0.0, 2 * np.pi, size=h)

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Map X (n, d) to features (sin_1, cos_1, ..., sin_h, cos_h, 1) of shape (n, D + 1).
        """
        proj = X @ self.W + self.b
        z = np.empty((X.shape[0], self.D))
        z[:, 0::2] = np.sin(proj)
        z[:, 1::2] = np.cos(proj)
        return np.hstack([z, np.ones((X.shape[0], 1))])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Draw the random projection and solve ridge regression in closed form, leaving the intercept unpenalized.
        """
        self._init_projection(X.shape[1])
        phi = self._features(X)
        P = np.eye(phi.shape[1])
        P[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(phi.T @ phi + self.ridge_lambda * P, phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)
        return phi @ self.coef_


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred).reshape(-1) - y) ** 2))


def sweep_rks_model(sweep_sigma: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                    D: int = 1000, ridge_lambda: float = 1e-3, seed: int = 0) -> Tuple:
    """
    Sweep over the kernel scale sigma (D and lambda fixed); return the best model in terms of validation MSE.
    Args:
        sweep_sigma: list of kernel scales sigma to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_sigma, history) where history holds train/val MSE for every sigma
    """
    best_model = None
    best_val_mse = float('inf')
    best_sigma = None
    history = {"sigma": [], "train_mse": [], "val_mse": []}
    for sigma in sweep_sigma:
        model = RandomKitchenSinks(D=D, sigma=sigma, ridge_lambda=ridge_lambda, seed=seed).fit(X_train, y_train)
        train_mse = mse(model.predict(X_train), y_train)
        val_mse = mse(model.predict(X_val), y_val)
        history["sigma"].append(sigma)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_sigma = sigma

    return best_model, best_val_mse, best_sigma, history


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 32,
                 activation: str = "relu", epochs: int = 500, weight_decay: float = 0.0, seed: int = 0, device: str = "cpu"):
        """
        Hyper-parameters:
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
            epochs: number of passes over the training data
        """
        super().__init__()
        self.device = device
        self.lr, self.batch_size, self.epochs, self.weight_decay, self.seed = lr, batch_size, epochs, weight_decay, seed
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            ACTIVATIONS[activation](),
            nn.Linear(d_hidden, d_out),
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        n = X.shape[0]
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                loss = loss_fn(self.net(X[b]), y[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy().reshape(-1)

def mlp_sweep(sweep_values: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              seeds=(0, 1, 2), **mlp_kwargs) -> Tuple:
    """
    Sweep over the hidden dimension (other hyper-parameters fixed via mlp_kwargs). Each width is trained with several
    seeds and scored by the mean validation MSE, since a single small-data NN run is noisy.
    Returns (best_model, best_val_mse, best_width, history); best_model is the best-seed model at the best width.
    """
    best = (None, float('inf'), None)
    history = {"d_hidden": [], "train_mse": [], "val_mse": [], "train_mse_std": [], "val_mse_std": []}
    for width in sweep_values:
        tr, va, models = [], [], []
        for seed in seeds:
            model = MLP(X_train.shape[1], d_hidden=width, seed=seed, **mlp_kwargs).fit(X_train, y_train)
            tr.append(mse(model.predict(X_train), y_train))
            va.append(mse(model.predict(X_val), y_val))
            models.append(model)
        history["d_hidden"].append(width)
        history["train_mse"].append(float(np.mean(tr)))
        history["val_mse"].append(float(np.mean(va)))
        history["train_mse_std"].append(float(np.std(tr)))
        history["val_mse_std"].append(float(np.std(va)))
        print(f"[mlp] d_hidden={width:4d}  train MSE {np.mean(tr):.4f}  val MSE {np.mean(va):.4f} (+/- {np.std(va):.4f})")
        if np.mean(va) < best[1]:
            best = (models[int(np.argmin(va))], float(np.mean(va)), width)
    return (*best, history)
