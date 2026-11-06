"""
Homework 2, Problem 1: heteroscedastic Gaussian MLP for NYC taxi trip durations.

Fits an MLP g: R^7 -> R^2 with link (mu, sigma) = (g_1, softplus(g_2) + eps), trained by
Gaussian negative log-likelihood, and compares it to baselines. Then predicts
P(trip_duration < 45 min) = Phi((45 - mu) / sigma) on the held-out test set and scores
those probabilities (Brier, log-loss, accuracy at 0.5).

Usage (from homework_2/):
    python problem1/taxi_mlp.py
    python problem1/taxi_mlp.py --quick

Outputs go to problem1/results/.
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "taxi_trips.parquet")
OUT_DIR = os.path.join(HERE, "results")
TARGET = "trip_duration"
THRESHOLD = 45.0
SIGMA_FLOOR = 1e-3
SEED = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(quick: bool = False):
    """70/15/15 random train/val/test split with a fixed seed."""
    df = pd.read_parquet(DATA_PATH)
    if quick:
        df = df.sample(n=30_000, random_state=SEED)
    features = [c for c in df.columns if c != TARGET]
    X = df[features].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)

    perm = np.random.default_rng(SEED).permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return {k: (X[v], y[v]) for k, v in idx.items()}, features


def gaussian_link(raw: torch.Tensor):
    """Link function: identity for the mean, softplus (+ small floor) for the std."""
    mu = raw[:, 0]
    sigma = F.softplus(raw[:, 1]) + SIGMA_FLOOR
    return mu, sigma


def gaussian_nll(mu, sigma, y):
    """Mean per-example negative log-likelihood of y under N(mu, sigma^2)."""
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * math.log(2 * math.pi)).mean()


class GaussianMLP(nn.Module):
    """MLP g: R^p -> R^2, mapped to (mu, sigma) by `gaussian_link`."""

    def __init__(self, d_in: int, hidden=(256, 256, 128)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return gaussian_link(self.net(x))


class LinearGaussian(nn.Module):
    """Baseline: mu and log-sigma both linear in x (heteroscedastic linear model)."""

    def __init__(self, d_in: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 2)

    def forward(self, x):
        return gaussian_link(self.lin(x))


class HomoscedasticMLP(nn.Module):
    """Baseline/ablation: same MLP for mu, but a single learned sigma shared by all inputs."""

    def __init__(self, d_in: int, hidden=(256, 256, 128)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.raw_sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = self.net(x).squeeze(-1)
        sigma = (F.softplus(self.raw_sigma) + SIGMA_FLOOR).expand_as(mu)
        return mu, sigma


def train_nll(model, train, val, *, epochs, lr, batch_size, patience, weight_decay=0.0, name=""):
    """Minibatch Adam on Gaussian NLL with early stopping on validation NLL."""
    Xtr, ytr = train
    Xva, yva = val
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad = float("inf"), None, 0
    history = {"train_nll": [], "val_nll": []}
    n = len(Xtr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            mu, sigma = model(Xtr[b])
            loss = gaussian_nll(mu, sigma, ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val_nll = gaussian_nll(*model(Xva), yva).item()
        sched.step(val_nll)
        history["train_nll"].append(total / n)
        history["val_nll"].append(val_nll)
        print(f"[{name}] epoch {epoch + 1:3d}  train NLL {total / n:.4f}  val NLL {val_nll:.4f}")
        if val_nll < best - 1e-4:
            best, bad = val_nll, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return history


def regression_metrics(mu, sigma, y):
    """Metrics in original units (minutes). NLL here is per-trip, in nats, for y in minutes."""
    nll = np.mean(np.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2 + 0.5 * np.log(2 * np.pi))
    return {
        "nll": float(nll),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mae": float(np.mean(np.abs(y - mu))),
    }


def prob_under(mu, sigma, threshold=THRESHOLD):
    """P(Y < threshold | X) under Y|X ~ N(mu, sigma^2)."""
    return norm.cdf((threshold - mu) / sigma)


def probability_metrics(p, y, threshold=THRESHOLD):
    label = (y < threshold).astype(np.float64)
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "brier": float(np.mean((p - label) ** 2)),
        "log_loss": float(-np.mean(label * np.log(p_clip) + (1 - label) * np.log(1 - p_clip))),
        "accuracy@0.5": float(np.mean((p >= 0.5) == label)),
    }


def predict(model, X, y_mean, y_std):
    with torch.no_grad():
        mu, sigma = model(X)
    return mu.numpy() * y_std + y_mean, sigma.numpy() * y_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny subsample + few epochs (smoke test)")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    epochs = 3 if args.quick else args.epochs

    set_seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(OUT_DIR, exist_ok=True)

    splits, features = load_splits(args.quick)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
    print(f"features: {features}")
    print(f"n_train={len(ytr)}  n_val={len(yva)}  n_test={len(yte)}")
    print(f"share of test trips < {THRESHOLD:.0f} min: {np.mean(yte < THRESHOLD):.4f}")

    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    t = lambda a: torch.from_numpy(a)
    tr = (t(Xtr), t((ytr - y_mean) / y_std))
    va = (t(Xva), t((yva - y_mean) / y_std))
    Xte_t = t(Xte)

    results, histories, preds = {}, {}, {}

    mu0 = np.full_like(yte, ytr.mean(), dtype=np.float64)
    sd0 = np.full_like(yte, ytr.std(), dtype=np.float64)
    preds["constant_gaussian"] = (mu0, sd0)

    base_rate = float(np.mean(ytr < THRESHOLD))

    common = dict(epochs=epochs, batch_size=1024, patience=6)
    specs = {
        "linear_gaussian": (LinearGaussian(len(features)), dict(lr=1e-2)),
        "homoscedastic_mlp": (HomoscedasticMLP(len(features)), dict(lr=1e-3, weight_decay=1e-5)),
        "heteroscedastic_mlp": (GaussianMLP(len(features)), dict(lr=1e-3, weight_decay=1e-5)),
    }
    for name, (model, kw) in specs.items():
        set_seed(SEED)
        start = time.time()
        histories[name] = train_nll(model, tr, va, name=name, **common, **kw)
        print(f"[{name}] trained in {time.time() - start:.1f}s")
        preds[name] = tuple(a.astype(np.float64) for a in predict(model, Xte_t, y_mean, y_std))
        if name == "heteroscedastic_mlp":
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "heteroscedastic_mlp.pt"))

    for name, (mu, sd) in preds.items():
        results[name] = {**regression_metrics(mu, sd, yte), **probability_metrics(prob_under(mu, sd), yte)}
    results["constant_base_rate"] = probability_metrics(np.full_like(yte, base_rate, dtype=np.float64), yte)

    meta = {
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "test_share_under_45": float(np.mean(yte < THRESHOLD)),
        "train_base_rate_under_45": base_rate,
        "epochs_run": {k: len(v["val_nll"]) for k, v in histories.items()},
        "quick": args.quick,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "test_metrics": results}, f, indent=2)

    write_latex_table(results, os.path.join(OUT_DIR, "metrics_table.tex"))
    make_figures(histories, preds, yte, OUT_DIR)

    print("\nTest-set metrics:")
    print(pd.DataFrame(results).T.round(4).to_string())


def write_latex_table(results, path):
    names = {
        "constant_base_rate": "Constant base rate",
        "constant_gaussian": "Constant Gaussian",
        "linear_gaussian": "Linear (heterosc.)",
        "homoscedastic_mlp": "MLP, constant $\\sigma$",
        "heteroscedastic_mlp": "\\textbf{MLP, $f_\\mu$ and $f_\\sigma$}",
    }
    fmt = lambda r, k, d: f"{r[k]:.{d}f}" if k in r else "--"
    rows = []
    for key, label in names.items():
        r = results[key]
        rows.append(" & ".join([label, fmt(r, "nll", 3), fmt(r, "rmse", 2), fmt(r, "mae", 2),
                                fmt(r, "brier", 5), fmt(r, "log_loss", 4), fmt(r, "accuracy@0.5", 4)]) + r" \\")
    header = [
        r"\begin{tabular}{lccc|ccc}", r"\toprule",
        r" & \multicolumn{3}{c|}{Regression, part (c)} & \multicolumn{3}{c}{$P(\text{trip}<45\text{ min})$, part (e)} \\",
        r"Model & NLL & RMSE & MAE & Brier & Log-loss & Accuracy \\",
        r"\midrule",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def make_figures(histories, preds, yte, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in histories.items():
        ax.plot(h["val_nll"], label=f"{name} (val)")
        ax.plot(h["train_nll"], "--", alpha=0.6, label=f"{name} (train)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian NLL (standardized target)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=200)
    plt.close(fig)

    mu, sd = preds["heteroscedastic_mlp"]

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(yte), size=min(20_000, len(yte)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(mu[sub], yte[sub], s=1, alpha=0.2)
    lim = [0, 80]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(xlim=lim, ylim=lim, xlabel="predicted mean $f_\\mu(x)$ (min)", ylabel="actual duration (min)")
    bins = np.quantile(sd, np.linspace(0, 1, 21))
    which = np.clip(np.digitize(sd, bins) - 1, 0, 19)
    pred_sd = [sd[which == b].mean() for b in range(20)]
    emp_sd = [np.sqrt(np.mean((yte - mu)[which == b] ** 2)) for b in range(20)]
    axes[1].plot(pred_sd, emp_sd, "o-")
    m = max(max(pred_sd), max(emp_sd))
    axes[1].plot([0, m], [0, m], "k--", lw=1)
    axes[1].set(xlabel="predicted $f_\\sigma(x)$ (min), binned", ylabel="empirical RMS residual in bin (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mean_and_sigma_fit.png"), dpi=200)
    plt.close(fig)

    label = (yte < THRESHOLD).astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in ["linear_gaussian", "homoscedastic_mlp", "heteroscedastic_mlp"]:
        p = prob_under(*preds[name])
        edges = np.array([0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0 + 1e-9])
        idx = np.digitize(p, edges) - 1
        xs, ys = [], []
        for b in range(len(edges) - 1):
            if (idx == b).sum() >= 50:
                xs.append(p[idx == b].mean())
                ys.append(label[idx == b].mean())
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="predicted P(trip < 45 min)", ylabel="observed frequency", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability_p45.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
