"""
Homework 2, Problem 3: Random Kitchen Sinks (RKS) + ridge vs. a 1-hidden-layer MLP on a 2-D toy dataset.

    (a) visualize the training data
    (b) RKS sweep over the kernel scale sigma (D = 1000 features, lambda = 1e-3 fixed)
    (c) MLP sweep over the hidden width (Adam, lr, batch size, epochs fixed; 3 seeds per width)
    (d) refit the best of each, report validation MSE, write test predictions, plot fitted surfaces

Usage (from homework_2/):
    python problem3/rks_vs_mlp.py
    python problem3/rks_vs_mlp.py --quick

Outputs: rks_test_predictions.csv and mlp_test_predictions.csv (one prediction per row, same order as test.csv,
no header) in problem3/, and figures + metrics.json in problem3/results/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from dataloader import HW2Dataset
from models import MLP, RandomKitchenSinks, mlp_sweep, mse, sweep_rks_model
from plotting import plot_surface_2d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
LIM = (-3, 3)

RKS_D = 1000
RKS_LAMBDA = 1e-3
MLP_KW = dict(lr=3e-3, batch_size=32, activation="relu", epochs=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_num_threads(1)

    p = lambda f: os.path.join(HERE, f)
    d = lambda f: os.path.join(HERE, "..", "data", f)
    ds = HW2Dataset.from_csv(d("train.csv"), d("val.csv"), d("test.csv"))
    Xtr, ytr, Xva, yva, Xte = ds.train.X, ds.train.y, ds.val.X, ds.val.y, ds.test.X

    plot_training_data(Xtr, ytr)

    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    if args.quick:
        sigmas = [0.1, 0.5, 1.0, 3.0]
    _, rks_val, best_sigma, rks_hist = sweep_rks_model(sigmas, Xtr, ytr, Xva, yva, D=RKS_D, ridge_lambda=RKS_LAMBDA)
    for s, a, b in zip(rks_hist["sigma"], rks_hist["train_mse"], rks_hist["val_mse"]):
        print(f"[rks] sigma={s:<5}  train MSE {a:.4f}  val MSE {b:.4f}")
    plot_sweep(rks_hist["sigma"], rks_hist["train_mse"], rks_hist["val_mse"], None, None,
               xlabel="kernel scale $\\sigma$ (log scale)", title=f"RKS (D={RKS_D}, $\\lambda$={RKS_LAMBDA})",
               best=best_sigma, filename="rks_sweep.png")

    widths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    mlp_kw = dict(MLP_KW)
    if args.quick:
        widths, mlp_kw["epochs"] = [4, 32, 128], 30
    _, mlp_val_sweep, best_width, mlp_hist = mlp_sweep(widths, Xtr, ytr, Xva, yva, **mlp_kw)
    plot_sweep(mlp_hist["d_hidden"], mlp_hist["train_mse"], mlp_hist["val_mse"],
               mlp_hist["train_mse_std"], mlp_hist["val_mse_std"],
               xlabel="hidden width (log scale)", title="1-hidden-layer MLP (mean $\\pm$ sd over 3 seeds)",
               best=best_width, filename="mlp_sweep.png")

    rks = RandomKitchenSinks(D=RKS_D, sigma=best_sigma, ridge_lambda=RKS_LAMBDA, seed=0).fit(Xtr, ytr)
    mlp = MLP(Xtr.shape[1], d_hidden=best_width, seed=0, **mlp_kw).fit(Xtr, ytr)
    final = {
        "rks": {"sigma": best_sigma, "D": RKS_D, "lambda": RKS_LAMBDA,
                "train_mse": mse(rks.predict(Xtr), ytr), "val_mse": mse(rks.predict(Xva), yva)},
        "mlp": {"d_hidden": best_width, **mlp_kw, "seed": 0,
                "train_mse": mse(mlp.predict(Xtr), ytr), "val_mse": mse(mlp.predict(Xva), yva)},
        "baseline_predict_train_mean_val_mse": mse(np.full_like(yva, ytr.mean()), yva),
    }
    final["rks"]["val_mse_over_10_feature_seeds"] = [
        mse(RandomKitchenSinks(D=RKS_D, sigma=best_sigma, ridge_lambda=RKS_LAMBDA, seed=s).fit(Xtr, ytr).predict(Xva), yva)
        for s in range(10)]

    np.savetxt(p("rks_test_predictions.csv"), rks.predict(Xte), fmt="%.10f")
    np.savetxt(p("mlp_test_predictions.csv"), mlp.predict(Xte), fmt="%.10f")

    plot_surface_2d(rks.predict, LIM, LIM, title=f"RKS surface ($\\sigma$={best_sigma})",
                    filename=os.path.join(OUT_DIR, "rks_surface.png"))
    plot_surface_2d(mlp, LIM, LIM, title=f"MLP surface (width {best_width})",
                    filename=os.path.join(OUT_DIR, "mlp_surface.png"))
    plt.close("all")
    plot_comparison(Xtr, ytr, rks, mlp, best_sigma, best_width)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"rks_sweep": rks_hist, "mlp_sweep": mlp_hist, "final": final, "quick": args.quick}, f, indent=2)
    print(json.dumps(final, indent=2))


def interpolate_training(Xtr, ytr, grid=200):
    xs = np.linspace(*LIM, grid)
    G1, G2 = np.meshgrid(xs, xs)
    Z = griddata(Xtr, ytr, (G1, G2), method="linear")
    return G1, G2, Z


def plot_training_data(Xtr, ytr):
    fig = plt.figure(figsize=(15, 4.5))
    ax = fig.add_subplot(1, 3, 1)
    sc = ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=18, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="y")
    ax.set(title="Training data (color = y)", xlabel="x1", ylabel="x2", xlim=LIM, ylim=LIM, aspect="equal")

    ax = fig.add_subplot(1, 3, 2)
    G1, G2, Z = interpolate_training(Xtr, ytr)
    cf = ax.contourf(G1, G2, Z, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="y (linear interpolation)")
    ax.set(title="Interpolated training data", xlabel="x1", ylabel="x2", xlim=LIM, ylim=LIM, aspect="equal")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(Xtr[:, 0], Xtr[:, 1], ytr, c=ytr, s=8, cmap="viridis")
    ax.set(title="3-D view", xlabel="x1", ylabel="x2", zlabel="y")
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_data.png"), dpi=200)
    plt.close(fig)


def plot_sweep(x, train, val, train_sd, val_sd, xlabel, title, best, filename):
    """Two panels as required: training MSE and validation MSE vs. the hyper-parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, vals, sd, name in [(axes[0], train, train_sd, "Training"), (axes[1], val, val_sd, "Validation")]:
        if sd is None:
            ax.plot(x, vals, "o-")
        else:
            ax.errorbar(x, vals, yerr=sd, fmt="o-", capsize=3)
        ax.axvline(best, color="red", ls="--", lw=1, label=f"best (val) = {best}")
        ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="MSE", title=f"{name} MSE")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close(fig)


def plot_comparison(Xtr, ytr, rks, mlp, best_sigma, best_width, grid=200):
    xs = np.linspace(*LIM, grid)
    G1, G2 = np.meshgrid(xs, xs)
    Xg = np.stack([G1.ravel(), G2.ravel()], axis=1)
    _, _, Zd = interpolate_training(Xtr, ytr, grid)
    panels = [("Training data (interpolated)", Zd),
              (f"RKS, $\\sigma$={best_sigma}", rks.predict(Xg).reshape(grid, grid)),
              (f"MLP, width {best_width}", mlp.predict(Xg).reshape(grid, grid))]
    vmin, vmax = np.nanmin(ytr), np.nanmax(ytr)
    levels = np.linspace(vmin, vmax, 31)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, Z) in zip(axes, panels):
        cf = ax.contourf(G1, G2, np.clip(Z, vmin, vmax), levels=levels, cmap="viridis")
        ax.set(title=title, xlabel="x1", ylabel="x2", aspect="equal")
    fig.colorbar(cf, ax=axes, label="y / prediction", shrink=0.9)
    fig.savefig(os.path.join(OUT_DIR, "surface_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
