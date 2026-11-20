"""
Homework 3, Problem 2(c): hyperparameter selection with a train / validation / test split.

White wine quality (regression, MSE loss). Data are split at random into 70% train, 10% validation
and 20% test. For two model classes (RandomForestRegressor and KNeighborsRegressor) we run a
grid search with more parameter settings than there are validation points. Each setting is fit on
train, the best is chosen by validation MSE, and the chosen model is scored on test.

We also record the test MSE of *every* setting so we can compare validation and test risk across
the whole grid. A repeated-split experiment (kNN only, because it is cheap) measures the gap
between the selected model's validation MSE and its test MSE over many random splits.

Usage (from homework_3/):
    python problem2/wine_hparam_search.py
    python problem2/wine_hparam_search.py --quick
    python problem2/wine_hparam_search.py --plots-only   # redraw figures from saved results

Outputs go to problem2/results/.
"""
import argparse
import itertools
import json
import os
import time

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import plot_style as ps  # noqa: E402

ps.apply()
DATA_PATH = os.path.join(HERE, "..", "data", "winequality-white.csv")
OUT_DIR = os.path.join(HERE, "results")
TARGET = "quality"
SEED = 0


# ----------------------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=None, engine="python")  # UCI file is ';'-separated
    X = df.drop(columns=TARGET).to_numpy(float)
    y = df[TARGET].to_numpy(float)
    return X, y


def split(n, seed):
    """Random 70 / 10 / 20 split of indices."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(round(0.70 * n)), int(round(0.10 * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


# ----------------------------------------------------------------------------- grids
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def rf_grid(quick=False):
    g = dict(
        n_estimators=[50, 100, 200, 400],
        max_depth=[None, 5, 10, 15, 20, 30],
        max_features=[0.2, 0.33, 0.5, 0.75, 1.0],
        min_samples_leaf=[1, 2, 4, 8, 16],
    )
    if quick:
        g = dict(n_estimators=[50], max_depth=[None, 10], max_features=[0.33, 1.0], min_samples_leaf=[1, 4])
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def knn_grid(quick=False):
    g = dict(
        n_neighbors=list(range(1, 51)),
        weights=["uniform", "distance"],
        p=[1, 2, 3],
        scaler=list(SCALERS),
    )
    if quick:
        g = dict(n_neighbors=[1, 5, 15, 30], weights=["uniform", "distance"], p=[1, 2], scaler=["standard"])
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def make_rf(params):
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)


def make_knn(params):
    params = dict(params)
    scaler = SCALERS[params.pop("scaler")]()
    return make_pipeline(scaler, KNeighborsRegressor(**params))


MODELS = {
    "RandomForest": (make_rf, rf_grid),
    "KNN": (make_knn, knn_grid),
}


# ----------------------------------------------------------------------------- search
def grid_search(make, grid, X, y, tr, va, te, label=""):
    """Fit every setting on train; record validation and test MSE for each."""
    rows = []
    t0 = time.time()
    for i, params in enumerate(grid):
        model = make(params).fit(X[tr], y[tr])
        rows.append(dict(params=params,
                         train_mse=mse(y[tr], model.predict(X[tr])),
                         val_mse=mse(y[va], model.predict(X[va])),
                         test_mse=mse(y[te], model.predict(X[te]))))
        if label and (i + 1) % 100 == 0:
            print(f"  [{label}] {i + 1}/{len(grid)} settings  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def _knn_predict(nbr_y, nbr_d, k, weights):
    """Exact KNeighborsRegressor prediction from precomputed (sorted) neighbor targets/distances."""
    yk, dk = nbr_y[:, :k], nbr_d[:, :k]
    if weights == "uniform":
        return yk.mean(axis=1)
    zero = dk == 0
    with np.errstate(divide="ignore"):
        w = np.where(zero, 0.0, 1.0 / dk)
    # sklearn rule: if a query has a zero-distance neighbor, only those neighbors get (equal) weight
    has_zero = zero.any(axis=1)
    w[has_zero] = zero[has_zero].astype(float)
    return (w * yk).sum(axis=1) / w.sum(axis=1)


def knn_grid_search(grid, X, y, tr, va, te, label=""):
    """Same output as grid_search(make_knn, ...), but neighbors are found once per (scaler, p) and every
    (n_neighbors, weights) is read off the first k columns. Results match sklearn exactly."""
    kmax = max(p["n_neighbors"] for p in grid)
    cache = {}
    for sc, p in sorted({(g["scaler"], g["p"]) for g in grid}):
        scaler = SCALERS[sc]().fit(X[tr])
        nn = KNeighborsRegressor(n_neighbors=kmax, p=p).fit(scaler.transform(X[tr]), y[tr])
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            d, ind = nn.kneighbors(scaler.transform(X[idx]))
            cache[(sc, p, name)] = (y[tr][ind], d)
    rows = []
    for params in grid:
        r = dict(params=params)
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            ny, nd = cache[(params["scaler"], params["p"], name)]
            r[f"{name}_mse"] = mse(y[idx], _knn_predict(ny, nd, params["n_neighbors"], params["weights"]))
        rows.append(r)
    if label:
        print(f"  [{label}] {len(grid)}/{len(grid)} settings", flush=True)
    return rows


def summarize(rows):
    val = np.array([r["val_mse"] for r in rows])
    test = np.array([r["test_mse"] for r in rows])
    best = int(np.argmin(val))
    oracle = int(np.argmin(test))
    return dict(
        n_settings=len(rows),
        best_params=rows[best]["params"],
        best_train_mse=rows[best]["train_mse"],
        best_val_mse=float(val[best]),
        best_test_mse=float(test[best]),
        optimism=float(test[best] - val[best]),
        oracle_test_mse=float(test[oracle]),
        oracle_params=rows[oracle]["params"],
        corr_val_test=float(np.corrcoef(val, test)[0, 1]),
        median_val_mse=float(np.median(val)),
        median_test_mse=float(np.median(test)),
        mean_val_minus_test=float(np.mean(val - test)),
    )


def repeated_split_knn(X, y, n_seeds, quick):
    """Over many random splits: validation MSE of the selected kNN vs. its test MSE."""
    grid = knn_grid(quick)
    out = []
    for s in range(n_seeds):
        tr, va, te = split(len(y), seed=1000 + s)
        rows = knn_grid_search(grid, X, y, tr, va, te)
        sm = summarize(rows)
        out.append(dict(seed=1000 + s, val=sm["best_val_mse"], test=sm["best_test_mse"],
                        median_val_all=sm["median_val_mse"], median_test_all=sm["median_test_mse"]))
        print(f"  [repeat] split {s + 1}/{n_seeds}: val={sm['best_val_mse']:.3f} test={sm['best_test_mse']:.3f}",
              flush=True)
    return out


# ----------------------------------------------------------------------------- outputs
def fmt_params(p):
    short = {"n_estimators": "trees", "max_depth": "depth", "max_features": "max\\_feat",
             "min_samples_leaf": "min\\_leaf", "n_neighbors": "$k$", "weights": "weights",
             "p": "$p$", "scaler": "scaler"}
    return ", ".join(f"{short.get(k, k)}={v}" for k, v in p.items())


def write_table(summaries, baseline, path):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & \# settings & Train MSE & Val MSE & Test MSE & Test $-$ Val \\", r"\midrule"]
    for name, s in summaries.items():
        lines.append(f"{name} & {s['n_settings']} & {s['best_train_mse']:.3f} & {s['best_val_mse']:.3f} & "
                     f"{s['best_test_mse']:.3f} & {s['optimism']:+.3f} \\\\")
    lines.append(f"Constant (train mean) & 1 & {baseline['train']:.3f} & {baseline['val']:.3f} & "
                 f"{baseline['test']:.3f} & {baseline['test'] - baseline['val']:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_params_table(summaries, path):
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Model & Selected hyperparameters \\", r"\midrule"]
    for name, s in summaries.items():
        lines.append(f"{name} & {fmt_params(s['best_params'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _equal_axes(ax, val, test):
    lo, hi = min(val.min(), test.min()), max(val.max(), test.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ps.MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(hi, hi + pad * 0.2, "test = val", color=ps.MUTED, fontsize=9, ha="right", va="bottom", rotation=45,
            rotation_mode="anchor")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")


def _mark_selected(ax, df):
    b = df.val_mse.idxmin()
    ax.scatter(df.val_mse[b], df.test_mse[b], s=150, marker="*", color=ps.INK, edgecolors="white",
               linewidths=1.2, zorder=4, label="selected on validation")
    ax.annotate(f"val {df.val_mse[b]:.3f}\ntest {df.test_mse[b]:.3f}", (df.val_mse[b], df.test_mse[b]),
                xytext=(18, -30), textcoords="offset points", fontsize=10, color=ps.INK,
                arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))


def plot_val_vs_test(grids, path):
    """grids: {'RandomForest': df, 'KNN': df} with val_mse/test_mse and hyperparameter columns."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0))
    # Random forest: colour by max_depth (ordered -> one-hue ramp, deeper = darker)
    rf = grids["RandomForest"].copy()
    rf["depth"] = rf.max_depth.fillna(np.inf)
    depths = sorted(rf.depth.unique())
    for d, c in zip(depths, ps.BLUE_RAMP):
        m = rf.depth == d
        ax_label = "None" if np.isinf(d) else str(int(d))
        a1.scatter(rf.val_mse[m], rf.test_mse[m], s=16, color=c, alpha=0.85, linewidths=0, zorder=2,
                   label=f"depth {ax_label}")
    _equal_axes(a1, rf.val_mse.values, rf.test_mse.values)
    _mark_selected(a1, rf)
    a1.set_title(f"Random forest: {len(rf)} settings, $r$ = {np.corrcoef(rf.val_mse, rf.test_mse)[0, 1]:.3f}")
    a1.legend(loc="upper left", title="max_depth", title_fontsize=10, handletextpad=0.2, markerscale=1.4)

    # kNN: colour by weighting scheme (categorical)
    kn = grids["KNN"]
    for w, c in (("distance", ps.BLUE), ("uniform", ps.ORANGE)):
        m = kn.weights == w
        a2.scatter(kn.val_mse[m], kn.test_mse[m], s=14, color=c, alpha=0.6, linewidths=0, zorder=2,
                   label=f"{w} weights")
    _equal_axes(a2, kn.val_mse.values, kn.test_mse.values)
    _mark_selected(a2, kn)
    a2.set_title(f"$k$-NN: {len(kn)} settings, $r$ = {np.corrcoef(kn.val_mse, kn.test_mse)[0, 1]:.3f}")
    a2.legend(loc="upper left", markerscale=1.6, handletextpad=0.2)

    for ax in (a1, a2):
        ax.set_xlabel("Validation MSE")
        ax.set_ylabel("Test MSE")
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)


def plot_repeats(reps, path):
    val = np.array([r["val"] for r in reps])
    test = np.array([r["test"] for r in reps])
    gap = test - val
    order = np.argsort(gap)
    x = np.arange(1, len(reps) + 1)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axhline(0, color=ps.MUTED, lw=1)
    colors = np.where(gap[order] > 0, ps.ORANGE, ps.BLUE)
    ax.bar(x, gap[order], width=0.7, color=colors, zorder=2)
    ax.axhline(gap.mean(), color=ps.INK, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.text(0.6, gap.mean() + 0.004, f"mean gap {gap.mean():+.3f}", color=ps.INK, fontsize=10, va="bottom")
    ax.set_xlim(0.3, len(reps) + 0.7)
    ax.set_xticks([])
    ax.set_xlabel(f"Random 70/10/20 split (sorted); test > validation in {int((gap > 0).sum())} of {len(reps)}")
    ax.set_ylabel("Test MSE $-$ validation MSE")
    ax.set_title("$k$-NN, selected model: test MSE minus validation MSE")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_grids():
    return {name: pd.read_csv(os.path.join(OUT_DIR, f"grid_{name.lower()}.csv")) for name in MODELS}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    ap.add_argument("--repeats", type=int, default=30, help="random splits for the kNN optimism study")
    ap.add_argument("--plots-only", action="store_true", help="redraw figures from saved results (no fitting)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.plots_only:
        with open(os.path.join(OUT_DIR, "metrics.json")) as f:
            reps = json.load(f)["knn_repeated_splits"]["splits"]
        plot_val_vs_test(load_grids(), os.path.join(OUT_DIR, "val_vs_test.png"))
        plot_repeats(reps, os.path.join(OUT_DIR, "knn_repeated_splits.png"))
        return

    X, y = load_data()
    tr, va, te = split(len(y), SEED)
    print(f"n = {len(y)}: train {len(tr)}, val {len(va)}, test {len(te)}")

    ybar = y[tr].mean()
    baseline = {k: mse(y[idx], np.full(len(idx), ybar)) for k, idx in [("train", tr), ("val", va), ("test", te)]}

    all_rows, summaries = {}, {}
    for name, (make, grid_fn) in MODELS.items():
        grid = grid_fn(args.quick)
        print(f"{name}: {len(grid)} settings (validation set has {len(va)} points)")
        t0 = time.time()
        if name == "KNN":
            rows = knn_grid_search(grid, X, y, tr, va, te, label=name)
        else:
            rows = grid_search(make, grid, X, y, tr, va, te, label=name)
        all_rows[name] = rows
        summaries[name] = summarize(rows)
        s = summaries[name]
        print(f"  done in {time.time() - t0:.0f}s. best val {s['best_val_mse']:.4f} -> test {s['best_test_mse']:.4f}"
              f"  params {s['best_params']}")

    reps = repeated_split_knn(X, y, 3 if args.quick else args.repeats, args.quick)

    write_table(summaries, baseline, os.path.join(OUT_DIR, "metrics_table.tex"))
    write_params_table(summaries, os.path.join(OUT_DIR, "params_table.tex"))
    for name, rows in all_rows.items():
        pd.DataFrame([{**r["params"], **{k: r[k] for k in ("train_mse", "val_mse", "test_mse")}} for r in rows]) \
            .to_csv(os.path.join(OUT_DIR, f"grid_{name.lower()}.csv"), index=False)
    plot_val_vs_test(load_grids(), os.path.join(OUT_DIR, "val_vs_test.png"))
    plot_repeats(reps, os.path.join(OUT_DIR, "knn_repeated_splits.png"))

    rv = np.array([r["val"] for r in reps]); rt = np.array([r["test"] for r in reps])
    metrics = dict(
        n=len(y), n_train=len(tr), n_val=len(va), n_test=len(te), seed=SEED,
        baseline_constant=baseline, models=summaries,
        knn_repeated_splits=dict(n_splits=len(reps), mean_val=float(rv.mean()), mean_test=float(rt.mean()),
                                 mean_gap=float((rt - rv).mean()), sd_gap=float((rt - rv).std(ddof=1)) if len(reps) > 1 else None,
                                 frac_test_gt_val=float(np.mean(rt > rv)), splits=reps),
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps({k: metrics[k] for k in ("baseline_constant",)}, indent=2))
    print(f"kNN repeated splits: mean val {rv.mean():.4f}, mean test {rt.mean():.4f}, "
          f"test > val in {np.sum(rt > rv)}/{len(reps)}")


if __name__ == "__main__":
    main()
