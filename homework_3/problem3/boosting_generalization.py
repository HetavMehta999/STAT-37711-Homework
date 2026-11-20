"""
Homework 3, Problem 3: generalization of gradient boosting as trees are added.

For dset0 and dset1 (label = first column, features = the other five) we fit sklearn's
GradientBoostingClassifier with 1, 10, 100, 300 and 1000 trees (defaults otherwise: learning rate 0.1,
depth-3 trees) and compare train and test error.

500 points is small, so a single split is noisy. We therefore repeat over N_SPLITS random stratified
70/30 splits and report mean +/- standard error. Boosting is sequential and deterministic here
(subsample=1), so the model with m trees equals the first m stages of the 1000-tree model; we read all
m from staged_predict_proba. A check that this matches separate fits is in `check_staged_equivalence`.

For part (c) we also compute learning curves (CV error vs. training-set size) for boosting and for
logistic regression, to tell label noise (error flat in n) from unlearned structure (error falling in n).

Usage (from homework_3/):
    python problem3/boosting_generalization.py
    python problem3/boosting_generalization.py --quick

Outputs go to problem3/results/.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import plot_style as ps  # noqa: E402

ps.apply()

DATA = {name: os.path.join(HERE, "..", "data", f"{name}.csv") for name in ("dset0", "dset1")}
OUT_DIR = os.path.join(HERE, "results")
TREES = [1, 10, 100, 300, 1000]
MAX_TREES = max(TREES)
N_SPLITS = 20
TEST_FRAC = 0.3


def load(name):
    df = pd.read_csv(DATA[name], header=None)
    return df.iloc[:, 1:].to_numpy(float), df.iloc[:, 0].to_numpy(int)


def log_loss_rows(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def staged_curves(X, y, seed):
    """Train/test 0-1 error and log-loss for every number of trees 1..MAX_TREES on one split."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=MAX_TREES, random_state=seed).fit(Xtr, ytr)
    out = {}
    for split, Xs, ys in (("train", Xtr, ytr), ("test", Xte, yte)):
        P = np.array([p[:, 1] for p in gb.staged_predict_proba(Xs)])  # (MAX_TREES, n)
        out[f"{split}_err"] = ((P > 0.5).astype(int) != ys).mean(axis=1)
        out[f"{split}_logloss"] = log_loss_rows(ys, P).mean(axis=1)
    return out


def check_staged_equivalence(X, y, seed=0, m=10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)
    big = GradientBoostingClassifier(n_estimators=MAX_TREES, random_state=seed).fit(Xtr, ytr)
    small = GradientBoostingClassifier(n_estimators=m, random_state=seed).fit(Xtr, ytr)
    staged = list(big.staged_predict_proba(Xte))[m - 1]
    return float(np.abs(staged - small.predict_proba(Xte)).max())


def paired_diff(curves, a, b, key="test_err"):
    """Per-split difference key(b trees) - key(a trees): mean, s.e., #splits where it is > 0."""
    d = np.array([c[key][b - 1] - c[key][a - 1] for c in curves])
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "n_pos": int((d > 0).sum()),
            "n_neg": int((d < 0).sum()), "n": len(d)}


LC_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]


def learning_curves(X, y, n_repeats):
    """(c) diagnostic: CV test error vs. training-set size for boosting (100 trees) and a linear model.
    A noise floor shows up as a curve that is flat in n; unlearned structure as one still falling."""
    models = {"boosting (100 trees)": lambda: GradientBoostingClassifier(n_estimators=100, random_state=0),
              "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    out = {}
    for name, make in models.items():
        errs = []
        for r in range(n_repeats):
            cv = StratifiedKFold(5, shuffle=True, random_state=r)
            sizes, _, te = learning_curve(make(), X, y, cv=cv, train_sizes=LC_SIZES, shuffle=True, random_state=r)
            errs.append(1 - te.mean(axis=1))
        E = np.array(errs)
        out[name] = {"n": sizes.tolist(), "err": E.mean(0).tolist(),
                     "se": (E.std(0, ddof=1) / np.sqrt(len(E))).tolist() if len(E) > 1 else [0] * len(sizes)}
    return out


def plot_learning_curves(lcs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, lc) in zip(axes, lcs.items()):
        for (model, r), color in zip(lc.items(), (ps.ORANGE, ps.AQUA)):
            n, e, se = np.array(r["n"]), np.array(r["err"]), np.array(r["se"])
            ax.plot(n, e, color=color, lw=2, marker="o", ms=5, mec="white", mew=1, label=model)
            ax.fill_between(n, e - 2 * se, e + 2 * se, color=color, alpha=0.18, linewidth=0)
        ax.set_title(name)
        ax.set_xlabel("Training-set size")
        ax.set_xticks(lc[next(iter(lc))]["n"])
    axes[0].set_ylabel("Cross-validated test error")
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=2)
    fig.savefig(path)
    plt.close(fig)


def summarize(curves):
    """curves: list of dicts of arrays -> mean and s.e. per metric, shape (MAX_TREES,)."""
    res = {}
    for key in curves[0]:
        A = np.stack([c[key] for c in curves])
        res[key] = (A.mean(0), A.std(0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros(A.shape[1]))
    return res


def write_table(summ, path):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Data & Metric & " + " & ".join(f"{m} tree{'s' if m > 1 else ''}" for m in TREES) + r" \\",
             r"\midrule"]
    labels = [("train_err", "Train error"), ("test_err", "Test error"), ("test_logloss", "Test log-loss")]
    for name, s in summ.items():
        for i, (key, lab) in enumerate(labels):
            mean, se = s[key]
            cells = [f"{mean[m - 1]:.3f} ({se[m - 1]:.3f})" if key != "train_err" else f"{mean[m - 1]:.3f}"
                     for m in TREES]
            lines.append(f"{name if i == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if name != list(summ)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_curves(summ, path):
    m = np.arange(1, MAX_TREES + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    rows = [("err", "0-1 error"), ("logloss", "Log-loss")]
    for col, (name, s) in enumerate(summ.items()):
        for row, (metric, ylabel) in enumerate(rows):
            ax = axes[row, col]
            for split, color in (("train", ps.BLUE), ("test", ps.ORANGE)):
                mean, se = s[f"{split}_{metric}"]
                ax.plot(m, mean, color=color, lw=2, label=split)
                ax.fill_between(m, mean - 2 * se, mean + 2 * se, color=color, alpha=0.18, linewidth=0)
                ax.scatter(TREES, mean[np.array(TREES) - 1], s=36, color=color, edgecolors="white",
                           linewidths=1.2, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(TREES)
            ax.set_xticklabels([str(t) for t in TREES])
            ax.set_ylabel(ylabel)
            if metric == "logloss":
                ax.set_ylim(bottom=0)
            if row == 0:
                best = int(np.argmin(s["test_err"][0])) + 1
                ax.set_title(f"{name}  (test error lowest at {best} trees)")
                ax.legend(loc="lower left")
            else:
                ax.set_xlabel("Number of trees (log scale)")
    fig.tight_layout(h_pad=1.2, w_pad=2.5)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 splits instead of 20")
    args = ap.parse_args()
    n_splits = 3 if args.quick else N_SPLITS
    os.makedirs(OUT_DIR, exist_ok=True)

    summ, metrics = {}, {"n_splits": n_splits, "test_frac": TEST_FRAC, "trees": TREES}
    for name in DATA:
        X, y = load(name)
        print(f"{name}: n={len(y)}, p={X.shape[1]}, class balance {y.mean():.3f}")
        curves = [staged_curves(X, y, seed) for seed in range(n_splits)]
        summ[name] = summarize(curves)
        s = summ[name]
        diffs = {"1000_minus_100": paired_diff(curves, 100, 1000), "1000_minus_300": paired_diff(curves, 300, 1000)}
        best = int(np.argmin(s["test_err"][0])) + 1
        metrics[name] = {
            "at_trees": {str(m): {k: [float(s[k][0][m - 1]), float(s[k][1][m - 1])] for k in s} for m in TREES},
            "best_test_err_trees": best,
            "best_test_err": float(s["test_err"][0][best - 1]),
            "best_test_logloss_trees": int(np.argmin(s["test_logloss"][0])) + 1,
            "paired_test_err_diff": diffs,
        }
        for m in TREES:
            print(f"  {m:5d} trees: train err {s['train_err'][0][m - 1]:.3f}  test err "
                  f"{s['test_err'][0][m - 1]:.3f} +/- {s['test_err'][1][m - 1]:.3f}  test logloss "
                  f"{s['test_logloss'][0][m - 1]:.3f}")
        print(f"  lowest mean test error at {best} trees")
        for k, v in diffs.items():
            print(f"  paired test-error diff {k}: {v['mean']:+.4f} +/- {v['se']:.4f}  (+ in {v['n_pos']}, - in {v['n_neg']} of {v['n']})")

    print("learning curves for (c) ...")
    lcs = {name: learning_curves(*load(name), n_repeats=2 if args.quick else 5) for name in DATA}
    metrics["learning_curves"] = lcs
    for name, lc in lcs.items():
        for model, r in lc.items():
            print(f"  {name} {model}: " + ", ".join(f"n={n}: {e:.3f}" for n, e in zip(r["n"], r["err"])))
    plot_learning_curves(lcs, os.path.join(OUT_DIR, "learning_curves.png"))

    X, y = load("dset0")
    metrics["staged_vs_separate_fit_max_abs_diff"] = check_staged_equivalence(X, y)
    print("staged vs. separate fit, max |diff| in predicted prob:", metrics["staged_vs_separate_fit_max_abs_diff"])

    write_table(summ, os.path.join(OUT_DIR, "error_table.tex"))
    plot_curves(summ, os.path.join(OUT_DIR, "error_curves.png"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
