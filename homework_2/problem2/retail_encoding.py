"""
Homework 2, Problem 2: encoding high-cardinality categoricals (StockCode, CustomerID) to predict Quantity.

Quantity is very heavy-tailed, so every model is trained with squared error on y = log1p(Quantity)
(equivalently, a log-normal model for Quantity). Models compared on a seeded 70/15/15 split:
    - global mean (baseline)
    - one-hot ridge regression (additive per-category effects; shows one-hot is feasible only as a sparse linear model)
    - (b) target encoding (category_encoders.TargetEncoder) + gradient-boosted trees
    - (c) MLP with an embedding layer per feature
    - (d) MLP with embeddings + target-encoded features
    - (e) error breakdown by how often a test row's customer / product was seen in training

Usage (from homework_2/):
    python problem2/retail_encoding.py
    python problem2/retail_encoding.py --quick

Outputs go to problem2/results/.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from category_encoders import TargetEncoder
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "retail.csv")
OUT_DIR = os.path.join(HERE, "results")
CATS = ["StockCode", "CustomerID"]
SEED = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits(quick: bool = False):
    df = pd.read_csv(DATA_PATH, dtype={"StockCode": str})
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    for c in CATS:
        df[c] = df[c].astype("category")
    if quick:
        df = df.sample(n=40_000, random_state=SEED)
    perm = np.random.default_rng(SEED).permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    parts = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return df, {k: df.iloc[v].reset_index(drop=True) for k, v in parts.items()}


def to_log(q):
    return np.log1p(q.to_numpy(np.float64))


def metrics(pred_log, y_log):
    """RMSE / MAE on the log1p scale (the training objective) plus MAE in raw units, using expm1(pred),
    which is a median-type prediction of Quantity under a log-normal model."""
    err = pred_log - y_log
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    return {
        "rmse_log": float(np.sqrt(np.mean(err ** 2))),
        "mae_log": float(np.mean(np.abs(err))),
        "r2_log": float(1 - np.sum(err ** 2) / ss_tot),
        "mae_raw": float(np.mean(np.abs(np.expm1(pred_log) - np.expm1(y_log)))),
    }


def target_encode(train, others, y_train_log, n_folds=5):
    """Target encoding with category_encoders.TargetEncoder (smoothed mean of log1p(Quantity) per category).

    Training rows are encoded out-of-fold (K-fold) so a row's encoding never uses its own target. Val/test rows
    are encoded with an encoder fit on the full training set; unseen categories fall back to the global prior.
    """
    def make():
        return TargetEncoder(cols=CATS, min_samples_leaf=20, smoothing=10.0,
                             handle_unknown="value", handle_missing="value")

    oof = np.zeros((len(train), len(CATS)))
    for tr_idx, ho_idx in KFold(n_folds, shuffle=True, random_state=SEED).split(train):
        enc = make().fit(train.iloc[tr_idx][CATS], y_train_log[tr_idx])
        oof[ho_idx] = enc.transform(train.iloc[ho_idx][CATS]).to_numpy()
    full = make().fit(train[CATS], y_train_log)
    return oof, [full.transform(o[CATS]).to_numpy() for o in others]


def build_index(train):
    """Category -> integer id; id 0 is reserved for categories unseen in training."""
    return {c: {v: i + 1 for i, v in enumerate(train[c].astype(str).unique())} for c in CATS}


def encode_ids(df, index):
    return np.stack([df[c].astype(str).map(index[c]).fillna(0).astype(np.int64).to_numpy() for c in CATS], axis=1)


class EmbeddingNet(nn.Module):
    """One embedding table per categorical feature (+ optional dense features), concatenated into an MLP."""

    def __init__(self, cardinalities, emb_dim=16, n_dense=0, hidden=(128, 64), dropout=0.1, unk_rate=0.01):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim) for n in cardinalities])
        self.unk_rate = unk_rate
        layers, d = [], emb_dim * len(cardinalities) + n_dense
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids, dense=None):
        if self.training and self.unk_rate > 0:
            ids = ids.masked_fill(torch.rand(ids.shape) < self.unk_rate, 0)
        parts = [emb(ids[:, j]) for j, emb in enumerate(self.embs)]
        if dense is not None:
            parts.append(dense)
        return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)


def train_net(model, train, val, *, epochs, lr=2e-3, weight_decay=1e-5, batch_size=2048, patience=4, name=""):
    """Adam on MSE of log1p(Quantity), early stopping on validation MSE; returns history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    hist = {"train_mse": [], "val_mse": []}
    n = len(train[-1])
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            pred = model(*[t[b] for t in train[:-1]])
            loss = ((pred - train[-1][b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val_mse = ((model(*val[:-1]) - val[-1]) ** 2).mean().item()
        hist["train_mse"].append(tot / n)
        hist["val_mse"].append(val_mse)
        print(f"[{name}] epoch {epoch + 1:3d}  train MSE {tot / n:.4f}  val MSE {val_mse:.4f}")
        if val_mse < best - 1e-4:
            best, bad = val_mse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return hist


def net_predict(model, *inputs):
    with torch.no_grad():
        return model(*inputs).numpy().astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--emb-dim", type=int, default=16)
    args = parser.parse_args()
    epochs = 3 if args.quick else args.epochs

    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    full, s = load_splits(args.quick)
    train, val, test = s["train"], s["val"], s["test"]
    ytr, yva, yte = to_log(train.Quantity), to_log(val.Quantity), to_log(test.Quantity)

    card = {c: int(full[c].nunique()) for c in CATS}
    train_counts = {c: train[c].astype(str).value_counts() for c in CATS}
    explore = {
        "n_rows": int(len(full)),
        "cardinality": card,
        "one_hot_columns": int(sum(card.values())),
        "dense_one_hot_gb_float32": float(len(full) * sum(card.values()) * 4 / 1e9),
        "pair_cardinality_observed": int(full.groupby(CATS, observed=True).ngroups),
        "pair_cardinality_possible": int(np.prod(list(card.values()))),
        "rows_per_category_median": {c: float(full[c].value_counts().median()) for c in CATS},
        "categories_with_lt5_rows": {c: int((full[c].value_counts() < 5).sum()) for c in CATS},
        "quantity_quantiles": {str(q): float(v) for q, v in full.Quantity.quantile([.5, .9, .99, .999, 1.0]).items()},
        "split_sizes": {k: int(len(v)) for k, v in s.items()},
        "test_rows_with_unseen": {c: float((~test[c].astype(str).isin(train_counts[c].index)).mean()) for c in CATS},
    }
    print(json.dumps(explore, indent=2))

    results, preds, hists = {}, {}, {}

    preds["global_mean"] = np.full_like(yte, ytr.mean())

    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=1)
    Otr = ohe.fit_transform(train[CATS].astype(str))
    best = None
    for alpha in [0.3, 1, 3, 10, 30]:
        r = Ridge(alpha=alpha).fit(Otr, ytr)
        v = np.mean((r.predict(ohe.transform(val[CATS].astype(str))) - yva) ** 2)
        if best is None or v < best[0]:
            best = (v, alpha, r)
    explore["one_hot_ridge_alpha"] = best[1]
    explore["one_hot_matrix_nnz_fraction"] = float(Otr.nnz / (Otr.shape[0] * Otr.shape[1]))
    preds["one_hot_ridge"] = best[2].predict(ohe.transform(test[CATS].astype(str)))

    te_tr, (te_va, te_te) = target_encode(train, [val, test], ytr)
    gbm_kw = dict(loss="squared_error", max_iter=1000 if not args.quick else 50, learning_rate=0.05,
                  max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0, random_state=SEED)
    gbm = HistGradientBoostingRegressor(**gbm_kw, early_stopping=False, warm_start=True)
    best_v, best_it, stall = np.inf, 0, 0
    for it in range(50, gbm_kw["max_iter"] + 1, 50):
        gbm.set_params(max_iter=it).fit(te_tr, ytr)
        v = np.mean((gbm.predict(te_va) - yva) ** 2)
        if v < best_v - 1e-5:
            best_v, best_it, stall = v, it, 0
        else:
            stall += 1
            if stall >= 3:
                break
    gbm = HistGradientBoostingRegressor(**{**gbm_kw, "max_iter": best_it}).fit(te_tr, ytr)
    explore["te_gbm_iters"] = best_it
    preds["te_gbm"] = gbm.predict(te_te)

    index = build_index(train)
    ids_tr, ids_va, ids_te = (torch.from_numpy(encode_ids(d, index)) for d in (train, val, test))
    card_tr = [len(index[c]) for c in CATS]
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    y_t = (t(ytr), t(yva))

    set_seed(SEED)
    emb_net = EmbeddingNet(card_tr, emb_dim=args.emb_dim)
    start = time.time()
    hists["embedding_nn"] = train_net(emb_net, (ids_tr, y_t[0]), (ids_va, y_t[1]), epochs=epochs, name="emb")
    explore["embedding_nn_seconds"] = time.time() - start
    preds["embedding_nn"] = net_predict(emb_net, ids_te)

    mu, sd = te_tr.mean(0), te_tr.std(0)
    z = lambda a: t((a - mu) / sd)
    set_seed(SEED)
    combo = EmbeddingNet(card_tr, emb_dim=args.emb_dim, n_dense=len(CATS))
    hists["embedding_plus_te_nn"] = train_net(combo, (ids_tr, z(te_tr), y_t[0]), (ids_va, z(te_va), y_t[1]),
                                              epochs=epochs, name="emb+te")
    preds["embedding_plus_te_nn"] = net_predict(combo, ids_te, z(te_te))

    for k, p in preds.items():
        results[k] = metrics(p, yte)

    breakdown = frequency_breakdown(preds, test, yte, train_counts)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"explore": explore, "test_metrics": results, "breakdown_rmse_log": breakdown,
                   "epochs_run": {k: len(h["val_mse"]) for k, h in hists.items()}, "quick": args.quick}, f, indent=2)
    write_latex_tables(results, breakdown, os.path.join(OUT_DIR))
    make_figures(hists, breakdown, OUT_DIR)

    print("\nTest metrics:")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nTest RMSE (log scale) by training frequency:")
    print(pd.DataFrame(breakdown).round(4).to_string())


def frequency_breakdown(preds, test, yte, train_counts):
    """RMSE on log scale, grouped by the smaller of the customer and product training counts."""
    n_c = test["CustomerID"].astype(str).map(train_counts["CustomerID"]).fillna(0).to_numpy()
    n_s = test["StockCode"].astype(str).map(train_counts["StockCode"]).fillna(0).to_numpy()
    m = np.minimum(n_c, n_s)
    groups = {"unseen (0)": m == 0, "1-9": (m >= 1) & (m < 10), "10-49": (m >= 10) & (m < 50),
              "50-199": (m >= 50) & (m < 200), "200+": m >= 200}
    out = {}
    for g, mask in groups.items():
        out[g] = {k: float(np.sqrt(np.mean((p[mask] - yte[mask]) ** 2))) if mask.any() else float("nan")
                  for k, p in preds.items()}
        out[g]["share_of_test"] = float(mask.mean())
    return out


NAMES = {
    "global_mean": "Global mean",
    "one_hot_ridge": "One-hot + ridge",
    "te_gbm": "(b) Target encoding + GBM",
    "embedding_nn": "(c) Embedding NN",
    "embedding_plus_te_nn": "(d) Embedding + TE NN",
}


def write_latex_tables(results, breakdown, out_dir):
    rows = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"Model & RMSE (log) & MAE (log) & $R^2$ (log) & MAE (units) \\", r"\midrule"]
    for k, label in NAMES.items():
        r = results[k]
        rows.append(f"{label} & {r['rmse_log']:.4f} & {r['mae_log']:.4f} & {round(r['r2_log'], 3) + 0.0:.3f} & "
                    f"{r['mae_raw']:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "metrics_table.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    groups = list(breakdown.keys())
    rows = [r"\begin{tabular}{l" + "c" * len(groups) + "}", r"\toprule",
            r"Training rows of rarer category & " + " & ".join(groups) + r" \\",
            r"Share of test rows & " + " & ".join(f"{breakdown[g]['share_of_test']:.3f}" for g in groups) + r" \\",
            r"\midrule"]
    for k, label in NAMES.items():
        rows.append(label + " & " + " & ".join(f"{breakdown[g][k]:.3f}" for g in groups) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "breakdown_table.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")


def make_figures(hists, breakdown, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, h in hists.items():
        ax.plot(h["val_mse"], label=f"{k} (val)")
        ax.plot(h["train_mse"], "--", alpha=0.6, label=f"{k} (train)")
    ax.set(xlabel="epoch", ylabel="MSE of log1p(Quantity)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=200)
    plt.close(fig)

    groups = list(breakdown.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in ["one_hot_ridge", "te_gbm", "embedding_nn", "embedding_plus_te_nn"]:
        ax.plot(groups, [breakdown[g][k] for g in groups], "o-", label=NAMES[k])
    ax.set(xlabel="training rows of the rarer category (customer or product)", ylabel="test RMSE, log1p(Quantity)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rmse_by_frequency.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
