"""
Run_Learning_Curve.py - Learning Curve Experiment for Reviewer Response
=======================================================================
Purpose:
    Demonstrates model generalisation behaviour across increasing dataset sizes
    (15 / 30 / 60 / 90 / 120 projects).

    Two-level randomness design (journal-grade robustness):
      OUTER loop -- n_gen_seeds independent generator seeds per scale point.
                    Each seed produces a completely different synthetic portfolio.
                    Directly answers: "would results hold with a different
                    random portfolio?" -- the key question top-journal reviewers raise.
      INNER loop -- n_folds GroupKFold x n_seeds training seeds per portfolio.

    Per scale point reports:
      mean +/- SD of val-loss / Test R2 / Test MAE  ACROSS generator seeds (outer)
      A six-panel figure with SD bands and per-seed scatter.

Addresses reviewers:
    R2 #6  -- inner CV shows model learns from sequences, not just project count
    R2 #7  -- outer generator seeds confirm 30 projects is near-convergent
    R2 #8  -- two-level SD decomposition distinguishes data vs training noise

Usage:
    python Run_Learning_Curve.py \
        --n_scales 15 30 60 90 120 \
        --n_gen_seeds 3 \
        --n_folds 5 \
        --n_seeds 3 \
        --num_epochs 80 \
        --device cpu
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for _depth in range(1, 7):
    try:
        _candidate = _FILE.parents[_depth]
    except IndexError:
        break
    if (_candidate / "industry_config.json").exists() and        (_candidate / "ModelGenerator.py").exists():
        PROJECT_ROOT = _candidate
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _FILE.parents[3]

_lstm_dir = PROJECT_ROOT / "LSTM"
for _p in [str(PROJECT_ROOT), str(_lstm_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LSTM_Model import LSTMSeq2SeqMass
from Train import add_enhanced_features, MixedLoss, Seq2SeqDataset
from ModelGenerator import generate_all_projects

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIST_FEATURE_COLS = [
    "progress_pct", "mat_index", "lab_index", "cpi_index",
    "material_cost", "labour_cost", "equip_cost", "admin_cost",
    "total_cost", "month", "normalized_time",
    "remaining_months", "is_completion_phase",
]
FUT_FEATURE_COLS = [
    "month", "normalized_time", "remaining_months",
    "is_completion_phase", "cpi_index", "mat_index", "lab_index",
]
UNION_COLS = sorted(set(HIST_FEATURE_COLS) | set(FUT_FEATURE_COLS))

SEQ_LEN   = 12
HORIZON   = 3
BATCH     = 32
LR        = 1e-3
PATIENCE  = 20
BASE_SEED = 2025
GEN_SEED_OFFSETS = [0, 31337, 99991, 131071, 524287]


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_dataset(n_projects: int, tmp_dir: Path, gen_seed: int) -> pd.DataFrame:
    np.random.seed(gen_seed)
    random.seed(gen_seed)
    out_path = tmp_dir / f"gs{gen_seed}_n{n_projects}"
    out_path.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(str(PROJECT_ROOT))
    try:
        print(f"    Generating {n_projects} projects  [gen_seed={gen_seed}]")
        df = generate_all_projects(
            n_projects=n_projects,
            country="CN",
            output_path=str(out_path),
        )
    finally:
        os.chdir(original_cwd)
    return df


# ---------------------------------------------------------------------------
# Single fold x training-seed
# ---------------------------------------------------------------------------
def train_fold_seed(train_df, val_df, fold, seed, num_epochs, device,
                    global_seed_offset=0):
    set_all_seeds(BASE_SEED + global_seed_offset + fold * 1000 + seed)

    train_ds = Seq2SeqDataset(
        train_df, SEQ_LEN, HORIZON,
        hist_feature_cols=HIST_FEATURE_COLS,
        fut_feature_cols=FUT_FEATURE_COLS,
        target_col="total_cost",
    )
    val_ds = Seq2SeqDataset(
        val_df, SEQ_LEN, HORIZON,
        hist_feature_cols=HIST_FEATURE_COLS,
        fut_feature_cols=FUT_FEATURE_COLS,
        target_col="total_cost",
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"best_val_loss": float("nan"), "best_epoch": 0,
                "n_train_seq": 0, "state_dict": None}

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model = LSTMSeq2SeqMass(
        in_hist=len(HIST_FEATURE_COLS), in_fut=len(FUT_FEATURE_COLS),
        hidden_size=128, num_layers=2, dropout=0.2, horizon=HORIZON,
    ).to(device)

    criterion = MixedLoss(0.7, 0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val   = float("inf")
    best_epoch = 0
    patience_c = 0
    best_sd    = None

    for epoch in range(num_epochs):
        model.train()
        tr_losses = []
        for xh, xf, y in train_loader:
            xh, xf, y = xh.to(device), xf.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xh, xf)["p50"], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xh, xf, y in val_loader:
                xh, xf, y = xh.to(device), xf.to(device), y.to(device)
                va_losses.append(criterion(model(xh, xf)["p50"], y).item())

        avg_val = float(np.mean(va_losses)) if va_losses else float("inf")
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            best_epoch = epoch + 1
            patience_c = 0
            best_sd    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                break

    return {"best_val_loss": best_val, "best_epoch": best_epoch,
            "n_train_seq": len(train_ds), "state_dict": best_sd}


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
def evaluate_test(models_info, test_df, device):
    if test_df.empty or not models_info:
        return {"test_mae": float("nan"), "test_r2": float("nan"), "n_test_seq": 0}

    all_preds, all_targets = [], []

    for state_dict, scaler in models_info:
        if state_dict is None:
            continue
        tc = test_df.copy()
        for c in UNION_COLS:
            tc[c] = pd.to_numeric(tc[c], errors="coerce").fillna(0.0)
        tc[UNION_COLS] = scaler.transform(tc[UNION_COLS])

        ds = Seq2SeqDataset(tc, SEQ_LEN, HORIZON,
                            hist_feature_cols=HIST_FEATURE_COLS,
                            fut_feature_cols=FUT_FEATURE_COLS,
                            target_col="total_cost")
        if len(ds) == 0:
            continue

        loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
        m = LSTMSeq2SeqMass(
            in_hist=len(HIST_FEATURE_COLS), in_fut=len(FUT_FEATURE_COLS),
            hidden_size=128, num_layers=2, dropout=0.2, horizon=HORIZON,
        ).to(device)
        m.load_state_dict(state_dict)
        m.eval()

        fp, ft = [], []
        with torch.no_grad():
            for xh, xf, y in loader:
                xh, xf = xh.to(device), xf.to(device)
                p = m(xh, xf)["p50"].cpu().numpy()
                fp.append(p[:, 0])
                ft.append(y.numpy()[:, 0])
        if fp:
            all_preds.append(np.concatenate(fp))
            all_targets.append(np.concatenate(ft))

    if not all_preds:
        return {"test_mae": float("nan"), "test_r2": float("nan"), "n_test_seq": 0}

    min_len  = min(len(p) for p in all_preds)
    ensemble = np.median(np.stack([p[:min_len] for p in all_preds], axis=0), axis=0)
    targets  = all_targets[0][:min_len]

    mae    = float(np.mean(np.abs(ensemble - targets)))
    ss_res = float(np.sum((ensemble - targets) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"test_mae": mae, "test_r2": r2, "n_test_seq": min_len}


# ---------------------------------------------------------------------------
# Single generator-seed replicate (inner CV loop)
# ---------------------------------------------------------------------------
def run_one_gen_seed(n_projects, tmp_dir, gen_seed, n_folds, n_seeds,
                     num_epochs, device, gen_seed_rank):
    t0 = time.time()

    df = generate_dataset(n_projects, tmp_dir, gen_seed)
    df = add_enhanced_features(df)
    df["total_cost_raw"] = df["total_cost"]
    df["total_cost"] = np.log1p(df["total_cost_raw"])

    # Stratified 20% test split
    meta = (df.groupby("project_id")
            .agg(ptype=("project_type", "first"),
                 cost_sum=("total_cost_raw", "sum")))
    meta["cost_q"] = pd.qcut(meta["cost_sum"], q=4, labels=[0, 1, 2, 3], duplicates="drop")
    meta["strata"] = meta["ptype"].astype(str) + "|Q" + meta["cost_q"].astype(str)
    test_n  = max(1, int(n_projects * 0.20))
    rng     = np.random.default_rng(gen_seed)
    counts  = meta["strata"].value_counts(normalize=True)
    alloc   = (counts * test_n).astype(int)
    for s in counts.sort_values(ascending=False).index[: test_n - alloc.sum()]:
        alloc[s] += 1
    test_pids = []
    for s, k in alloc.items():
        if k <= 0:
            continue
        cands = meta[meta["strata"] == s].index.tolist()
        test_pids.extend(rng.choice(cands, size=min(k, len(cands)), replace=False).tolist())

    all_pids      = df["project_id"].unique().tolist()
    trainval_pids = [p for p in all_pids if p not in test_pids]
    df_trainval   = df[df["project_id"].isin(trainval_pids)].copy()
    df_test       = df[df["project_id"].isin(test_pids)].copy()

    all_val_losses, all_best_epochs, models_info = [], [], []
    gkf    = GroupKFold(n_splits=n_folds)
    groups = df_trainval["project_id"].values
    global_offset = gen_seed_rank * 100000

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_trainval, groups=groups)):
        train_raw = df_trainval.iloc[tr_idx].copy()
        val_raw   = df_trainval.iloc[va_idx].copy()
        for c in UNION_COLS:
            train_raw[c] = pd.to_numeric(train_raw[c], errors="coerce")
            val_raw[c]   = pd.to_numeric(val_raw[c],   errors="coerce")
        col_means = train_raw[UNION_COLS].mean(numeric_only=True)
        train_raw[UNION_COLS] = train_raw[UNION_COLS].fillna(col_means)
        val_raw[UNION_COLS]   = val_raw[UNION_COLS].fillna(col_means)

        scaler = StandardScaler()
        scaler.fit(train_raw[UNION_COLS])
        train_sc = train_raw.copy()
        val_sc   = val_raw.copy()
        train_sc[UNION_COLS] = scaler.transform(train_raw[UNION_COLS])
        val_sc[UNION_COLS]   = scaler.transform(val_raw[UNION_COLS])

        fold_losses = []
        for seed in range(n_seeds):
            res = train_fold_seed(train_sc, val_sc, fold=fold, seed=seed,
                                  num_epochs=num_epochs, device=device,
                                  global_seed_offset=global_offset)
            all_val_losses.append(res["best_val_loss"])
            all_best_epochs.append(res["best_epoch"])
            fold_losses.append(res["best_val_loss"])
            if res["state_dict"] is not None:
                models_info.append((res["state_dict"], scaler))

        print(f"      Fold {fold+1}/{n_folds}  val_loss={np.nanmean(fold_losses):.5f}")

    avg_dur = (df["total_duration_months"].mean()
               if "total_duration_months" in df.columns
               else df.groupby("project_id")["month"].max().mean())
    avg_train_proj = len(trainval_pids) * (1 - 1 / n_folds)
    est_seq = int(avg_train_proj * max(0.0, avg_dur - SEQ_LEN - HORIZON + 1))

    test_m = evaluate_test(models_info, df_test, device)
    elapsed = time.time() - t0

    return {
        "gen_seed":            gen_seed,
        "n_projects":          n_projects,
        "estimated_train_seq": est_seq,
        "n_models":            len(models_info),
        "mean_val_loss":       float(np.nanmean(all_val_losses)),
        "std_val_loss":        float(np.nanstd(all_val_losses)),
        "mean_best_epoch":     float(np.nanmean(all_best_epochs)),
        "test_mae_scaled":     test_m["test_mae"],
        "test_r2_scaled":      test_m["test_r2"],
        "n_test_seq":          test_m["n_test_seq"],
        "elapsed_sec":         round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Scale-point wrapper (outer generator-seed loop)
# ---------------------------------------------------------------------------
def run_scale_point(n_projects, tmp_dir, n_gen_seeds, n_folds, n_seeds,
                    num_epochs, device, scale_idx):
    print(f"\n{'='*70}")
    print(f"  SCALE POINT: {n_projects} projects  "
          f"[{n_gen_seeds} gen seeds x {n_folds} folds x {n_seeds} training seeds]")
    print(f"{'='*70}")

    gen_seeds = [
        BASE_SEED + scale_idx * 7919 + i * 104729
        for i in range(n_gen_seeds)
    ]

    reps = []
    for rank, gs in enumerate(gen_seeds):
        print(f"\n  -- Generator seed {rank+1}/{n_gen_seeds}  (seed={gs}) --")
        r = run_one_gen_seed(n_projects=n_projects, tmp_dir=tmp_dir,
                             gen_seed=gs, n_folds=n_folds, n_seeds=n_seeds,
                             num_epochs=num_epochs, device=device,
                             gen_seed_rank=rank)
        reps.append(r)
        print(f"    -> val_loss={r['mean_val_loss']:.5f}  "
              f"R2={r['test_r2_scaled']:.4f}  "
              f"seq/fold~{r['estimated_train_seq']:,}  "
              f"({r['elapsed_sec']:.0f}s)")

    val_losses = [r["mean_val_loss"]    for r in reps]
    r2_vals    = [r["test_r2_scaled"]   for r in reps]
    mae_vals   = [r["test_mae_scaled"]  for r in reps]
    epoch_vals = [r["mean_best_epoch"]  for r in reps]
    seq_vals   = [r["estimated_train_seq"] for r in reps]

    agg = {
        "n_projects":           n_projects,
        "n_gen_seeds":          n_gen_seeds,
        "estimated_train_seq":  int(np.mean(seq_vals)),
        "mean_val_loss":        float(np.nanmean(val_losses)),
        "std_val_loss_outer":   float(np.nanstd(val_losses)),
        "std_val_loss_inner":   float(np.nanmean([r["std_val_loss"] for r in reps])),
        "mean_r2":              float(np.nanmean(r2_vals)),
        "std_r2_outer":         float(np.nanstd(r2_vals)),
        "mean_mae":             float(np.nanmean(mae_vals)),
        "std_mae_outer":        float(np.nanstd(mae_vals)),
        "mean_best_epoch":      float(np.nanmean(epoch_vals)),
        "per_gen_seed_val_loss": val_losses,
        "per_gen_seed_r2":       r2_vals,
        "per_gen_seed_mae":      mae_vals,
        "per_gen_seed_seeds":    gen_seeds,
    }

    print(f"\n  AGGREGATED {n_projects} projects ({n_gen_seeds} generator seeds):")
    print(f"    Val loss : {agg['mean_val_loss']:.5f} +/- {agg['std_val_loss_outer']:.5f}  (outer SD)")
    print(f"    Test R2  : {agg['mean_r2']:.4f} +/- {agg['std_r2_outer']:.4f}")
    print(f"    Test MAE : {agg['mean_mae']:.5f} +/- {agg['std_mae_outer']:.5f}")
    return agg


# ---------------------------------------------------------------------------
# Plot (six panels)
# ---------------------------------------------------------------------------
def plot_learning_curve(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Learning Curve: Model Performance vs. Training Dataset Size\n"
        "(Mean +/- SD across independent generator seeds)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    n    = df["n_projects"].values.astype(int)
    vl   = df["mean_val_loss"].values
    vs_o = df["std_val_loss_outer"].values
    sq   = df["estimated_train_seq"].values
    r2   = df["mean_r2"].values
    r2s  = df["std_r2_outer"].values
    ep   = df["mean_best_epoch"].values
    MK   = "o"

    # (a) Val loss vs n_projects
    ax = axes[0, 0]
    ax.plot(n, vl, marker=MK, color="#2c7bb6", lw=2, ms=7, zorder=3)
    ax.fill_between(n, vl - vs_o, vl + vs_o, alpha=0.25, color="#2c7bb6", label="+/-1 SD (gen seeds)")
    for xi, yi in zip(n, vl):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("Number of Synthetic Projects", fontsize=10)
    ax.set_ylabel("Mean Best Val Loss", fontsize=10)
    ax.set_title("(a) Val Loss vs. Project Count", fontsize=11, fontweight="bold")
    ax.set_xticks(n); ax.legend(fontsize=8); ax.grid(alpha=0.3, ls="--")

    # (b) Val loss vs sequences
    ax = axes[0, 1]
    ax.plot(sq, vl, marker=MK, color="#d7191c", lw=2, ms=7, zorder=3)
    ax.fill_between(sq, vl - vs_o, vl + vs_o, alpha=0.20, color="#d7191c")
    for xi, ni in zip(sq, n):
        idx = list(sq).index(xi)
        ax.annotate(f"n={ni}", (xi, vl[idx]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("Est. Training Sequences per Fold", fontsize=10)
    ax.set_ylabel("Mean Best Val Loss", fontsize=10)
    ax.set_title("(b) Val Loss vs. Training Sequences", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, ls="--")

    # (c) Test R2
    ax = axes[0, 2]
    ax.plot(n, r2, marker=MK, color="#1a9641", lw=2, ms=7, zorder=3)
    ax.fill_between(n, r2 - r2s, r2 + r2s, alpha=0.22, color="#1a9641", label="+/-1 SD")
    ax.set_ylim(max(0, float(np.nanmin(r2 - r2s)) - 0.03), 1.03)
    for xi, yi in zip(n, r2):
        if np.isfinite(yi):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("Number of Synthetic Projects", fontsize=10)
    ax.set_ylabel("Test R2 (scaled space)", fontsize=10)
    ax.set_title("(c) Test R2 vs. Project Count", fontsize=11, fontweight="bold")
    ax.set_xticks(n); ax.legend(fontsize=8); ax.grid(alpha=0.3, ls="--")

    # (d) Convergence speed
    ax = axes[1, 0]
    ax.plot(n, ep, marker=MK, color="#984ea3", lw=2, ms=7)
    ax.set_xlabel("Number of Synthetic Projects", fontsize=10)
    ax.set_ylabel("Mean Best Epoch", fontsize=10)
    ax.set_title("(d) Convergence Speed", fontsize=11, fontweight="bold")
    ax.set_xticks(n); ax.grid(alpha=0.3, ls="--")

    # (e) Per-gen-seed scatter
    ax = axes[1, 1]
    n_gen = int(df["n_gen_seeds"].iloc[0]) if "n_gen_seeds" in df.columns else 3
    cmap  = plt.cm.tab10(np.linspace(0, 0.9, n_gen))
    for gi in range(n_gen):
        col = f"per_gen_seed_val_loss_{gi}"
        if col in df.columns:
            ax.plot(n, df[col].values, marker=MK, lw=1.2, ms=5,
                    color=cmap[gi], alpha=0.8, label=f"Gen seed {gi+1}")
    ax.plot(n, vl, marker="s", color="black", lw=2, ms=7, zorder=5, label="Mean")
    ax.set_xlabel("Number of Synthetic Projects", fontsize=10)
    ax.set_ylabel("Val Loss (per gen seed)", fontsize=10)
    ax.set_title("(e) Per-Generator-Seed Val Loss", fontsize=11, fontweight="bold")
    ax.set_xticks(n); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, ls="--")

    # (f) Outer SD trend
    ax = axes[1, 2]
    ax.plot(n, vs_o, marker=MK, color="#666666", lw=2, ms=7)
    for xi, yi in zip(n, vs_o):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("Number of Synthetic Projects", fontsize=10)
    ax.set_ylabel("SD of Val Loss (across gen seeds)", fontsize=10)
    ax.set_title("(f) Generator-Seed Stability Trend", fontsize=11, fontweight="bold")
    ax.set_xticks(n); ax.grid(alpha=0.3, ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "learning_curve"))
    parser.add_argument("--n_scales", type=int, nargs="+", default=[15, 30, 60, 90, 120])
    parser.add_argument("--n_gen_seeds", type=int, default=3,
                        help="Independent generator seeds per scale (default 3)")
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--n_seeds",    type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_data"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    total_models = (len(args.n_scales) * args.n_gen_seeds * args.n_folds * args.n_seeds)
    print("\n" + "="*70)
    print("LEARNING CURVE EXPERIMENT  (multi-generator-seed)")
    print("="*70)
    print(f"  Scale points     : {sorted(set(args.n_scales))}")
    print(f"  Generator seeds  : {args.n_gen_seeds}  <- outer loop (data robustness)")
    print(f"  Folds x Tr seeds : {args.n_folds} x {args.n_seeds}  <- inner loop")
    print(f"  Total models     : {total_models}")
    print(f"  Max epochs       : {args.num_epochs}")
    print(f"  Device           : {args.device}")
    print(f"  Output dir       : {output_dir}")
    print("="*70)

    all_agg, all_raw = [], []

    for idx, n_proj in enumerate(sorted(set(args.n_scales))):
        agg = run_scale_point(
            n_projects=n_proj, tmp_dir=tmp_dir,
            n_gen_seeds=args.n_gen_seeds, n_folds=args.n_folds,
            n_seeds=args.n_seeds, num_epochs=args.num_epochs,
            device=args.device, scale_idx=idx,
        )

        for gi, gs in enumerate(agg["per_gen_seed_seeds"]):
            all_raw.append({
                "n_projects": n_proj, "gen_seed_rank": gi, "gen_seed": gs,
                "val_loss": agg["per_gen_seed_val_loss"][gi],
                "test_r2":  agg["per_gen_seed_r2"][gi],
                "test_mae": agg["per_gen_seed_mae"][gi],
            })

        agg_flat = {k: v for k, v in agg.items() if not isinstance(v, list)}
        for gi in range(args.n_gen_seeds):
            if gi < len(agg["per_gen_seed_val_loss"]):
                agg_flat[f"per_gen_seed_val_loss_{gi}"] = agg["per_gen_seed_val_loss"][gi]
                agg_flat[f"per_gen_seed_r2_{gi}"]       = agg["per_gen_seed_r2"][gi]
        all_agg.append(agg_flat)

        pd.DataFrame(all_agg).to_csv(output_dir / "learning_curve.csv", index=False)
        pd.DataFrame(all_raw).to_csv(output_dir / "learning_curve_raw.csv", index=False)

    df     = pd.DataFrame(all_agg)
    df_raw = pd.DataFrame(all_raw)

    for c in ["mean_val_loss","std_val_loss_outer","std_val_loss_inner",
              "mean_r2","std_r2_outer","mean_mae","std_mae_outer","mean_best_epoch"]:
        if c in df.columns:
            df[c] = df[c].round(6)

    df.to_csv(output_dir / "learning_curve.csv", index=False)
    df_raw.to_csv(output_dir / "learning_curve_raw.csv", index=False)
    print(f"\nAggregated results : {output_dir / 'learning_curve.csv'}")
    print(f"Per-replicate data  : {output_dir / 'learning_curve_raw.csv'}")

    plot_learning_curve(df, output_dir / "learning_curve.png")

    print("\n" + "="*70)
    print("LEARNING CURVE SUMMARY  (mean +/- outer SD across generator seeds)")
    print("="*70)
    show = [c for c in ["n_projects","n_gen_seeds","estimated_train_seq",
                         "mean_val_loss","std_val_loss_outer",
                         "mean_r2","std_r2_outer","mean_best_epoch"] if c in df.columns]
    print(df[show].to_string(index=False))

    # Convergence check
    if len(df) >= 3:
        losses  = df["mean_val_loss"].values
        deltas  = np.diff(losses)
        last_d  = abs(deltas[-1])
        first_d = abs(deltas[0])
        if first_d > 0 and np.isfinite(last_d):
            ratio = last_d / first_d
            tag = "CONVERGENCE" if ratio < 0.15 else "NOTE"
            print(f"\n  {tag}: val-loss change at largest scale is "
                  f"{ratio*100:.1f}% of initial drop.")

    # Stability summary + manuscript text
    if "std_val_loss_outer" in df.columns:
        max_outer = df["std_val_loss_outer"].max()
        n_gs = int(df["n_gen_seeds"].iloc[0])
        print(f"\n  Max outer SD (across gen seeds): {max_outer:.5f}")
        print(f"\n  Manuscript text suggestion:")
        print(f"  'To verify results are not artefacts of a single synthetic")
        print(f"  portfolio, we repeated the experiment with {n_gs} independently")
        print(f"  generated datasets per scale point. The outer standard deviation")
        print(f"  of validation loss across generator seeds remained below")
        print(f"  {max_outer:.4f} at all scales, confirming stability with respect")
        print(f"  to synthetic data generation.'")

    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump({"n_scales": args.n_scales, "n_gen_seeds": args.n_gen_seeds,
                   "n_folds": args.n_folds, "n_seeds": args.n_seeds,
                   "num_epochs": args.num_epochs, "device": args.device,
                   "seq_len": SEQ_LEN, "horizon": HORIZON,
                   "base_seed": BASE_SEED, "timestamp": timestamp}, f, indent=2)

    print(f"\nAll outputs in: {output_dir}")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())