import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from pathlib import Path
import joblib
import random


from LSTM_Model import LSTMSeq2SeqMass, get_model_summary
from Train import add_enhanced_features, Seq2SeqDataset, MixedLoss


# Path Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "generated" / "synthetic_CN_projects.csv"


# ============================================================
# NEW: Global random seed setting function
# ============================================================
def set_all_seeds(seed: int):
    """
    Set all random seeds to ensure reproducibility.

    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure the determinism of CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_single_fold(train_df, val_df,
                      hist_feature_cols, fut_feature_cols,
                      fold=0, seed=0,  # NEW: 添加seed参数
                      sequence_length=12, prediction_horizon=3,
                      batch_size=32, num_epochs=100,
                      learning_rate=0.001, device='cpu', use_mixed_loss=True):
    """
    Train single fold single seed (Seq2Seq)
    NEW: add seed parameter for identification and saving
    """
    print(f"\n{'='*70}")
    print(f"TRAINING FOLD {fold+1} | SEED {seed}")
    print(f"{'='*70}")

    # Datasets
    train_dataset = Seq2SeqDataset(
        train_df, sequence_length, prediction_horizon,
        hist_feature_cols=hist_feature_cols, fut_feature_cols=fut_feature_cols,
        target_col='total_cost'
    )
    val_dataset = Seq2SeqDataset(
        val_df, sequence_length, prediction_horizon,
        hist_feature_cols=hist_feature_cols, fut_feature_cols=fut_feature_cols,
        target_col='total_cost'
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f"  • Training sequences: {len(train_dataset)}")
    print(f"  • Validation sequences: {len(val_dataset)}")

    # Model
    model = LSTMSeq2SeqMass(
        in_hist=len(hist_feature_cols),
        in_fut=len(fut_feature_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        horizon=prediction_horizon
    ).to(device)

    criterion = MixedLoss(0.7, 0.3) if use_mixed_loss else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter, max_patience = 0, 20

    print(f"\n{'-'*70}")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15}")
    print(f"{'-'*70}")

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        tr_losses = []
        for x_hist, x_fut, y_seq in train_loader:
            x_hist, x_fut, y_seq = x_hist.to(device), x_fut.to(device), y_seq.to(device)
            optimizer.zero_grad()
            pred = model(x_hist, x_fut)['p50']
            loss = criterion(pred, y_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        # ---- val ----
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x_hist, x_fut, y_seq in val_loader:
                x_hist, x_fut, y_seq = x_hist.to(device), x_fut.to(device), y_seq.to(device)
                pred = model(x_hist, x_fut)['p50']
                loss = criterion(pred, y_seq)
                va_losses.append(loss.item())

        avg_train = np.mean(tr_losses) if tr_losses else 0.0
        avg_val = np.mean(va_losses) if va_losses else 0.0
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), CV_MODEL_DIR / f"best_model_fold{fold}_seed{seed}.pt")
        else:
            patience_counter += 1

        # Print logs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {avg_train:<15.6f} {avg_val:<15.6f} {best_val_loss:<15.6f}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n✓ Fold {fold+1} Seed {seed} training complete")
    print(f"  Best validation loss: {best_val_loss:.6f}")

    return history, best_val_loss


def main_cv(n_splits=5, n_seeds=5,
            test_ratio=0.2,
            batch_size=32,
            num_epochs=100,
            learning_rate=0.001,
            use_log_transform=True,
            use_mixed_loss=True,
            device='cpu',
            exp_tag=None):
    """
    Main cross-validation function - Multi-seed version

    Args:
        n_splits: K value for K-fold (default 5)
        n_seeds: Number of seeds per fold (default 5)
        ...
    """
    # Set experiment tag
    if exp_tag is None:
        exp_tag = f"cv{n_splits}_seeds{n_seeds}_stratified"

    global CV_MODEL_DIR
    CV_MODEL_DIR = PROJECT_ROOT / "models" / exp_tag
    CV_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MULTI-SEED GROUP K-FOLD CROSS-VALIDATION TRAINING")
    print("="*70)
    print(f"  • Number of folds: {n_splits}")
    print(f"  • Seeds per fold: {n_seeds}")
    print(f"  • Total models: {n_splits * n_seeds}")
    print(f"  • Experiment tag: {exp_tag}")
    print("="*70)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df)} records from {df['project_id'].nunique()} projects")
    raw_df = df.copy()

    # Add enhanced features
    df = add_enhanced_features(df)

    # Log transform
    if use_log_transform:
        df['total_cost'] = np.log1p(df['total_cost'])
        print(f"Applied log1p transform to total_cost")

    # Historical/future columns (consistent with Train.py)
    hist_feature_cols = [
        'progress_pct', 'mat_index', 'lab_index', 'cpi_index',
        'material_cost', 'labour_cost', 'equip_cost', 'admin_cost',
        'total_cost', 'month', 'normalized_time',
        'remaining_months', 'is_completion_phase'
    ]
    fut_feature_cols = [
        'month', 'normalized_time', 'remaining_months',
        'is_completion_phase', 'cpi_index', 'mat_index', 'lab_index'
    ]
    union_cols = sorted(set(hist_feature_cols) | set(fut_feature_cols))


    # ==== OUTER TEST SPLIT (stratified + seeded) ====
    TEST_SIZE = int(df['project_id'].nunique() * test_ratio)
    BASE_RANDOM_SEED = 2025

    # Set base seed
    set_all_seeds(BASE_RANDOM_SEED)

    # Use raw_df for stratification
    meta = (raw_df.groupby('project_id')
            .agg(project_type_name=('project_type_name', 'first'),
                 total_cost_sum=('total_cost', 'sum'))
            )

    # Cost quartiles for stratification
    meta['cost_q'] = pd.qcut(meta['total_cost_sum'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    meta['strata'] = meta['project_type_name'].astype(str) + '|Q' + meta['cost_q'].astype(str)

    # Allocate slots proportionally by stratum
    counts = meta['strata'].value_counts(normalize=True)
    alloc = (counts * TEST_SIZE).astype(int)
    remainder = TEST_SIZE - alloc.sum()
    for s in counts.sort_values(ascending=False).index[:remainder]:
        alloc[s] += 1

    # Fixed seed stratified sampling
    rng = np.random.default_rng(BASE_RANDOM_SEED)
    test_projects = []
    for s, k in alloc.items():
        if k <= 0:
            continue
        cand = meta[meta['strata'] == s].index.tolist()
        test_projects.extend(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())

    # Train+validation projects
    all_projects = sorted(df['project_id'].unique())
    train_val_projects = [p for p in all_projects if p not in test_projects]

    print(f"\n  Data split:")
    print(f"  • Total projects: {len(all_projects)}")
    print(f"  • Train+Val projects: {len(train_val_projects)} (for CV)")
    print(f"  • Test projects: {len(test_projects)} (held out)")

    # Save test set info
    test_info = {
        'mode': 'stratified_random',
        'base_random_seed': BASE_RANDOM_SEED,
        'n_seeds_per_fold': n_seeds,
        'test_projects': test_projects,
        'n_test': len(test_projects),
        'test_ratio': test_ratio
    }
    with open(CV_MODEL_DIR / "test_projects.json", 'w', encoding='utf-8') as f:
        json.dump(test_info, f, indent=2)


    # Use only train+validation projects for cross-validation
    df_train_val = df[df['project_id'].isin(train_val_projects)].copy()

    # GroupKFold split
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_train_val['project_id'].values
    fold_map = {}

    print(f"\nCross-validation setup:")
    print(f"  CV projects: {len(train_val_projects)}")
    print(f"  Number of folds: {n_splits}")
    print(f"  Seeds per fold: {n_seeds}")

    # Store all model metrics
    all_model_metrics = []
    fold_histories = []

    # ============================================================
    # NEW: Double loop - Fold × Seed
    # ============================================================
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_train_val, groups=groups)):
        fold_display = fold + 1

        # Record train/validation project IDs for this fold
        train_projects_fold = sorted(df_train_val.iloc[train_idx]['project_id'].unique().tolist())
        val_projects_fold = sorted(df_train_val.iloc[val_idx]['project_id'].unique().tolist())

        fold_map[f"fold_{fold_display}"] = {
            "train": train_projects_fold,
            "val": val_projects_fold
        }

        print("\n" + "=" * 70)
        print(f"FOLD {fold_display}/{n_splits} - Training {n_seeds} seeds")
        print("=" * 70)
        print(f"  • Train projects ({len(train_projects_fold)}): {', '.join(map(str, train_projects_fold))}")
        print(f"  • Validation projects ({len(val_projects_fold)}): {', '.join(map(str, val_projects_fold))}")

        # Split data using indices directly
        train_df = df_train_val.iloc[train_idx].copy()
        val_df = df_train_val.iloc[val_idx].copy()

        # Preserve original columns
        preserved_cols = ['project_id']
        if 'project_type_name' in train_df.columns:
            preserved_cols.append('project_type_name')
        if 'total_duration_months' in train_df.columns:
            preserved_cols.append('total_duration_months')

        train_preserved = train_df[preserved_cols].copy()
        val_preserved = val_df[preserved_cols].copy()

        # Standardization (fit only on training set, each fold shares one scaler)
        scaler = StandardScaler()

        for c in union_cols:
            train_df[c] = pd.to_numeric(train_df[c], errors='coerce')
            val_df[c] = pd.to_numeric(val_df[c], errors='coerce')
        train_df[union_cols] = train_df[union_cols].fillna(train_df[union_cols].mean(numeric_only=True))
        val_df[union_cols] = val_df[union_cols].fillna(train_df[union_cols].mean(numeric_only=True))

        # Scaling: use DataFrame fit/transform (preserve column names -> scaler.feature_names_in_)
        scaler.fit(train_df[union_cols])
        train_df[union_cols] = scaler.transform(train_df[union_cols])
        val_df[union_cols] = scaler.transform(val_df[union_cols])

        # Save one scaler per fold (shared by all seeds)
        joblib.dump(scaler, CV_MODEL_DIR / f"scaler_fold{fold}.pkl")

        # Restore preserved columns
        for col in preserved_cols:
            train_df[col] = train_preserved[col].values
            val_df[col] = val_preserved[col].values

        # ============================================================
        # NEW: Inner loop - Multi-seed training
        # ============================================================
        seed_metrics = []

        for seed_idx in range(n_seeds):
            # Calculate current seed value: base seed + fold*1000 + seed_idx
            current_seed = BASE_RANDOM_SEED + fold * 1000 + seed_idx

            # Set all random seeds
            set_all_seeds(current_seed)

            print(f"\n{'─'*70}")
            print(f"  Training Fold {fold_display} | Seed {seed_idx+1}/{n_seeds} (seed_value={current_seed})")
            print(f"{'─'*70}")

            # Train model
            history, best_val_loss = train_single_fold(
                train_df, val_df,
                hist_feature_cols=hist_feature_cols,
                fut_feature_cols=fut_feature_cols,
                fold=fold,
                seed=seed_idx,
                sequence_length=12,
                prediction_horizon=3,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                use_mixed_loss=use_mixed_loss
            )

            # Save config (each fold+seed combination)
            config = {
                'fold': fold,
                'seed': seed_idx,
                'seed_value': current_seed,
                'hist_feature_cols': hist_feature_cols,
                'fut_feature_cols': fut_feature_cols,
                'feature_cols': union_cols,
                'sequence_length': 12,
                'prediction_horizon': 3,
                'horizon': 3,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'use_log_transform': use_log_transform,
                'train_projects': train_projects_fold,
                'val_projects': val_projects_fold
            }

            config_path = CV_MODEL_DIR / f"model_config_fold{fold}_seed{seed_idx}.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # Record metrics
            seed_metrics.append({
                'fold': fold + 1,
                'seed': seed_idx,
                'seed_value': current_seed,
                'best_val_loss': float(best_val_loss),
                'final_train_loss': float(history['train_loss'][-1])
            })

        # Calculate average validation loss for all seeds in this fold
        avg_fold_val_loss = np.mean([m['best_val_loss'] for m in seed_metrics])

        print(f"\n{'='*70}")
        print(f"FOLD {fold_display} SUMMARY ({n_seeds} seeds)")
        print(f"{'='*70}")
        print(f"  • Average validation loss: {avg_fold_val_loss:.6f}")
        print(f"  • Val loss range: {min(m['best_val_loss'] for m in seed_metrics):.6f} - {max(m['best_val_loss'] for m in seed_metrics):.6f}")
        print(f"{'='*70}")

        # Aggregate metrics for this fold (for cv_summary.json)
        all_model_metrics.extend(seed_metrics)
        fold_histories.append({
            'fold': fold + 1,
            'avg_val_loss': avg_fold_val_loss,
            'seeds': seed_metrics
        })

    # ============================================================
    # Save CV summary
    # ============================================================
    cv_summary = {
        'n_splits': n_splits,
        'n_seeds_per_fold': n_seeds,
        'total_models': n_splits * n_seeds,
        'base_random_seed': BASE_RANDOM_SEED,
        'test_projects': test_projects,
        'fold_details': [],
        'all_models': all_model_metrics,
        'experiment_tag': exp_tag
    }

    # Aggregate by fold
    for fold_idx in range(n_splits):
        fold_models = [m for m in all_model_metrics if m['fold'] == fold_idx + 1]
        avg_val = np.mean([m['best_val_loss'] for m in fold_models])

        cv_summary['fold_details'].append({
            'fold': fold_idx + 1,
            'best_val_loss': float(avg_val),
            'n_seeds': n_seeds,
            'val_loss_std': float(np.std([m['best_val_loss'] for m in fold_models]))
        })

    # Overall statistics
    all_val_losses = [m['best_val_loss'] for m in all_model_metrics]
    cv_summary['overall'] = {
        'mean_val_loss': float(np.mean(all_val_losses)),
        'std_val_loss': float(np.std(all_val_losses)),
        'min_val_loss': float(np.min(all_val_losses)),
        'max_val_loss': float(np.max(all_val_losses))
    }

    summary_path = CV_MODEL_DIR / "cv_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"  • Total models trained: {n_splits * n_seeds}")
    print(f"  • Overall mean val loss: {cv_summary['overall']['mean_val_loss']:.6f} ± {cv_summary['overall']['std_val_loss']:.6f}")
    print(f"  • Results saved to: {CV_MODEL_DIR}")
    print(f"{'='*70}\n")

    return cv_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-seed Cross-Validation Training")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds per fold (default: 5)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--exp_tag", type=str, default=None, help="Experiment tag for output directory")

    args = parser.parse_args()

    main_cv(
        n_splits=args.n_splits,
        n_seeds=args.n_seeds,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        exp_tag=args.exp_tag
    )