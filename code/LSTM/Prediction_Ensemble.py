"""
Prediction_Ensemble_MultiSeed.py - Multi-seed Ensemble Evaluation
Purpose: Evaluate multi-seed ensemble performance on held-out test projects

NEW FEATURES:
- Use 25 models (5 folds × 5 seeds) for ensemble prediction
- Compare performance of different ensemble strategies
- Validate improvement of multi-seed ensemble over single-seed
- Generate detailed performance reports

Based on: Prediction_Ensemble.py
Author: Enhanced for multi-seed ensemble evaluation
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import json
from pathlib import Path
from LSTM_Model import LSTMSeq2SeqMass
from Train_CV import add_enhanced_features
from Prediction import predict_future_costs, calculate_metrics, _get_cfg_cols

# ======== Configuration  ========
AGG_MODE = "median"
EPS = 1e-8

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "generated" / "synthetic_CN_projects.csv"

sns.set_style("whitegrid")


# ============================================================
# Helper Functions
# ============================================================

def _auto_scale_predictions(preds_WH: np.ndarray,
                            acts_WH: np.ndarray,
                            method: str = "per_window",
                            trigger: float = 2.0,
                            clamp_ratio: tuple = (1e-4, 1e4),
                            verbose: bool = True):
    """Scale alignment"""
    preds = np.asarray(preds_WH, dtype=float)
    acts = np.asarray(acts_WH, dtype=float)
    assert preds.shape == acts.shape and preds.ndim == 2

    W, H = preds.shape
    sum_p = np.clip(preds.sum(axis=1), 1e-8, None)
    sum_a = acts.sum(axis=1)
    ratio = sum_a / sum_p
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)

    lo, hi = clamp_ratio

    if method == "global":
        ratio_med = float(np.median(ratio))
        if ratio_med > trigger or ratio_med < 1.0/trigger:
            ratio_use = np.clip(ratio_med, lo, hi)
            preds_scaled = preds * ratio_use
            stats = {"mode": "global", "used_ratio": ratio_use, "n_scaled": W}
            if verbose:
                print(f"Global scale alignment ×{ratio_use:.4g}")
            return preds_scaled, stats
        else:
            return preds, {"mode": "global", "used_ratio": 1.0, "n_scaled": 0}

    # per-window
    mask = (ratio > trigger) | (ratio < 1.0/trigger)
    ratio_use = np.clip(ratio, lo, hi)
    preds_scaled = preds.copy()
    preds_scaled[mask, :] = (preds[mask, :].T * ratio_use[mask]).T

    n_scaled = int(np.count_nonzero(mask))
    stats = {
        "mode": "per_window",
        "n_windows": W,
        "n_scaled": n_scaled,
        "ratio_median": float(np.median(ratio)),
    }
    if verbose and n_scaled > 0:
        print(f"Scaled {n_scaled}/{W} windows (median ratio={stats['ratio_median']:.3g})")
    return preds_scaled, stats


def prepare_raw_test_data(test_project_ids):
    """Prepare test data"""
    df = pd.read_csv(DATA_PATH)
    test_df = df[df['project_id'].isin(test_project_ids)].copy()
    test_df = add_enhanced_features(test_df)
    return test_df


def _scale_project_for_prediction(raw_df, scaler, config):
    """Standardize project data"""
    use_log = config.get('use_log_transform', True)

    df = raw_df.copy()
    if use_log and 'total_cost' in df.columns:
        if df['total_cost'].max() > 20:
            df['total_cost'] = np.log1p(df['total_cost'])

    # Get feature columns
    if 'feature_cols' in config:
        feature_cols = config['feature_cols']
    elif hasattr(scaler, 'feature_names_in_'):
        feature_cols = list(scaler.feature_names_in_)
    else:
        raise ValueError("Cannot determine feature columns")

    # Standardize
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[feature_cols] = df[feature_cols].fillna(0.0)

    scaled = scaler.transform(df[feature_cols].values)
    df_scaled = pd.DataFrame(scaled, columns=feature_cols, index=df.index)

    return df_scaled


def _normalize_predict_output(result, H=3):
    """Normalize prediction output"""
    if 'predictions' not in result:
        raise ValueError("Result missing 'predictions' key")

    preds = result['predictions']
    if not preds:
        raise ValueError("Empty predictions")

    # Convert to matrix
    pred_matrix = []
    actual_matrix = []

    for pred_seq in preds:
        if len(pred_seq) == H:
            pred_matrix.append(pred_seq)

    if 'actuals' in result:
        for act_seq in result['actuals']:
            if len(act_seq) == H:
                actual_matrix.append(act_seq)

    return {
        'pred_matrix': np.array(pred_matrix),
        'actual_matrix': np.array(actual_matrix),
    }


def calculate_metrics_original_scale(preds, acts, flags=None):
    """Calculate performance metrics"""
    preds = np.asarray(preds, dtype=float).flatten()
    acts = np.asarray(acts, dtype=float).flatten()

    abs_err = np.abs(preds - acts)
    sq_err = (preds - acts) ** 2

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))

    # R²
    ss_res = float(np.sum(sq_err))
    ss_tot = float(np.sum((acts - acts.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.mean(np.abs((acts - preds) / np.maximum(acts, 1e-8))) * 100)

    # sMAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = float(np.mean(2.0 * abs_err / (np.abs(preds) + np.abs(acts) + 1e-8)) * 100)

    # WAPE
    wape = float(np.sum(abs_err) / (np.sum(np.abs(acts)) + 1e-8) * 100)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "sMAPE": smape,
        "WAPE": wape,
        "N": len(preds)
    }

    return metrics


# ============================================================
# Multi-seed Ensemble Functions
# ============================================================

def load_multiseed_models(cv_model_dir: Path, n_folds=5, n_seeds=5, device='cpu'):
    """
    Load all fold×seed combination models

    Returns:
        models: dict {(fold, seed): model}
        scalers: dict {fold: scaler}
        configs: dict {(fold, seed): config}
        val_losses: dict {(fold, seed): val_loss}
    """
    print("\n" + "="*70)
    print("LOADING MULTI-SEED MODELS")
    print("="*70)

    models = {}
    scalers = {}
    configs = {}
    val_losses = {}

    # Load cv_summary for validation losses
    cv_summary_path = cv_model_dir / "cv_summary.json"
    if cv_summary_path.exists():
        with open(cv_summary_path, 'r') as f:
            cv_summary = json.load(f)
        all_models_info = cv_summary.get('all_models', [])
    else:
        all_models_info = []

    loaded_count = 0
    failed_count = 0

    for fold in range(n_folds):
        # Load scaler (shared across seeds)
        scaler_path = cv_model_dir / f"scaler_fold{fold}.pkl"
        if scaler_path.exists():
            scalers[fold] = joblib.load(scaler_path)
        else:
            print(f"Scaler not found: {scaler_path}")
            continue

        for seed in range(n_seeds):
            # Load config
            config_path = cv_model_dir / f"model_config_fold{fold}_seed{seed}.json"
            if not config_path.exists():
                print(f"Config not found for fold{fold}_seed{seed}")
                failed_count += 1
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)
            configs[(fold, seed)] = config

            # Get validation loss
            for model_info in all_models_info:
                if model_info.get('fold') == fold + 1 and model_info.get('seed') == seed:
                    val_losses[(fold, seed)] = model_info.get('best_val_loss', 999.0)
                    break
            else:
                val_losses[(fold, seed)] = 999.0

            # Load model
            model_path = cv_model_dir / f"best_model_fold{fold}_seed{seed}.pt"
            if not model_path.exists():
                print(f"Model not found for fold{fold}_seed{seed}")
                failed_count += 1
                continue

            try:
                # Create model
                model = LSTMSeq2SeqMass(
                    in_hist=len(config.get('hist_feature_cols', [])),
                    in_fut=len(config.get('fut_feature_cols', [])),
                    hidden_size=config.get('hidden_size', 128),
                    num_layers=config.get('num_layers', 2),
                    dropout=config.get('dropout', 0.2),
                    horizon=config.get('horizon', 3)
                ).to(device)

                # Load weights
                ckpt = torch.load(model_path, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=True)
                model.eval()

                models[(fold, seed)] = model
                loaded_count += 1

            except Exception as e:
                print(f"Failed to load model fold{fold}_seed{seed}: {e}")
                failed_count += 1
                continue

    print(f"\nLoaded {loaded_count} models successfully")
    if failed_count > 0:
        print(f"Failed to load {failed_count} models")
    print(f"Total expected: {n_folds * n_seeds}, Loaded: {loaded_count}")

    return models, scalers, configs, val_losses


def predict_multiseed_ensemble(
    models, scalers, configs, val_losses,
    test_project_ids,
    agg_mode="median",
    device='cpu'
):
    """
    Multi-seed ensemble prediction on test projects

    Args:
        models: dict {(fold, seed): model}
        scalers: dict {fold: scaler}
        configs: dict {(fold, seed): config}
        val_losses: dict {(fold, seed): val_loss}
        test_project_ids: list of test project IDs
        agg_mode: ensemble aggregation mode
        device: computation device

    Returns:
        preds_arr: predictions (n_windows, H)
        acts_arr: actuals (n_windows, H)
    """
    print("\n" + "="*70)
    print(f"ENSEMBLE PREDICTION ({agg_mode.upper()})")
    print("="*70)
    print(f"Using {len(models)} models for ensemble")

    H = 3  # Fixed horizon
    test_df_raw = prepare_raw_test_data(test_project_ids)

    all_preds_ensemble = []
    all_actuals = []

    for pid in test_project_ids:
        print(f"\nPredicting project: {pid}")
        project_data_raw = test_df_raw[test_df_raw['project_id'] == pid].copy()

        # Collect predictions from all models
        project_preds = []
        project_actuals = None

        for (fold, seed), model in models.items():
            scaler = scalers[fold]
            config = configs[(fold, seed)]

            sequence_length = config.get('sequence_length', 12)

            # Scale data
            try:
                scaled_df = _scale_project_for_prediction(project_data_raw, scaler, config)
            except Exception as e:
                print(f"Scaling failed for fold{fold}_seed{seed}: {e}")
                continue

            # Check if enough data
            N = len(scaled_df)
            W = N - sequence_length - H + 1
            if W <= 0:
                continue

            # Predict
            try:
                res = predict_future_costs(
                    model=model,
                    project_df_scaled=scaled_df,
                    config=config,
                    scaler=scaler,
                    device=device,
                    horizon=H
                )

                if not res:
                    continue

                norm = _normalize_predict_output(res, H=H)
                pred_matrix = np.asarray(norm['pred_matrix'], dtype=float)
                actual_matrix = np.asarray(norm['actual_matrix'], dtype=float)

                project_preds.append(pred_matrix)

                if project_actuals is None:
                    project_actuals = actual_matrix

            except Exception as e:
                print(f"Prediction failed for fold{fold}_seed{seed}: {e}")
                continue

        if len(project_preds) == 0:
            print(f"No valid predictions for project {pid}")
            continue

        # Ensemble predictions
        project_preds = np.array(project_preds)

        if agg_mode == "median":
            ensemble_pred = np.median(project_preds, axis=0)
        elif agg_mode == "mean":
            ensemble_pred = np.mean(project_preds, axis=0)
        elif agg_mode == "weighted":

            weights = []
            for (fold, seed) in models.keys():
                val_loss = val_losses.get((fold, seed), 1.0)
                weights.append(1.0 / val_loss)
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = np.average(project_preds, axis=0, weights=weights)
        else:
            ensemble_pred = np.median(project_preds, axis=0)

        all_preds_ensemble.extend(ensemble_pred.tolist())
        all_actuals.extend(project_actuals.tolist())

        print(f"Ensemble: {len(project_preds)} models → {len(ensemble_pred)} windows")

    print(f"\nTotal windows: {len(all_preds_ensemble)}")

    # Convert to arrays and scale
    preds_arr = np.asarray(all_preds_ensemble, dtype=float)
    acts_arr = np.asarray(all_actuals, dtype=float)

    preds_arr, _ = _auto_scale_predictions(
        preds_arr, acts_arr,
        method="per_window",
        trigger=2.0,
        verbose=True
    )

    return preds_arr, acts_arr


def compare_ensemble_sizes(
    models, scalers, configs, val_losses,
    test_project_ids,
    agg_mode="median",
    device='cpu'
):
    """
    Compare performance of different ensemble sizes

    Returns:
        results: dict with performance for different ensemble sizes
    """
    print("\n" + "="*70)
    print("COMPARING ENSEMBLE SIZES")
    print("="*70)

    results = {}

    # Test different ensemble sizes
    ensemble_configs = [
        ("Single Best", 1),
        ("5-Model (1 per fold)", 5),
        ("25-Model (5×5 multi-seed)", len(models))
    ]

    for name, n_models in ensemble_configs:
        print(f"\n[{name}] Using {n_models} model(s)")

        if n_models == 1:
            # Best single model
            best_key = min(val_losses.keys(), key=lambda k: val_losses[k])
            subset_models = {best_key: models[best_key]}
            subset_scalers = {best_key[0]: scalers[best_key[0]]}
            subset_configs = {best_key: configs[best_key]}
            subset_losses = {best_key: val_losses[best_key]}

        elif n_models == 5:
            # One best model per fold
            subset_models = {}
            subset_scalers = {}
            subset_configs = {}
            subset_losses = {}

            for fold in range(5):
                # Find best seed for this fold
                fold_keys = [(f, s) for (f, s) in models.keys() if f == fold]
                if fold_keys:
                    best_key = min(fold_keys, key=lambda k: val_losses[k])
                    subset_models[best_key] = models[best_key]
                    subset_scalers[fold] = scalers[fold]
                    subset_configs[best_key] = configs[best_key]
                    subset_losses[best_key] = val_losses[best_key]

        else:
            # All models
            subset_models = models
            subset_scalers = scalers
            subset_configs = configs
            subset_losses = val_losses

        # Predict
        preds, acts = predict_multiseed_ensemble(
            subset_models, subset_scalers, subset_configs, subset_losses,
            test_project_ids, agg_mode, device
        )

        # Calculate metrics
        metrics = calculate_metrics_original_scale(preds, acts)
        results[name] = metrics

        # Print summary
        print(f"\n{name} Results:")
        print(f"  MAE:  {metrics['MAE']:,.2f}")
        print(f"  RMSE: {metrics['RMSE']:,.2f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")

    return results


# ============================================================
# Main Function
# ============================================================

def main(cv_model_dir: str, n_folds=5, n_seeds=5, agg_mode="median"):
    """
    Main function

    Args:
        cv_model_dir: CV model directory
        n_folds: Number of folds
        n_seeds: Number of seeds per fold
        agg_mode: Ensemble aggregation mode
    """
    cv_model_dir = Path(cv_model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print("MULTI-SEED ENSEMBLE EVALUATION")
    print("="*70)
    print(f"CV Model Dir: {cv_model_dir}")
    print(f"N Folds:      {n_folds}")
    print(f"N Seeds:      {n_seeds}")
    print(f"Total Models: {n_folds * n_seeds}")
    print(f"Agg Mode:     {agg_mode.upper()}")
    print(f"Device:       {device}")
    print("="*70)

    # Load test projects
    test_info_path = cv_model_dir / "test_projects.json"
    if test_info_path.exists():
        with open(test_info_path, 'r') as f:
            test_info = json.load(f)
        test_project_ids = test_info['test_projects']
        print(f"\nTest projects: {test_project_ids}")
    else:
        raise FileNotFoundError(f"Test projects file not found: {test_info_path}")

    # Load models
    models, scalers, configs, val_losses = load_multiseed_models(
        cv_model_dir, n_folds, n_seeds, device
    )

    if len(models) == 0:
        print("\nNo models loaded! Exiting.")
        return

    # Compare ensemble sizes
    results = compare_ensemble_sizes(
        models, scalers, configs, val_losses,
        test_project_ids, agg_mode, device
    )

    # Print comparison table
    print("\n" + "="*70)
    print("ENSEMBLE SIZE COMPARISON")
    print("="*70)
    print(f"{'Method':<30} {'MAE':<15} {'RMSE':<15} {'R²':<10} {'MAPE':<10}")
    print("-"*70)

    for name, metrics in results.items():
        print(f"{name:<30} {metrics['MAE']:>14,.2f} {metrics['RMSE']:>14,.2f} "
              f"{metrics['R2']:>9.4f} {metrics['MAPE']:>9.2f}%")

    # Calculate improvement
    if "25-Model (5×5 multi-seed)" in results and "5-Model (1 per fold)" in results:
        mae_5 = results["5-Model (1 per fold)"]["MAE"]
        mae_25 = results["25-Model (5×5 multi-seed)"]["MAE"]
        improvement = (mae_5 - mae_25) / mae_5 * 100

        print("\n" + "="*70)
        print("MULTI-SEED IMPROVEMENT")
        print("="*70)
        print(f"5-Model MAE:  {mae_5:,.2f}")
        print(f"25-Model MAE: {mae_25:,.2f}")
        print(f"Improvement:  {improvement:.1f}%")
        print("="*70)

    # Save results
    output_dir = cv_model_dir / "ensemble_evaluation"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "multiseed_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-seed Ensemble Evaluation")
    parser.add_argument("--cv_model_dir", type=str, required=True,
                        help="CV model directory")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--agg_mode", type=str, default="median",
                        choices=["median", "mean", "weighted"])

    args = parser.parse_args()

    main(
        cv_model_dir=args.cv_model_dir,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
        agg_mode=args.agg_mode
    )