import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
from LSTM_Model import LSTMSeq2SeqMass as LSTMCostForecast

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR    = PROJECT_ROOT / "models"

DATA_PATH    = PROJECT_ROOT / "data" / "generated" / "synthetic_CN_projects.csv"
MODEL_PATH   = MODEL_DIR / "lstm_cost_forecast.pth"
CONFIG_PATH  = MODEL_DIR / "model_config.json"
SCALER_PATH  = MODEL_DIR / "scaler.pkl"

OUT_DIR      = MODEL_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDUSTRY_CONFIG_PATH = PROJECT_ROOT / "industry_config.json"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# ===========================
# Load Model and Configuration
# ===========================
def load_trained_model(model_path='models/lstm_cost_forecast.pth',
                       config_path='models/model_config.json',
                       scaler_path='models/scaler.pkl',
                       device='cpu'):
    """
    Load trained model, configuration, and scaler

    Args:
        model_path: Path to saved model
        config_path: Path to configuration JSON
        scaler_path: Path to saved scaler
        device: Device to load model on

    Returns:
        Tuple of (model, config, scaler)
    """
    print("\n" + "=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {config_path}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler from: {scaler_path}")

    # Initialize model
    hist_cols, fut_cols = _get_cfg_cols(config)
    model = LSTMCostForecast(
        in_hist=len(hist_cols),
        in_fut=len(fut_cols),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        horizon=config.get('horizon', 12)
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        train_loss = checkpoint.get('train_loss')
        val_loss = checkpoint.get('val_loss')
    else:
        # likely a pure state_dict saved via torch.save(model.state_dict(), ...)
        state_dict = checkpoint
        train_loss = None
        val_loss = None

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    if train_loss is not None and val_loss is not None:
        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")


    config.setdefault('feature_cols_hist', config.get('hist_feature_cols', hist_cols))
    config.setdefault('feature_cols_fut', config.get('fut_feature_cols', fut_cols))
    config.setdefault('horizon', config.get('prediction_horizon', config.get('horizon', 12)))

    return model, config, scaler


def _load_industry_config(cfg_path):
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Industry config not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        icfg = json.load(f)
    # Allows both radix 1.x and 100.x, unified to ~1.00
    for k in ['cpi_index', 'material_index', 'labour_index']:
        if k in icfg:
            arr = icfg[k]
            if max(arr) > 5:
                icfg[k] = [x/100.0 for x in arr]
    return icfg


def _get_cfg_cols(config):

    if (
        ('feature_cols_hist' in config or 'hist_feature_cols' in config)
        and ('feature_cols_fut' in config or 'fut_feature_cols' in config)
    ):
        hist_cols = config.get('feature_cols_hist', config.get('hist_feature_cols', []))
        fut_cols  = config.get('feature_cols_fut',  config.get('fut_feature_cols',  []))
        return hist_cols, fut_cols

    elif 'feature_cols_hist' in config:
        hist_cols = config['feature_cols_hist']
        fut_cols  = config.get('feature_cols_fut', [])
        return hist_cols, fut_cols

    elif 'hist_feature_cols' in config:
        hist_cols = config['hist_feature_cols']
        fut_cols  = config.get('fut_feature_cols', [])
        return hist_cols, fut_cols

    elif 'feature_cols' in config:
        all_cols = config['feature_cols']
        hist_cols = all_cols[:]  # 保留旧逻辑
        fut_cols = [c for c in [
            'month', 'normalized_time', 'remaining_months',
            'is_completion_phase', 'cpi_index', 'mat_index',
            'lab_index', 'plan_monthly_share'
        ] if c in all_cols]

        base_time = ['month', 'normalized_time', 'remaining_months', 'is_completion_phase']
        for c in base_time:
            if c not in fut_cols and c in all_cols:
                fut_cols.append(c)
        return hist_cols, fut_cols

    else:
        raise KeyError("Configuration must contain one of ['feature_cols_hist', 'hist_feature_cols', or 'feature_cols'].")


def _prepare_future_covariates(total_duration, month_start, horizon, icfg, fut_cols):
    """
    Construct a DataFrame (unscaled) with covariant `future horizon month`, column order = `fut_cols`
    icfg: _load_industry_config()` result
    """
    rows = []
    for t in range(horizon):
        m = month_start + t
        row = {
            'month': m,
            'normalized_time': m / total_duration if total_duration > 0 else 0.0,
            'remaining_months': max(total_duration - m, 0),
            'is_completion_phase': 1.0 if (total_duration - m) <= 2 else 0.0,
            'cpi_index': icfg.get('cpi_index', [1.0]*100)[(m-1) % len(icfg.get('cpi_index',[1.0]))],
            'mat_index': icfg.get('material_index', [1.0]*100)[(m-1) % len(icfg.get('material_index',[1.0]))],
            'lab_index': icfg.get('labour_index', [1.0]*100)[(m-1) % len(icfg.get('labour_index',[1.0]))],
            'plan_monthly_share': icfg.get('plan_monthly_share', [0.0]*100)[(m-1) % len(icfg.get('plan_monthly_share',[0.0]))],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    # Keep only/reorder as needed to fut_cols
    keep = [c for c in fut_cols if c in df.columns]
    return df[keep].copy()


# ===========================
# Prepare Test Data
# ===========================
def prepare_test_data(data_path, test_project_ids, scaler, feature_cols, sequence_length=12):
    """
    Prepare test data for prediction

    Args:
        data_path: Path to data CSV
        test_project_ids: List of test project IDs
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        sequence_length: Sequence length for model input

    Returns:
        DataFrame with test data
    """
    print("\n" + "=" * 60)
    print("PREPARING TEST DATA")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_path)

    try:
        from LSTM.Train_CV import add_enhanced_features
    except Exception:
        from LSTM.Train import add_enhanced_features
    df = add_enhanced_features(df)

    # Perform log1p transformation on total_cost
    df['total_cost'] = np.log1p(df['total_cost'])

    # Screening test items
    test_df = df[df['project_id'].isin(test_project_ids)].copy()

    print(f"Loaded {len(test_df)} records from {len(test_project_ids)} test projects")

    # Preserve the original important columns such as total_duration_months and project_type_name.
    preserved_cols = ['project_id', 'project_type_name']
    if 'total_duration_months' in test_df.columns:
        preserved_cols.append('total_duration_months')
    preserved_data = test_df[preserved_cols].copy()

    scaler_cols = list(getattr(scaler, 'feature_names_in_', []))
    if not scaler_cols:
        scaler_cols = list(feature_cols)

    # Validate for missing and redundant columns, and provide clear error messages.
    missing = [c for c in scaler_cols if c not in test_df.columns]
    extra = [c for c in test_df.columns if c in feature_cols and c not in scaler_cols]
    if missing:
        raise ValueError(
            "Test data is missing columns required by the scaler.\n"
            f"Missing: {missing}\n"
            f"Scaler expects (ordered): {scaler_cols}\n"
            f"Available in test_df: {[c for c in test_df.columns if c in set(scaler_cols + feature_cols)]}"
        )

    # Perform type conversion and scaling only on scaler_cols
    test_df[scaler_cols] = test_df[scaler_cols].astype("float64")
    X_scaled = scaler.transform(test_df[scaler_cols])
    test_df[scaler_cols] = X_scaled

    # Merge back the original columns that were retained (if they were lost during scaling).
    for col in preserved_cols:
        if col not in test_df.columns or col in scaler_cols:
            test_df[col] = preserved_data[col].values

    return test_df


# ===========================
# Make Predictions
# ===========================
def predict_future_costs(model, project_df_scaled, config, scaler,
                         device='cpu', horizon=None, industry_config_path=INDUSTRY_CONFIG_PATH):

    model.eval()
    hist_cols, fut_cols = _get_cfg_cols(config)
    seq_len   = config.get('sequence_length', 12)
    horizon   = horizon or config.get('horizon', 12)


    if 'total_duration_months' in project_df_scaled.columns:
        total_duration = float(project_df_scaled.iloc[0]['total_duration_months'])
    else:

        scaler_cols = list(scaler.feature_names_in_)
        row0 = project_df_scaled.iloc[0]
        dummy = np.zeros((1, scaler.n_features_in_), dtype=float)
        for c in ('month', 'remaining_months'):
            if c in scaler_cols and c in row0.index:
                dummy[0, scaler_cols.index(c)] = float(row0[c])
        raw0 = scaler.inverse_transform(dummy)[0]
        total_duration = float(raw0[scaler_cols.index('month')] + raw0[scaler_cols.index('remaining_months')])

    icfg = _load_industry_config(industry_config_path)

    predictions_scaled = []
    actuals_scaled     = []
    months_scaled      = []

    project_df_scaled = project_df_scaled.sort_values('month').reset_index(drop=True)
    n = len(project_df_scaled)
    max_start = n - seq_len - horizon
    if max_start < 0:
        return None

    for start in range(max_start + 1):
        hist_block = project_df_scaled.iloc[start:start+seq_len][hist_cols].to_numpy()

        scaler_cols = list(scaler.feature_names_in_)
        row = project_df_scaled.iloc[start + seq_len - 1]
        dummy = np.zeros((1, scaler.n_features_in_), dtype=float)

        for c in hist_cols:
            if c in scaler_cols:
                dummy[0, scaler_cols.index(c)] = float(row[c])
        last_row_raw = scaler.inverse_transform(dummy)
        last_month = int(round(last_row_raw[0, scaler_cols.index('month')]))
        month_start     = last_month + 1

        fut_raw = _prepare_future_covariates(total_duration, month_start, horizon, icfg, fut_cols)
        fut_scaled = fut_raw.copy()

        scaler_cols = list(scaler.feature_names_in_)
        dummy = np.zeros((len(fut_raw), scaler.n_features_in_), dtype=float)
        for c in fut_cols:
            if c in scaler_cols:
                dummy.fill(0.0)
                idx = scaler_cols.index(c)
                dummy[:, idx] = fut_raw[c].astype(float).values
                fut_scaled[c] = scaler.transform(dummy)[:, idx]
            else:
                fut_scaled[c] = fut_raw[c].astype(float).values


        x_hist = torch.tensor(hist_block, dtype=torch.float32, device=device).unsqueeze(0)
        x_fut  = torch.tensor(fut_scaled[fut_cols].to_numpy(), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model(x_hist, x_fut)
            p50 = out['p50'].detach().cpu().numpy().reshape(-1)

        future_slice = project_df_scaled.iloc[start+seq_len:start+seq_len+horizon]
        y_true = future_slice['total_cost'].to_numpy().reshape(-1)

        predictions_scaled.append(p50.tolist())
        actuals_scaled.append(y_true.tolist())
        months_scaled.append(future_slice['month'].to_numpy().tolist())

    return {
        'predictions': predictions_scaled,
        'actuals': actuals_scaled,
        'months': months_scaled,
        'hist_cols': hist_cols,
        'fut_cols': fut_cols,
        'horizon': horizon
    }


# ===========================
# Calculate Metrics
# ===========================
def calculate_metrics(predictions, actuals, scaler, target_idx=None, is_completion_flags=None):
    """
    Calculate comprehensive prediction metrics including WAPE and step-wise metrics

    Args:
        predictions: List of prediction arrays (each array has multiple time steps)
        actuals: List of actual value arrays
        scaler: StandardScaler used for normalization
        target_idx: Index of target column in scaler

    Returns:
        Dictionary with comprehensive metrics
    """
    # Flatten predictions and actuals, keeping track of time steps
    all_preds_by_step = {1: [], 2: [], 3: []}
    all_actuals_by_step = {1: [], 2: [], 3: []}
    all_preds, all_actuals = [], []

    for pred_seq, actual_seq in zip(predictions, actuals):
        for step_idx, (pred, actual) in enumerate(zip(pred_seq, actual_seq)):
            step = step_idx + 1
            if step <= 3:
                all_preds_by_step[step].append(pred)
                all_actuals_by_step[step].append(actual)
            all_preds.append(pred)
            all_actuals.append(actual)

    all_preds = np.array(all_preds).reshape(-1, 1)
    all_actuals = np.array(all_actuals).reshape(-1, 1)

    scaler_cols = list(getattr(scaler, 'feature_names_in_', []))
    if not scaler_cols:
        raise ValueError("Scaler is missing feature_names_in_.")
    tc_idx = scaler_cols.index('total_cost') if target_idx is None else target_idx  # 兼容旧调用

    dummy = np.zeros((len(all_preds), scaler.n_features_in_), dtype=float)
    dummy[:, tc_idx] = all_preds.ravel()
    all_preds_original = np.expm1(scaler.inverse_transform(dummy)[:, tc_idx])

    dummy[:, tc_idx] = all_actuals.ravel()
    all_actuals_original = np.expm1(scaler.inverse_transform(dummy)[:, tc_idx])

    err = all_actuals_original - all_preds_original
    mae = mean_absolute_error(all_actuals_original, all_preds_original)
    rmse = np.sqrt(mean_squared_error(all_actuals_original, all_preds_original))
    r2 = r2_score(all_actuals_original, all_preds_original)

    denom = np.clip(np.abs(all_actuals_original), 1e-6, None)  # 防小分母
    mape = np.mean(np.abs(err) / denom) * 100

    wape_denom = np.sum(np.abs(all_actuals_original))
    wape = (np.sum(np.abs(err)) / max(wape_denom, 1e-6)) * 100

    step_metrics = {}
    for step in (1, 2, 3):
        if len(all_preds_by_step[step]) == 0:
            continue

        step_preds = np.array(all_preds_by_step[step]).reshape(-1, 1)
        step_actuals = np.array(all_actuals_by_step[step]).reshape(-1, 1)

        s_dummy = np.zeros((len(step_preds), scaler.n_features_in_), dtype=float)
        s_dummy[:, tc_idx] = step_preds.ravel()
        step_preds_orig = np.expm1(scaler.inverse_transform(s_dummy)[:, tc_idx])

        s_dummy[:, tc_idx] = step_actuals.ravel()
        step_actuals_orig = np.expm1(scaler.inverse_transform(s_dummy)[:, tc_idx])

        s_err = step_actuals_orig - step_preds_orig
        s_mae = mean_absolute_error(step_actuals_orig, step_preds_orig)
        s_rmse = np.sqrt(mean_squared_error(step_actuals_orig, step_preds_orig))
        s_denom = np.clip(np.abs(step_actuals_orig), 1e-6, None)
        s_mape = np.mean(np.abs(s_err) / s_denom) * 100
        s_wape = (np.sum(np.abs(s_err)) / max(np.sum(np.abs(step_actuals_orig)), 1e-6)) * 100

        step_metrics[f't+{step}'] = {
            'mae': s_mae, 'rmse': s_rmse, 'mape': s_mape, 'wape': s_wape,
            'n_samples': len(step_preds)
        }

    completion_metrics = None
    if is_completion_flags is not None:
        flags = np.asarray(is_completion_flags).astype(int).reshape(-1)
        if flags.shape[0] == all_preds_original.shape[0]:
            mask = flags == 1
            if mask.any():
                sub_pred = all_preds_original[mask]
                sub_true = all_actuals_original[mask]
                sub_err  = np.abs(sub_true - sub_pred)
                sub_wape = float(sub_err.sum() / max(np.abs(sub_true).sum(), 1e-6) * 100.0)
                sub_mae  = float(sub_err.mean())
                completion_metrics = {
                    "mae": sub_mae,
                    "wape": sub_wape,
                    "n": int(mask.sum())
                }

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'wape': wape,
        'predictions_original': all_preds_original,
        'actuals_original': all_actuals_original,
        'step_metrics': step_metrics,
        'completion_phase': completion_metrics
    }


# ===========================
# Calculate Per-Project Metrics
# ===========================

def calculate_project_metrics(test_df, test_project_ids, model, config, scaler, device='cpu'):
    """
    Calculate metrics for each project individually

    Returns:
        DataFrame with per-project metrics
    """
    hist_cols, fut_cols = _get_cfg_cols(config)
    sequence_length = config.get('sequence_length', 12)
    target_idx = hist_cols.index('total_cost')
    future_steps = 3

    project_results = []

    for pid in test_project_ids:
        project_data = test_df[test_df['project_id'] == pid].copy()

        # Get project type
        original_df = pd.read_csv(config['data_path'])
        ptype = original_df[original_df['project_id'] == pid].iloc[0]['project_type_name']

        # Get predictions
        result = predict_future_costs(
            model, project_data, config, scaler,
            device=device, horizon=future_steps
        )

        if result is None or len(result['predictions']) == 0:
            continue

        # Calculate metrics for this project（从 result 里拿 hist_cols 更稳）
        scaler_cols = list(scaler.feature_names_in_)
        target_idx_res = scaler_cols.index('total_cost')

        metrics = calculate_metrics(
            result['predictions'],
            result['actuals'],
            scaler,
            target_idx_res
        )

        project_results.append({
            'project_id': pid,
            'project_type': ptype,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'mape': metrics['mape'],
            'wape': metrics['wape'],
            'n_predictions': len(metrics['predictions_original'])
        })

    return pd.DataFrame(project_results)


# ===========================
# Visualize Predictions
# ===========================
def visualize_predictions(test_df, test_project_ids, model, config, scaler,
                          device='cpu', output_dir='models/predictions/'):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING PREDICTION VISUALIZATIONS")
    print("=" * 60)

    hist_cols, fut_cols = _get_cfg_cols(config)
    sequence_length = config.get('sequence_length', 12)
    future_steps = 3  # Predict 3 months ahead
    scaler_cols = list(scaler.feature_names_in_)
    target_idx = scaler_cols.index('total_cost')

    sample_size = min(6, len(test_project_ids))
    sample_projects = np.random.choice(test_project_ids, sample_size, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    all_predictions, all_actuals = [], []
    all_iscomp = []

    for idx, pid in enumerate(sample_projects):
        project_data = test_df[test_df['project_id'] == pid].copy()
        ptype = project_data.iloc[0]['project_type_name']

        result = predict_future_costs(
            model, project_data, config, scaler,
            device=device, horizon=future_steps
        )

        if result is None:
            continue

        all_predictions.extend(result['predictions'])
        all_actuals.extend(result['actuals'])

        if 'total_duration_months' in project_data.columns:
            total_duration = float(project_data.iloc[0]['total_duration_months'])
        else:
            # 如果列不存在，通过反缩放计算
            month_idx = hist_cols.index('month')
            rem_idx = hist_cols.index('remaining_months')
            sample_scaled = project_data.iloc[[0]][hist_cols].to_numpy()
            sample_raw = scaler.inverse_transform(sample_scaled)
            first_month = float(sample_raw[0, month_idx])
            first_remaining = float(sample_raw[0, rem_idx])
            total_duration = first_month + first_remaining

        scaler_cols = list(scaler.feature_names_in_)
        month_idx_s = scaler_cols.index('month')

        for mseq in result['months']:
            mseq = np.asarray(mseq, dtype=float).reshape(-1, 1)

            # 只反缩放 month 一列（按 scaler 列序）
            dummy = np.zeros((len(mseq), scaler.n_features_in_), dtype=float)
            dummy[:, month_idx_s] = mseq.ravel()
            raw_months = scaler.inverse_transform(dummy)[:, month_idx_s]

            flags = [1.0 if (total_duration - int(round(m))) <= 2 else 0.0 for m in raw_months]
            all_iscomp.extend(flags)


        if len(result['predictions']) > 0:
            example_idx   = len(result['predictions']) // 2
            pred_values   = np.array(result['predictions'][example_idx])
            actual_values = np.array(result['actuals'][example_idx])

            dummy = np.zeros((len(pred_values), scaler.n_features_in_), dtype=float)
            dummy[:, target_idx] = pred_values.ravel()
            pred_original = np.expm1(scaler.inverse_transform(dummy)[:, target_idx])

            dummy[:, target_idx] = actual_values.ravel()
            actual_original = np.expm1(scaler.inverse_transform(dummy)[:, target_idx])

            ax = axes[idx]
            x = np.arange(len(pred_values))
            ax.plot(x, actual_original, marker='o', linewidth=2.2,
                    markersize=6, label='Actual', color='blue')
            ax.plot(x, pred_original, marker='s', linewidth=2.2,
                    markersize=6, label='Predicted', color='red', linestyle='--')


            for i, (a, p) in enumerate(zip(actual_original, pred_original)):
                ax.text(i, a, f'{a/1e6:.2f}M', ha='center', va='bottom', fontsize=8)
                ax.text(i, p, f'{p/1e6:.2f}M', ha='center', va='top',    fontsize=8)

            ax.set_xlabel('Future Month Step')
            ax.set_ylabel('Total Cost (CNY)')
            ax.set_title(f'{pid} - {ptype}\n3-Month Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_predictions.png'), dpi=300, bbox_inches='tight')
    print("Saved: detailed_predictions.png")
    plt.close()


    hist_cols, _ = _get_cfg_cols(config)
    target_idx = hist_cols.index('total_cost')

    metrics = calculate_metrics(
        all_predictions,
        all_actuals,
        scaler,
        target_idx=None,
        is_completion_flags=all_iscomp
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(metrics['actuals_original'], metrics['predictions_original'],
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val = min(metrics['actuals_original'].min(), metrics['predictions_original'].min())
    max_val = max(metrics['actuals_original'].max(), metrics['predictions_original'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Cost (CNY)')
    ax.set_ylabel('Predicted Cost (CNY)')
    ax.set_title(f'Predictions vs Actuals\nR² = {metrics["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain')

    ax = axes[1]
    residuals = metrics['predictions_original'] - metrics['actuals_original']
    ax.scatter(metrics['actuals_original'], residuals,
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual Cost (CNY)')
    ax.set_ylabel('Residual (Predicted - Actual)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_residual.png'), dpi=300, bbox_inches='tight')
    print("Saved: scatter_residual.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    abs_errors = np.abs(residuals)
    ax.hist(abs_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=metrics['mae'], color='r', linestyle='--', linewidth=2,
               label=f'MAE = {metrics["mae"]/1e6:.2f}M')
    ax.set_xlabel('Absolute Error (CNY)')
    ax.set_ylabel('Frequency')
    ax.set_title('Absolute Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.ticklabel_format(style='plain', axis='x')

    ax = axes[1]
    denom = np.clip(metrics['actuals_original'], 1e-6, None)
    pct_errors = (residuals / denom) * 100
    ax.hist(pct_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Percentage Error Distribution\nMAPE = {metrics["mape"]:.2f}%  |  WAPE = {metrics["wape"]:.2f}%')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: error_distribution.png")
    plt.close()

    return metrics


# ===========================
# Enhanced Print Metrics Summary
# ===========================
def print_metrics_summary(metrics, project_metrics=None):
    """
    Print comprehensive metrics summary including WAPE and step-wise metrics

    Args:
        metrics: Dictionary with overall metrics
        project_metrics: DataFrame with per-project metrics (optional)
    """
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)

    # Overall metrics
    print(f"\n  Overall Metrics:")
    print(f"  • MAE (Mean Absolute Error):        {metrics['mae']:>15,.2f}")
    print(f"  • RMSE (Root Mean Squared Error):   {metrics['rmse']:>15,.2f}")
    print(f"  • R² Score:                          {metrics['r2']:>15.4f}")
    print(f"  • MAPE (Mean Abs % Error):          {metrics['mape']:>15.2f}%")
    print(f"  • WAPE (Weighted Abs % Error):      {metrics['wape']:>15.2f}%  ⭐ More Robust")

    # Step-wise metrics
    print(f"\nStep-wise Forecast Metrics:")
    print(f"{'Step':<8} {'MAE':<15} {'RMSE':<15} {'MAPE':<12} {'WAPE':<12} {'Samples':<10}")
    print("-" * 70)
    for step in ['t+1', 't+2', 't+3']:
        if step in metrics['step_metrics']:
            sm = metrics['step_metrics'][step]
            print(f"{step:<8} {sm['mae']:>14,.0f} {sm['rmse']:>14,.0f} "
                  f"{sm['mape']:>10.2f}% {sm['wape']:>10.2f}% {sm['n_samples']:>10}")

    # Per-project metrics
    if project_metrics is not None and len(project_metrics) > 0:
        print(f"\n Per-Project Performance:")
        print(f"{'Project':<10} {'Type':<20} {'MAE':<15} {'WAPE':<12} {'R²':<10}")
        print("-" * 70)

        # Sort by WAPE to identify problematic projects
        sorted_df = project_metrics.sort_values('wape', ascending=False)
        for _, row in sorted_df.iterrows():
            print(f"{row['project_id']:<10} {row['project_type']:<20} "
                  f"{row['mae']:>14,.0f} {row['wape']:>10.2f}% {row['r2']:>9.4f}")

        # Summary by project type
        print(f"\n Performance by Project Type:")
        type_summary = project_metrics.groupby('project_type').agg({
            'mae': 'mean',
            'wape': 'mean',
            'r2': 'mean',
            'project_id': 'count'
        }).round(2)
        type_summary.columns = ['Avg MAE', 'Avg WAPE', 'Avg R²', 'Count']
        print(type_summary.to_string())

    # Interpretation
    print(f"\n Interpretation:")
    if metrics['r2'] > 0.8:
        print(f"Excellent model performance (R² > 0.8)")
    elif metrics['r2'] > 0.6:
        print(f"Good model performance (R² > 0.6)")
    elif metrics['r2'] > 0.4:
        print(f"Moderate model performance (R² > 0.4)")
    else:
        print(f"Poor model performance (R² < 0.4)")

    if metrics['wape'] < 10:
        print(f"Excellent accuracy (WAPE < 10%)")
    elif metrics['wape'] < 20:
        print(f"Good accuracy (WAPE < 20%)")
    else:
        print(f"Moderate accuracy (WAPE > 20%)")

    # Check if error increases with forecast horizon
    if 't+1' in metrics['step_metrics'] and 't+3' in metrics['step_metrics']:
        wape_increase = metrics['step_metrics']['t+3']['wape'] - \
                        metrics['step_metrics']['t+1']['wape']
        if wape_increase > 5:
            print(f"Forecast degradation: WAPE increases {wape_increase:.1f}% "
                  f"from t+1 to t+3")
        else:
            print(f"Stable forecast: WAPE remains consistent across horizons")

    # Completion phase subset
    if metrics.get('completion_phase'):
        sub = metrics['completion_phase']
        print(f"\n  Completion-phase subset (is_completion_phase==1):")
        print(f"  • MAE:  {sub['mae']:,.2f}")
        print(f"  • WAPE: {sub['wape']:.2f}%")
        print(f"  • N:    {sub['n']}")

    print("=" * 70)


# ===========================
# Main Prediction Pipeline
# ===========================
def main():
    """
    Main prediction pipeline
    """

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")

    # Load trained model
    model, config, scaler = load_trained_model(
        model_path=str(MODEL_PATH),
        config_path=str(CONFIG_PATH),
        scaler_path=str(SCALER_PATH),
        device=device
    )

    hist_cols = config.get('feature_cols_hist', config.get('hist_feature_cols', config.get('feature_cols')))

    # Load all projects and split
    df = pd.read_csv(DATA_PATH)

    # Use last 20% of projects as test set
    all_projects = df['project_id'].unique()
    n_train = int(len(all_projects) * 0.8)
    test_project_ids = all_projects[n_train:]
    print(f"\n Test set: {len(test_project_ids)} projects")
    print(f"   Projects: {', '.join(test_project_ids[:5])}{'...' if len(test_project_ids) > 5 else ''}")

    # Prepare test data
    hist_cols, fut_cols = _get_cfg_cols(config)
    test_df = prepare_test_data(
        data_path=str(DATA_PATH),
        test_project_ids=test_project_ids,
        scaler=scaler,
        feature_cols=hist_cols,
        sequence_length=config.get('sequence_length', 12)
    )

    # Generate predictions and visualizations
    metrics = visualize_predictions(
        test_df=test_df,
        test_project_ids=test_project_ids,
        model=model,
        config=config,
        scaler=scaler,
        device=device,
        output_dir=str(OUT_DIR)
    )

    project_metrics = calculate_project_metrics(
        test_df, test_project_ids, model, config, scaler, device
    )
    # Print metrics summary
    print_metrics_summary(metrics, project_metrics)

    # Save metrics to file
    metrics_file = OUT_DIR / 'metrics_enhanced.json'
    metrics_to_save = {
        'overall': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items() if not isinstance(v, np.ndarray)},
        'by_project': project_metrics.to_dict('records') if project_metrics is not None else []
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()