"""
NEW FEATURES:
- Loop through fold{k}_seed{s} combinations
- Generate pred_progress_fold{k}_seed{s}.csv for each model
- Compatible with Train_CV_MultiSeed.py output
- Automatic model discovery from cv_model_dir

Based on: Prediction_CS.py
Author: Enhanced for multi-seed ensemble
"""

import argparse
import json
import pickle
from pathlib import Path
import torch
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from LSTM_Model import LSTMSeq2SeqMass
from Train_CV import add_enhanced_features
from Prediction import predict_future_costs


# ============================================================
# Feature synthesis functions
# ============================================================

def _standardize_case_df(df):
    """Standardize column names"""
    rename_map = {
        'cumulative_cost_pct': 'progress_pct',
        'month_index': 'month',
        'equipment_cost': 'equip_cost',
    }
    df = df.rename(columns=rename_map)
    return df


def _try_fill_time_features(df, feature_cols):
    """Fill time-related features if missing"""
    if "normalized_time" in feature_cols and "normalized_time" not in df.columns:
        if "month" in df.columns:
            T_max = df["month"].max()
            df["normalized_time"] = df["month"].astype(float) / max(T_max, 1.0)
        else:
            df["normalized_time"] = 0.0

    if "remaining_months" in feature_cols:
        need_rem = "remaining_months" not in df.columns
        if need_rem and "month" in df.columns:
            T = df["month"].max() if "month" in df.columns else 24
        else:
            T = 24
        if need_rem:
            df["remaining_months"] = np.maximum(T - df["month"].astype(float), 0.0)

    if "is_completion_phase" in feature_cols and "is_completion_phase" not in df.columns:
        if "remaining_months" in df.columns:
            df["is_completion_phase"] = (df["remaining_months"] <= 2).astype(float)
        else:
            df["is_completion_phase"] = 0.0

    return df


def _ensure_columns(df, feature_cols):
    """Ensure input CSV contains all required feature columns"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV missing required feature columns: {missing}\n"
            f"Expected feature columns: {feature_cols}"
        )
    return df[feature_cols].copy()


def _synthesize_missing_features(
        raw_df: pd.DataFrame,
        total_budget: float,
        industry_config_path: str | None = None,
        required_hist_cols=None,
        required_fut_cols=None,
):
    df = raw_df.copy()

    # 1) Progress percentage
    if 'progress_pct' not in df.columns:
        if 'cumulative_cost_pct' in df.columns:
            df['progress_pct'] = df['cumulative_cost_pct'].astype(float)
        else:
            raise ValueError("需要 progress_pct 或 cumulative_cost_pct 列")

    prog = pd.to_numeric(df['progress_pct'], errors='coerce').fillna(0.0).astype(float)
    if prog.max() <= 1.2:
        prog = prog * 100.0
    prog = prog.clip(lower=0.0, upper=100.0)
    df['progress_pct'] = prog

    # 2) Month column
    if 'month' not in df.columns:
        if 'month_index' in df.columns:
            df['month'] = df['month_index'].astype(int)
        else:
            df['month'] = np.arange(1, len(df) + 1, dtype=int)

    # 3) total_cost (monthly cost)
    if 'total_cost' not in df.columns:
        if 'cumulative_cost_pct' in df.columns:
            cum_pct = df['cumulative_cost_pct'].values
        else:
            cum_pct = prog.values

        prev = np.r_[0.0, cum_pct[:-1]]
        incr_pct = (cum_pct - prev).clip(min=0.0) / 100.0
        monthly_cost = total_budget * incr_pct
        df['total_cost'] = monthly_cost

        print(f"   Generated total_cost from progress:")
        print(f"   Range: [{monthly_cost.min():,.0f}, {monthly_cost.max():,.0f}] CNY")
        print(f"   Sum: {monthly_cost.sum():,.0f} CNY (budget: {total_budget:,.0f})")

    # 4) Cost breakdown (material/labour/equip/admin)
    ratios = {'material': 0.55, 'labour': 0.30, 'equip': 0.10, 'admin': 0.05}

    if 'material_ratio' in df.columns:
        for key, col in [('material', 'material_ratio'), ('labour', 'labour_ratio'),
                         ('equip', 'equipment_ratio'), ('admin', 'admin_ratio')]:
            if col in df.columns:
                if f'{key}_cost' not in df.columns:
                    df[f'{key}_cost'] = df['total_cost'] * df[col].astype(float)
            else:
                if f'{key}_cost' not in df.columns:
                    df[f'{key}_cost'] = df['total_cost'] * ratios[key]
    else:
        for key, ratio in ratios.items():
            if f'{key}_cost' not in df.columns:
                df[f'{key}_cost'] = df['total_cost'] * ratio

    # 5) External indices (default to 1.0)
    for idx_col in ['mat_index', 'lab_index', 'cpi_index']:
        if idx_col not in df.columns:
            df[idx_col] = 1.0

    return df


def load_case_and_scale(input_csv, union_feature_cols, scaler, total_budget, industry_config_path):
    """Load and scale case study data"""
    raw = pd.read_csv(input_csv)
    print(f"   Raw data: {len(raw)} rows × {len(raw.columns)} columns")

    # 1) Standardize column names
    raw = _standardize_case_df(raw)

    # 2) Fill time features
    raw = _try_fill_time_features(raw, union_feature_cols)

    # 3) Enhanced features
    raw = add_enhanced_features(raw)

    # 4) Synthesize missing features
    raw = _synthesize_missing_features(
        raw_df=raw,
        total_budget=total_budget,
        industry_config_path=industry_config_path,
        required_hist_cols=union_feature_cols,
        required_fut_cols=[]
    )

    # 5) Log1p + standardization
    # Remove non-numeric columns before conversion
    non_numeric_cols = raw.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"   Dropping non-numeric columns: {non_numeric_cols}")
        raw = raw.drop(columns=non_numeric_cols)

    use = raw.astype(float).copy()

    # Check if log1p is needed
    if 'total_cost' in use.columns:
        tc_max = use['total_cost'].max()
        tc_min = use['total_cost'].min()

        print(f"\n   total_cost before log1p:")
        print(f"     Range: [{tc_min:,.2f}, {tc_max:,.2f}] CNY")

        if tc_max > 20:
            use['total_cost'] = np.log1p(use['total_cost'])
            print(f"   Applied log1p to total_cost")
            print(f"   After log1p: [{use['total_cost'].min():.4f}, {use['total_cost'].max():.4f}]")
        else:
            print(f"   Skipped log1p (values already small)")

    # Apply log1p to other cost columns
    for cost_col in ['material_cost', 'labour_cost', 'equip_cost', 'admin_cost']:
        if cost_col in use.columns and use[cost_col].max() > 20:
            use[cost_col] = np.log1p(use[cost_col])

    # 6) Standardize
    print(f"\n   [STANDARDIZING] Applying StandardScaler...")
    # Select only training features before scaling
    use_selected = use[union_feature_cols]
    print(f"   Selected {len(union_feature_cols)} training features from {len(use.columns)} total features")
    X = scaler.transform(use_selected.values.astype(float))
    df_scaled = pd.DataFrame(X, columns=union_feature_cols)

    # Validate
    if 'total_cost' in union_feature_cols:
        tc_idx = union_feature_cols.index('total_cost')
        tc_scaled = X[:, tc_idx]
        print(f"      total_cost (scaled) range: [{tc_scaled.min():.4f}, {tc_scaled.max():.4f}]")
        print(f"      total_cost (scaled) mean: {tc_scaled.mean():.4f}, std: {tc_scaled.std():.4f}")

        if tc_scaled.max() > 5 or tc_scaled.min() < -5:
            print(f"      WARNING: Scaled values outside typical range!")

    return raw, df_scaled


def reconstruct_monthly_cost(result, scaler, union_feature_cols, raw_df, scaled_df, sequence_length):
    """Reconstruct monthly cost series"""
    if "total_cost" not in union_feature_cols:
        raise ValueError("union_feature_cols must include 'total_cost'")
    idx_cost = union_feature_cols.index("total_cost")

    month_preds = {}
    n_win = len(result['predictions'])

    # Generate window_starts if not provided (sliding window assumption)
    if 'window_starts' not in result:
        window_starts = list(range(n_win))
    else:
        window_starts = result['window_starts']

    for w_idx in range(n_win):
        pred_seq = result['predictions'][w_idx]
        if not pred_seq:
            continue

        for h in range(len(pred_seq)):
            pred_val = pred_seq[h]
            if not np.isfinite(pred_val):
                continue

            start_idx = window_starts[w_idx]
            input_end_idx = start_idx + sequence_length - 1
            target_month = input_end_idx + 1 + h

            if target_month >= len(raw_df):
                continue

            # Use actual features as baseline
            base_row = scaled_df.iloc[target_month].values.copy().astype(float)
            base_row[idx_cost] = pred_val

            # Inverse transform
            try:
                restored = scaler.inverse_transform(base_row.reshape(1, -1))
                cost_restored = restored[0, idx_cost]

                # Expm1 if was log1p
                if cost_restored > 0:
                    cost_restored = np.expm1(cost_restored)

                cost_restored = max(cost_restored, 0.0)

                if target_month not in month_preds:
                    month_preds[target_month] = []
                month_preds[target_month].append(cost_restored)
            except Exception as e:
                continue

    # Average predictions
    monthly_cost = pd.Series({
        m: np.median(preds) for m, preds in month_preds.items()
    }, dtype=float)

    return monthly_cost


# ============================================================
# Multi-seed prediction function
# ============================================================

def predict_single_model(
        fold: int,
        seed: int,
        cv_model_dir: Path,
        input_csv: Path,
        output_dir: Path,
        total_budget: float,
        industry_config_path: Path,
        anchor_mode: str = "default",
        anchor_csv: Path = None,
        device: str = 'cpu'
):
    """
    Predict using a single fold×seed model

    Args:
        fold: Fold index (0-4)
        seed: Seed index (0-4)
        cv_model_dir: Directory containing trained models
        input_csv: Input case study CSV
        output_dir: Output directory for predictions
        total_budget: Total project budget
        industry_config_path: Industry config file
        anchor_mode: Anchoring mode
        anchor_csv: CSV with anchoring data
        device: CPU or CUDA

    Returns:
        Path to output CSV or None if failed
    """

    print(f"\n{'=' * 70}")
    print(f"PREDICTING: Fold {fold} | Seed {seed}")
    print(f"{'=' * 70}")

    try:
        # Load config
        config_path = cv_model_dir / f"model_config_fold{fold}_seed{seed}.json"
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return None

        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Load scaler (shared across seeds in same fold)
        scaler_path = cv_model_dir / f"scaler_fold{fold}.pkl"
        if not scaler_path.exists():
            print(f"Scaler not found: {scaler_path}")
            return None

        scaler = joblib.load(scaler_path)

        # Extract parameters
        sequence_length = cfg.get("sequence_length", 12)
        horizon = cfg.get("prediction_horizon", 3)

        hist_feature_cols = cfg.get("hist_feature_cols", [])
        fut_feature_cols = cfg.get("fut_feature_cols", [])

        if hasattr(scaler, "feature_names_in_"):
            union_feature_cols = list(scaler.feature_names_in_)
        else:
            union_feature_cols = list(hist_feature_cols)
            scaler.feature_names_in_ = np.array(union_feature_cols)

        print(f"  Features: hist={len(hist_feature_cols)}, fut={len(fut_feature_cols)}")
        print(f"  Sequence length: {sequence_length}, Horizon: {horizon}")

        # Load model
        model_path = cv_model_dir / f"best_model_fold{fold}_seed{seed}.pt"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None

        model = LSTMSeq2SeqMass(
            in_hist=len(hist_feature_cols),
            in_fut=len(fut_feature_cols),
            hidden_size=int(cfg.get("hidden_size", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
            horizon=horizon
        ).to(device)

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"  ✓ Model loaded from {model_path.name}")

        # Prepare data
        raw_df, scaled_df = load_case_and_scale(
            input_csv, union_feature_cols, scaler, total_budget, industry_config_path
        )

        # Predict
        result = predict_future_costs(
            model=model,
            project_df_scaled=scaled_df,
            config={
                "hist_feature_cols": hist_feature_cols,
                "fut_feature_cols": fut_feature_cols,
                "sequence_length": sequence_length,
                "horizon": horizon
            },
            scaler=scaler,
            device=device,
            horizon=horizon
        )

        if result is None or len(result.get("predictions", [])) == 0:
            print(f"No predictions generated")
            return None

        print(f"Generated {len(result['predictions'])} window predictions")

        # Reconstruct monthly cost
        monthly_cost = reconstruct_monthly_cost(
            result, scaler, union_feature_cols, raw_df, scaled_df, sequence_length
        )

        if len(monthly_cost) == 0:
            print(f"Unable to reconstruct monthly series")
            return None

        print(
            f"Monthly series: {monthly_cost.index.min()} - {monthly_cost.index.max()} ({len(monthly_cost)} months)")

        # Budget alignment
        months_sorted = np.array(sorted(monthly_cost.index))
        monthly_cost_pred = monthly_cost.loc[months_sorted].astype(float).clip(lower=0.0).values

        # Anchor
        if anchor_mode == "csv" and anchor_csv:
            try:
                df_act = pd.read_csv(anchor_csv)
                if {"month_index", "cumulative_share_pct"}.issubset(df_act.columns):
                    m12 = df_act.loc[df_act["month_index"] == 12, "cumulative_share_pct"]
                    base_share = float(m12.iloc[0]) / 100.0 if not m12.empty else 0.50
                else:
                    base_share = 0.50
            except Exception:
                base_share = 0.50
        elif anchor_mode == "none":
            base_share = 0.0
        else:
            base_share = 0.50

        remain_budget = max(0.0, (1.0 - base_share)) * total_budget

        # Check for zero predictions
        all_zero_or_nan = (
                (not np.isfinite(monthly_cost_pred).any()) or
                np.isclose(monthly_cost_pred, 0.0).all() or
                float(np.nansum(monthly_cost_pred)) <= 1e-12
        )

        if all_zero_or_nan:
            print(f"Using momentum fallback...")
            weights = np.arange(1, len(months_sorted) + 1, dtype=float)
            monthly_cost_pred = weights / weights.sum() * remain_budget
        else:
            s = float(np.sum(monthly_cost_pred))
            monthly_cost_pred = monthly_cost_pred * (remain_budget / s)

        # Calculate cumulative
        cumulative_cost_pred = base_share * total_budget + np.cumsum(monthly_cost_pred)
        cumulative_share_pred = np.clip(cumulative_cost_pred / max(total_budget, 1e-6), 0.0, 1.0)

        # Save output
        out = pd.DataFrame({
            "month": months_sorted,
            "p50": cumulative_share_pred * 100.0
        })

        output_path = output_dir / f"pred_progress_fold{fold}_seed{seed}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Saved: {output_path.name}")
        print(f"Progress: {out['p50'].iloc[0]:.2f}% → {out['p50'].iloc[-1]:.2f}%")

        return output_path

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Case Study Prediction")

    # Multi-seed specific
    parser.add_argument("--cv_model_dir", type=str, required=True,
                        help="CV model directory containing fold×seed models")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds per fold (default: 5)")

    # Input/output
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input case study CSV (e.g., chengbei_24m.csv)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all predictions")

    # Project parameters
    parser.add_argument("--total_budget", type=float, default=1e8,
                        help="Total project budget (CNY)")
    parser.add_argument("--industry_config", type=str, default="",
                        help="Industry config file (optional)")

    # Anchoring
    parser.add_argument("--anchor_mode", type=str, default="default",
                        choices=["default", "csv", "none"],
                        help="Anchoring mode")
    parser.add_argument("--anchor_csv", type=str, default=None,
                        help="CSV with actual data for anchoring")

    # Device
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Convert paths
    cv_model_dir = Path(args.cv_model_dir)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    industry_config = Path(args.industry_config) if args.industry_config else None
    anchor_csv = Path(args.anchor_csv) if args.anchor_csv else None

    # Validate
    if not cv_model_dir.exists():
        raise FileNotFoundError(f"CV model directory not found: {cv_model_dir}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SEED CASE STUDY PREDICTION")
    print("=" * 70)
    print(f"CV Model Dir:    {cv_model_dir}")
    print(f"Input CSV:       {input_csv}")
    print(f"Output Dir:      {output_dir}")
    print(f"Total Budget:    {args.total_budget:,.0f} CNY")
    print(f"N Folds:         {args.n_folds}")
    print(f"N Seeds:         {args.n_seeds}")
    print(f"Total Models:    {args.n_folds * args.n_seeds}")
    print(f"Device:          {args.device}")
    print("=" * 70)

    # Predict all models
    successful = 0
    failed = 0

    for fold in range(args.n_folds):
        for seed in range(args.n_seeds):
            result = predict_single_model(
                fold=fold,
                seed=seed,
                cv_model_dir=cv_model_dir,
                input_csv=input_csv,
                output_dir=output_dir,
                total_budget=args.total_budget,
                industry_config_path=industry_config,
                anchor_mode=args.anchor_mode,
                anchor_csv=anchor_csv,
                device=args.device
            )

            if result:
                successful += 1
            else:
                failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total models:      {args.n_folds * args.n_seeds}")
    print(f"Successful:        {successful}")
    print(f"Failed:            {failed}")
    print(f"Success rate:      {successful / (args.n_folds * args.n_seeds) * 100:.1f}%")
    print(f"\nOutput files in:   {output_dir}")
    print("=" * 70)

    if successful == 0:
        print("\nNo predictions generated! Check error messages above.")
        return 1
    elif failed > 0:
        print(f"\n{failed} predictions failed. Continuing with {successful} successful predictions.")
    else:
        print(f"\nAll predictions generated successfully!")

    # List output files
    output_files = sorted(output_dir.glob("pred_progress_fold*_seed*.csv"))
    print(f"\nGenerated {len(output_files)} prediction files:")
    for f in output_files[:10]:  # Show first 10
        print(f"  - {f.name}")
    if len(output_files) > 10:
        print(f"  ... and {len(output_files) - 10} more")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run Combine_Ensemble_CS.py to merge predictions")
    print("=" * 70)
    print(f"\nCommand:")
    print(f"python Combine_Ensemble_CS.py \\")
    print(f"    --case_dir {output_dir} \\")
    print(f"    --n_folds {args.n_folds} \\")
    print(f"    --n_seeds {args.n_seeds} \\")
    print(f"    --actual_file <path_to_actual_data.csv>")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())