"""
Combine_Ensemble_CS.py - Multi-seed Ensemble Combiner
Combines predictions from multiple folds × multiple seeds into probabilistic forecasts

NEW: Supports fold{k}_seed{s}.csv format for richer uncertainty quantification
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# ------------------------------
# Calibration Metrics (NEW)
# ------------------------------
def calculate_calibration_metrics(p10, p50, p90, actual):
    """
    Calculate probabilistic forecast calibration metrics

    Args:
        p10, p50, p90: Predicted quantiles (arrays)
        actual: Actual values (array)

    Returns:
        dict with calibration metrics
    """
    metrics = {}

    # 1. Coverage / Empirical Coverage
    metrics['p90_coverage'] = np.mean(actual <= p90) * 100
    metrics['p50_coverage'] = np.mean(actual <= p50) * 100
    metrics['p10_coverage'] = np.mean(actual <= p10) * 100

    # 2. Interval Score (Gneiting & Raftery 2007)
    alpha = 0.2  # For 80% interval (p10-p90)
    interval_width = p90 - p10
    lower_penalty = (2 / alpha) * np.maximum(0, p10 - actual)
    upper_penalty = (2 / alpha) * np.maximum(0, actual - p90)
    interval_scores = interval_width + lower_penalty + upper_penalty
    metrics['mean_interval_score'] = np.mean(interval_scores)
    metrics['interval_width_mean'] = np.mean(interval_width)

    # 3. Pinball Loss for P90 (quantile regression loss)
    tau = 0.9
    pinball_90 = np.where(actual >= p90,
                          tau * (actual - p90),
                          (1 - tau) * (p90 - actual))
    metrics['pinball_loss_p90'] = np.mean(pinball_90)

    # 4. Pinball Loss for P50 (median)
    tau = 0.5
    pinball_50 = np.where(actual >= p50,
                          tau * (actual - p50),
                          (1 - tau) * (p50 - actual))
    metrics['pinball_loss_p50'] = np.mean(pinball_50)

    # 5. Sharpness (average interval width)
    metrics['sharpness'] = np.mean(p90 - p10)

    return metrics


def print_calibration_report(metrics):
    """Print calibration metrics in a formatted report"""
    print("\n" + "=" * 70)
    print("PROBABILISTIC FORECAST CALIBRATION METRICS")
    print("=" * 70)

    print("\n1. Coverage (Reliability):")
    print(f"   P90 Coverage: {metrics['p90_coverage']:.1f}% (ideal: 90%)")
    print(f"   P50 Coverage: {metrics['p50_coverage']:.1f}% (ideal: 50%)")
    print(f"   P10 Coverage: {metrics['p10_coverage']:.1f}% (ideal: 10%)")

    # Calibration quality assessment
    p90_error = abs(metrics['p90_coverage'] - 90)
    p50_error = abs(metrics['p50_coverage'] - 50)

    if p90_error < 5 and p50_error < 5:
        cal_quality = "Excellent"
    elif p90_error < 10 and p50_error < 10:
        cal_quality = "Good"
    elif p90_error < 15 and p50_error < 15:
        cal_quality = "Acceptable"
    else:
        cal_quality = "Poor (needs recalibration)"

    print(f"   Calibration Quality: {cal_quality}")

    print("\n2. Sharpness (Precision):")
    print(f"   Mean Interval Width (P10-P90): {metrics['interval_width_mean']:.2f} pp")
    print(f"   Overall Sharpness: {metrics['sharpness']:.2f} pp")

    print("\n3. Interval Score (Lower is better):")
    print(f"   Mean Interval Score: {metrics['mean_interval_score']:.2f}")

    print("\n4. Pinball Loss (Lower is better):")
    print(f"   P90 Pinball Loss: {metrics['pinball_loss_p90']:.2f}")
    print(f"   P50 Pinball Loss: {metrics['pinball_loss_p50']:.2f}")

    print("=" * 70 + "\n")


# ------------------------------
# Helpers
# ------------------------------
def _read_fold_csv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if not {"month", "p50"}.issubset(df.columns):
            print(f"{path.name} missing required columns month/p50, skipping.")
            return None
        df = df[["month", "p50"]].copy()
        df = df.drop_duplicates(subset=["month"])
        df = df.sort_values("month")
        df["p50"] = pd.to_numeric(df["p50"], errors="coerce")
        return df
    except Exception as e:
        print(f"Unable to read {path.name}: {e}")
        return None


def _enforce_monotone_0_100(arr: np.ndarray) -> np.ndarray:
    """Enforce monotonic non-decreasing + [0,100] clipping."""
    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=0.0)
    for i in range(1, arr.size):
        if arr[i] < arr[i - 1]:
            arr[i] = arr[i - 1]
    arr = np.clip(arr, 0.0, 100.0)
    return arr


# ------------------------------
# Main combiner (Multi-seed version)
# ------------------------------
def combine_multiseed(case_dir: Path,
                      n_folds=5,
                      n_seeds=5,
                      save_plot: bool = True,
                      actual_file: Path = None) -> Path:
    """
    Combine multi-seed fold predictions

    Args:
        case_dir: Directory containing pred_progress_fold{k}_seed{s}.csv files
        n_folds: Number of folds
        n_seeds: Number of seeds per fold
        save_plot: Whether to generate plot
        actual_file: Optional path to actual data for calibration metrics
    """
    case_dir = Path(case_dir)
    assert case_dir.exists(), f"Case dir not found: {case_dir}"

    print("\n" + "=" * 70)
    print("MULTI-SEED ENSEMBLE COMBINATION")
    print("=" * 70)
    print(f"  • Number of folds: {n_folds}")
    print(f"  • Seeds per fold: {n_seeds}")
    print(f"  • Total trajectories: {n_folds * n_seeds}")
    print("=" * 70)

    # Read all fold×seed combinations
    trajectories = []
    month_union = set()
    missing_files = []

    for fold in range(n_folds):
        for seed in range(n_seeds):
            fpath = case_dir / f"pred_progress_fold{fold}_seed{seed}.csv"

            if not fpath.exists():
                missing_files.append(f"fold{fold}_seed{seed}")
                continue

            df = _read_fold_csv(fpath)
            if df is None or df.empty:
                missing_files.append(f"fold{fold}_seed{seed}")
                continue

            df["fold"] = fold
            df["seed"] = seed
            trajectories.append(df)
            month_union.update(df["month"].unique().tolist())

    if missing_files:
        print(f"\nWarning: Missing {len(missing_files)} files:")
        print(f"  {', '.join(missing_files[:10])}" +
              (f" ... and {len(missing_files) - 10} more" if len(missing_files) > 10 else ""))

    if not trajectories:
        raise RuntimeError("No valid prediction files found!")

    month_list = sorted(month_union)
    print(f"\nRead {len(trajectories)} trajectories")
    print(f"Covering months {month_list[0]}-{month_list[-1]} ({len(month_list)} months)")

    # Assemble into wide format: index=month, columns=fold{k}_seed{s}
    wide = pd.DataFrame(index=month_list)
    for df in trajectories:
        fold = int(df["fold"].iloc[0])
        seed = int(df["seed"].iloc[0])
        wide[f"fold{fold}_seed{seed}"] = df.set_index("month")["p50"].reindex(month_list)

    values = wide.values  # shape (M, N_trajectories)

    print(f"\nTrajectory statistics:")
    print(f"  • Total trajectories: {values.shape[1]}")
    print(f"  • Expected: {n_folds * n_seeds}")
    print(f"  • Coverage: {values.shape[1] / (n_folds * n_seeds) * 100:.1f}%")

    # Calculate quantiles
    p10 = np.nanpercentile(values, 10, axis=1, method="linear")
    p50 = np.nanpercentile(values, 50, axis=1, method="linear")
    p90 = np.nanpercentile(values, 90, axis=1, method="linear")

    # Post-processing
    p10 = _enforce_monotone_0_100(p10)
    p50 = _enforce_monotone_0_100(p50)
    p90 = _enforce_monotone_0_100(p90)

    # Check for P50/P90 separation
    separation = np.mean(p90 - p50)
    print(f"\nUncertainty band analysis:")
    print(f"  • Mean P50-P90 separation: {separation:.2f} pp")

    if separation < 1.0:
        print(f"Warning: Very narrow uncertainty bands!")
        print(f"Consider increasing n_seeds or checking model diversity")
    elif separation > 10.0:
        print(f"Warning: Very wide uncertainty bands!")
        print(f"Check for outlier models or excessive variance")
    else:
        print(f"Reasonable uncertainty quantification")

    # Assemble output
    out = pd.DataFrame({
        "month": month_list,
        "p10": p10,
        "p50": p50,
        "p90": p90,
    })

    out_path = case_dir / "pred_progress.csv"
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n✓ Combined predictions saved: {out_path}")

    # Calculate calibration metrics if actual data provided
    if actual_file and Path(actual_file).exists():
        try:
            actual_df = pd.read_csv(actual_file)
            actual_df.columns = [c.strip().replace('\ufeff', '') for c in actual_df.columns]

            # --- build 'month' column ---
            if 'month' not in actual_df.columns:
                if 'month_index' in actual_df.columns:
                    actual_df['month'] = actual_df['month_index']
                else:
                    raise ValueError(f"Actual CSV missing month/month_index columns: {list(actual_df.columns)}")

            # --- build 'actual' column ---
            if 'actual' not in actual_df.columns:
                if 'cumulative_share_pct' in actual_df.columns:
                    actual_df['actual'] = actual_df['cumulative_share_pct']
                else:
                    raise ValueError(
                        f"Actual CSV missing actual/cumulative_share_pct columns: {list(actual_df.columns)}")

            # --- drop NaN summary rows (same idea as Compare_Prediction does) ---
            actual_df = actual_df[['month', 'actual']].dropna(subset=['month', 'actual']).copy()
            actual_df['month'] = actual_df['month'].astype(int)
            actual_df['actual'] = actual_df['actual'].astype(float)

            # --- IMPORTANT: align month index with Compare_Prediction logic ---
            # Compare_Prediction shifts pred month back by 1 to match month-end actuals :contentReference[oaicite:7]{index=7}
            out_for_cal = out[['month', 'p10', 'p50', 'p90']].copy()
            out_for_cal['month_eom'] = out_for_cal['month'].astype(int) - 1

            merged = out_for_cal.merge(
                actual_df,
                left_on='month_eom',
                right_on='month',
                how='inner',
                suffixes=('', '_act')
            )

            if len(merged) > 0:
                metrics = calculate_calibration_metrics(
                    merged['p10'].values,
                    merged['p50'].values,
                    merged['p90'].values,
                    merged['actual'].values
                )
                print_calibration_report(metrics)

                metrics_path = case_dir / "calibration_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"✓ Calibration metrics saved: {metrics_path}")
            else:
                print("\nWarning: No overlapping months between predictions and actuals (after month alignment)")

        except Exception as e:
            print(f"\nWarning: Could not calculate calibration metrics: {e}")

    # Plotting
    if save_plot:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Top panel: Predictions with uncertainty
            ax1.plot(out["month"], out["p50"], label="P50 (Median)",
                     linewidth=2, color='blue')
            ax1.fill_between(out["month"], out["p10"], out["p90"],
                             alpha=0.3, label="P10-P90", color='blue')

            # Add actual if available
            if actual_file and Path(actual_file).exists():
                actual_df = pd.read_csv(actual_file)
                actual_df.columns = [c.strip().replace('\ufeff', '') for c in actual_df.columns]

                if 'month' not in actual_df.columns and 'month_index' in actual_df.columns:
                    actual_df['month'] = actual_df['month_index']
                if 'actual' not in actual_df.columns and 'cumulative_share_pct' in actual_df.columns:
                    actual_df['actual'] = actual_df['cumulative_share_pct']

                actual_df = actual_df[['month', 'actual']].dropna(subset=['month', 'actual']).copy()
                actual_df['month'] = actual_df['month'].astype(int)
                actual_df['actual'] = actual_df['actual'].astype(float)

                # to overlay with pred months (pred is month-start), shift actual forward by +1
                ax1.plot(actual_df["month"] + 1, actual_df["actual"],
                         label="Actual", linewidth=2, marker='o', markersize=4)


            ax1.set_xlabel("Month")
            ax1.set_ylabel("Cumulative Progress (%)")
            ax1.set_ylim(0, 105)
            ax1.grid(alpha=0.3, linestyle="--")
            ax1.set_title(f"Ensemble Prediction ({n_folds} folds × {n_seeds} seeds = {n_folds * n_seeds} trajectories)")
            ax1.legend()

            # Bottom panel: Uncertainty evolution
            uncertainty = out["p90"] - out["p10"]
            ax2.plot(out["month"], uncertainty, linewidth=2, color='orange')
            ax2.fill_between(out["month"], 0, uncertainty, alpha=0.3, color='orange')
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Uncertainty Width (pp)")
            ax2.grid(alpha=0.3, linestyle="--")
            ax2.set_title("Prediction Uncertainty (P90 - P10)")

            fig.tight_layout()
            fig_path = case_dir / "pred_progress_plot.png"
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"✓ Visualization saved: {fig_path}")
        except Exception as e:
            print(f"Warning: Plotting failed: {e}")

    return out_path


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Combine multi-seed CV predictions")
    ap.add_argument("--case_dir", type=str, required=True,
                    help="Directory containing pred_progress_fold{k}_seed{s}.csv files")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Number of folds (default: 5)")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Number of seeds per fold (default: 5)")
    ap.add_argument("--actual_file", type=str, default=None,
                    help="CSV file with actual values for calibration (must have 'month' and 'actual' columns)")
    ap.add_argument("--no_plot", action="store_true",
                    help="Do not generate plots")

    return ap.parse_args()


def main():
    args = parse_args()

    combine_multiseed(
        case_dir=Path(args.case_dir),
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
        save_plot=not args.no_plot,
        actual_file=Path(args.actual_file) if args.actual_file else None
    )


if __name__ == "__main__":
    main()