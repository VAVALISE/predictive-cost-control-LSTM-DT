"""
Evaluation script for comparing predicted progress with actual progress

Input:
- pred_progress.csv: Predicted progress (month, p50, p90)
- Preview_progress.csv: Actual progress (DATE/month, Monthly_%_Weighted6/Cumulative_%_Weighted6, etc.)

Output:
- Comparison charts
- Evaluation metrics (MAE, RMSE, MAPE, etc.)
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not installed, will use basic matplotlib style")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not installed, will use numpy implementation for metrics")

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def extract_month_number(value):
    """
    Extract month number from various formats: M01/M1/1 -> 1

    Args:
        value: Month value (could be string or number)

    Returns:
        int: Month number
    """
    if pd.isna(value):
        return None

    # If already a number
    if isinstance(value, (int, float)):
        return int(value)

    # If string, extract number
    s = str(value).strip().upper()
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())

    return None


def normalize_month_column(df, month_col='month'):
    """
    Unify month format to integer, automatically recognize month/index/name columns
    """
    if month_col not in df.columns:
        candidates = ['month', 'Month', 'month_index', 'Month_Index', 'date', 'DATE', 'period', 'time', 'month_name']
        for col in candidates:
            if col in df.columns:
                month_col = col
                break
        else:
            raise ValueError(f"Month column not found, tried: {month_col}, {candidates}")

    # Extract month number (for 'Sep-25' format, you can modify as needed)
    def _to_month(v):
        if pd.isna(v): return None
        s = str(v).strip()
        # Prioritize pure numbers
        if s.isdigit(): return int(float(s))
        # Try extracting numbers
        import re
        m = re.search(r'\d+', s)
        if m: return int(m.group())
        return None

    df['month'] = df[month_col].apply(_to_month)
    df = df.dropna(subset=['month'])
    df['month'] = df['month'].astype(int)
    return df


def identify_progress_column(df, user_specified=None):
    """
    Automatically identify actual progress column name (supports more naming conventions)

    Args:
        df: DataFrame
        user_specified: User-specified column name (highest priority)

    Returns:
        str: Progress column name
    """
    # If user specified column name
    if user_specified:
        if user_specified in df.columns:
            return user_specified
        else:
            raise ValueError(f"Specified column '{user_specified}' does not exist in CSV")

    # Extended candidate column names (in priority order)
    candidates = [
        # Cumulative progress related
        'cumulative_%_weighted6',
        'cumulative_weighted6',
        'cum_pct',
        'cumulative',
        'cumulative_progress',
        # Regular progress related
        'progress_pct',
        'progress',
        'actual',
        'p50',
        # Monthly progress (if no cumulative, can use monthly)
        'monthly_%_weighted6',
        'monthly_weighted6',
        'weighted6',
        'monthly_progress',
    ]

    # Convert to lowercase matching
    df_lower = {col.lower().strip(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in df_lower:
            actual_col = df_lower[candidate.lower()]
            print(f"Identified progress column: '{actual_col}'")
            return actual_col

    # If none found, list all columns for user selection
    raise ValueError(
        f"Unable to automatically identify progress column, please use --actual_col to specify.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_and_validate(predicted_csv, actual_csv, actual_col=None, align_mode="none"):
    """
    Load and validate predicted and actual progress data

    Args:
        predicted_csv: Prediction CSV path
        actual_csv: Actual CSV path
        actual_col: Actual progress column name (optional)

    Returns:
        DataFrame: Aligned data
    """
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    # ========== 1. Load Prediction Data ==========
    pred = pd.read_csv(predicted_csv)
    pred.columns = [c.strip() for c in pred.columns]

    print(f"Prediction data original columns: {list(pred.columns)}")

    # Unify month format
    pred = normalize_month_column(pred, 'month')

    if 'p50' not in pred.columns:
        raise ValueError(f"Prediction CSV must contain 'p50' column")

    print(f"Prediction data: {len(pred)} months (months {pred['month'].min()}-{pred['month'].max()})")

    # ========== 2. Load Actual Data ==========
    actual = pd.read_csv(actual_csv)
    actual.columns = [c.strip() for c in actual.columns]
    print(f"Actual data original columns: {list(actual.columns)}")

    # Month standardization
    actual = normalize_month_column(actual)

    # Actual side: prioritize cost caliber
    actual_lower = {c.lower(): c for c in actual.columns}
    use_cost = False
    if 'cumulative_cost' in actual_lower:
        ac = actual[actual_lower['cumulative_cost']].astype(float)
        total = float(ac.iloc[-1] if len(ac) else 0.0) or 1e-6
        actual['actual'] = ac / total * 100.0
        use_cost = True
        print("Actual progress using: cumulative_cost → cost caliber cumulative percentage")
    elif 'monthly_cost' in actual_lower:
        mc = actual[actual_lower['monthly_cost']].astype(float)
        ac = mc.cumsum()
        total = float(ac.iloc[-1] if len(ac) else 0.0) or 1e-6
        actual['actual'] = ac / total * 100.0
        use_cost = True
        print("Actual progress using: monthly_cost.cumsum() → cost caliber cumulative percentage")

    if not use_cost:
        # Fallback: component caliber/other cumulative columns
        progress_col = identify_progress_column(actual, actual_col)
        actual = actual.rename(columns={progress_col: 'actual'})
        print(f"Cost caliber columns missing, fallback to: '{progress_col}'")

    print(f"Actual data: {len(actual)} months (months {actual['month'].min()}-{actual['month'].max()})")

    # ========== 3. Align Months ==========
    # CRITICAL FIX: Prediction values represent month-start (cumulative at beginning of month)
    # Actual values represent month-end (cumulative at end of month)
    #
    # Example: pred[month=13] = progress at start of month 13 (= end of month 12)
    #          actual[month=13] = progress at end of month 13
    #
    # To align properly:
    # pred[month=13] should compare with actual[month=12] (both are end-of-month-12)
    # So we shift pred month back by 1

    # Only keep needed columns
    pred_clean = pred[['month', 'p50']].copy()
    actual_clean = actual[['month', 'actual']].copy()

    # Add P90 if exists
    if 'p90' in pred.columns:
        pred_clean['p90'] = pred['p90']

    # FIX: Prediction_CS 输出的 month 更像是 0-based index（12≈第13月）
    pred_clean['month'] = pred_clean['month'] + 1

    print("Applied month index fix: prediction month shifted forward by +1 (0-based → 1-based)")
    print(f"  Rationale: pred values are month-start, actual values are month-end")
    print(f"  Example: pred[month=13-start] → compare with actual[month=12-end]")

    # Merge (inner join ensures only common months retained)
    merged = pred_clean.merge(actual_clean, on='month', how='inner')

    if len(merged) == 0:
        raise ValueError(
            f"Prediction and actual data have no common months!\n"
            f"Prediction months: {sorted(pred['month'].unique())}\n"
            f"Actual months: {sorted(actual['month'].unique())}"
        )

    # ========== 4. Sort ==========
    merged = merged.sort_values('month').reset_index(drop=True)

    # ========== 5/6. Optional: Evaluation stage second anchoring (default none, no alignment) ==========
    if align_mode == "offset":
        # — Consistent with your original logic: shift prediction curve to previous month's actual value
        m_start = int(merged['month'].min())
        if (m_start - 1) in actual['month'].values:
            baseline_prev = float(actual.loc[actual['month'] == (m_start - 1), 'actual'].iloc[0])
        else:
            baseline_prev = 0.0

        p50_native = merged['p50'].astype(float).values
        p50_anchored = (p50_native - p50_native[0]) + baseline_prev
        merged['p50_anchored'] = np.clip(p50_anchored, 0, 100)

        if 'p90' in merged.columns:
            p90_native = merged['p90'].astype(float).values
            p90_anchored = (p90_native - p90_native[0]) + baseline_prev
            p90_anchored = np.clip(p90_anchored, 0, 100)
            # Ensure P90 ≥ P50
            merged['p90_anchored'] = np.maximum(p90_anchored, merged['p50_anchored'].values)

        merged['baseline_prev'] = baseline_prev

        print(f"After alignment: {len(merged)} common months (months {merged['month'].min()}-{merged['month'].max()})")
        print(f"Progress range: actual={merged['actual'].min():.1f}%-{merged['actual'].max():.1f}%, "
              f"prediction(anchored)={merged['p50_anchored'].min():.1f}%-{merged['p50_anchored'].max():.1f}%")

    else:
        # — Default: no second anchoring for evaluation stage, directly respect prediction CSV's cumulative percentage
        merged['baseline_prev'] = 0.0
        # For robustness, can do safe clipping, but no shifting/scaling
        merged['actual'] = merged['actual'].clip(0, 100)
        merged['p50'] = merged['p50'].clip(0, 100)
        if 'p90' in merged.columns:
            merged['p90'] = np.maximum(merged['p90'].clip(0, 100), merged['p50'])

        print(
            f"After alignment (no second anchoring): {len(merged)} common months (months {merged['month'].min()}-{merged['month'].max()})")
        print(f"Progress range: actual={merged['actual'].min():.1f}%-{merged['actual'].max():.1f}%, "
              f"prediction(original)={merged['p50'].min():.1f}%-{merged['p50'].max():.1f}%")

    return merged


def calculate_metrics(pred, actual):
    """
    Calculate evaluation metrics (compatible with sklearn and numpy implementations)

    Returns:
        dict: Various metrics
    """
    pred = np.array(pred)
    actual = np.array(actual)

    # Basic metrics
    if HAS_SKLEARN:
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
    else:
        # numpy implementation
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Percentage error
    errors = pred - actual
    abs_errors = np.abs(errors)

    # MAPE (avoid division by zero)
    mask = actual > 0.1  # Only calculate MAPE for months with progress>0.1%
    if mask.any():
        mape = np.mean(abs_errors[mask] / actual[mask]) * 100
    else:
        mape = np.nan

    # WAPE (weighted absolute percentage error)
    wape = np.sum(abs_errors) / np.sum(np.abs(actual)) * 100

    # Maximum error
    max_error = np.max(abs_errors)
    max_error_idx = np.argmax(abs_errors)

    # Sign check (overestimation vs underestimation)
    overestimate_ratio = np.mean(errors > 0) * 100

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None,
        'wape': float(wape),
        'max_error': float(max_error),
        'max_error_idx': int(max_error_idx),
        'overestimate_ratio': float(overestimate_ratio),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
    }


def plot_comparison(df, metrics, output_dir):
    """Generate comparison charts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    months = df['month'].values
    actual = df['actual'].values
    pred = (df['p50_anchored'] if 'p50_anchored' in df.columns else df['p50']).values

    # Fix maximum error month (use actual month value instead of index)
    max_err_idx = metrics['max_error_idx']
    max_err_month = months[max_err_idx]

    # ========== Chart 1: Main Comparison Chart (Prediction vs Actual) ==========
    fig, ax = plt.subplots(figsize=(14, 8))

    # Actual progress
    ax.plot(months, actual, marker='o', linewidth=2.5, markersize=8,
            label='Actual Progress', color='#2E7D32', zorder=3)

    # Predicted progress (P50)
    ax.plot(months, pred, marker='s', linewidth=2.5, markersize=8,
            label='Predicted Progress (P50)', color='#1976D2', linestyle='--', zorder=2)

    # Uncertainty interval (P90)
    p90_col = 'p90_anchored' if 'p90_anchored' in df.columns else ('p90' if 'p90' in df.columns else None)
    if p90_col is not None:
        p90 = df[p90_col].values
        ax.fill_between(months, pred, p90, alpha=0.2, color='#1976D2',
                        label='Uncertainty Interval (P50-P90)')

    # Mark maximum error point (use correct month)
    ax.scatter(max_err_month, actual[max_err_idx],
               s=200, color='red', zorder=4, marker='x', linewidths=3)
    ax.annotate(f'Max Error\n{metrics["max_error"]:.1f}%\n(Month {max_err_month})',
                xy=(max_err_month, actual[max_err_idx]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=10, color='red',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Style settings
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Progress (%)', fontsize=14, fontweight='bold')
    ax.set_title('LSTM Prediction vs Actual Progress Comparison\n' +
                 f'MAE={metrics["mae"]:.2f}%  |  ' +
                 f'RMSE={metrics["rmse"]:.2f}%  |  ' +
                 f'R²={metrics["r2"]:.4f}',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(months.min() - 0.5, months.max() + 0.5)
    ax.set_ylim(0, 105)

    # Add metrics text box
    textstr = f'Mean Error: {metrics["mean_error"]:+.2f}%\n'
    textstr += f'WAPE: {metrics["wape"]:.2f}%\n'
    textstr += f'Overestimate Ratio: {metrics["overestimate_ratio"]:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_main.png', dpi=300, bbox_inches='tight')
    print(f"Saved: comparison_main.png")
    plt.close()

    # ========== Chart 2: Error Analysis ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    errors = pred - actual
    abs_errors = np.abs(errors)

    # 2.1 Error variation over time
    ax = axes[0, 0]
    ax.plot(months, errors, marker='o', linewidth=2, color='#D32F2F')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.fill_between(months, 0, errors, alpha=0.3, color='#D32F2F')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Error (Prediction - Actual) %', fontsize=12)
    ax.set_title('Error Variation Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2.2 Error distribution histogram
    ax = axes[0, 1]
    ax.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='#1976D2')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(errors):.2f}%')
    ax.set_xlabel('Error (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2.3 Scatter plot
    ax = axes[1, 0]
    ax.scatter(actual, pred, s=100, alpha=0.6, edgecolors='black', linewidth=1)

    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction Line')

    ax.set_xlabel('Actual Progress (%)', fontsize=12)
    ax.set_ylabel('Predicted Progress (%)', fontsize=12)
    ax.set_title(f'Prediction vs Actual Scatter Plot (R²={metrics["r2"]:.4f})',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.4 Absolute error vs progress variation
    ax = axes[1, 1]
    ax.scatter(actual, abs_errors, s=100, alpha=0.6,
               edgecolors='black', linewidth=1, color='orange')
    ax.set_xlabel('Actual Progress (%)', fontsize=12)
    ax.set_ylabel('Absolute Error (%)', fontsize=12)
    ax.set_title('Absolute Error vs Actual Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(actual) >= 3:
        z = np.polyfit(actual, abs_errors, min(2, len(actual) - 1))
        p = np.poly1d(z)
        x_smooth = np.linspace(actual.min(), actual.max(), 100)
        ax.plot(x_smooth, p(x_smooth), "r--", linewidth=2, label='Trend Line')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: error_analysis.png")
    plt.close()

    # ========== Chart 3: Monthly Growth Comparison ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    base_prev = float(df['baseline_prev'].iloc[0]) if 'baseline_prev' in df.columns else 0.0
    actual_growth = np.r_[actual[0] - base_prev, np.diff(actual)]
    pred_growth = np.r_[pred[0] - base_prev, np.diff(pred)]

    x = np.arange(len(months))
    width = 0.35

    ax.bar(x - width / 2, actual_growth, width, label='Actual Monthly Growth',
           alpha=0.8, edgecolor='black')
    ax.bar(x + width / 2, pred_growth, width, label='Predicted Monthly Growth',
           alpha=0.8, edgecolor='black')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Monthly Progress Growth (%)', fontsize=12)
    ax.set_title('Monthly Progress Growth Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_growth.png', dpi=300, bbox_inches='tight')
    print(f"Saved: monthly_growth.png")
    plt.close()


def print_metrics_summary(metrics, df):
    """Print metrics summary"""
    print("\n" + "=" * 70)
    print("Evaluation Metrics")
    print("=" * 70)

    print("\n Overall Performance:")
    print(f"  • MAE (Mean Absolute Error):        {metrics['mae']:>10.2f}%")
    print(f"  • RMSE (Root Mean Square Error):    {metrics['rmse']:>10.2f}%")
    print(f"  • R² (Coefficient of Determination): {metrics['r2']:>10.4f}")

    if metrics['mape'] is not None:
        print(f"  • MAPE (Mean Absolute Percentage Error): {metrics['mape']:>10.2f}%")

    print(f"  • WAPE (Weighted Absolute Percentage Error): {metrics['wape']:>10.2f}%")

    # Use correct month value
    max_err_month = df['month'].iloc[metrics['max_error_idx']]

    print("\n Error Analysis:")
    print(f"  • Mean Error:                   {metrics['mean_error']:>+10.2f}%")
    print(f"  • Error Standard Deviation:     {metrics['std_error']:>10.2f}%")
    print(f"  • Maximum Absolute Error:       {metrics['max_error']:>10.2f}% (Month {max_err_month})")
    print(f"  • Overestimation Ratio:         {metrics['overestimate_ratio']:>10.1f}%")

    print("\n Interpretation:")

    # R² interpretation
    if metrics['r2'] > 0.9:
        print("Excellent fit (R² > 0.9)")
    elif metrics['r2'] > 0.8:
        print("Very good fit (R² > 0.8)")
    elif metrics['r2'] > 0.6:
        print("Moderate fit (R² > 0.6)")
    else:
        print("Lower fit quality (R² < 0.6)")

    # MAE interpretation
    if metrics['mae'] < 3:
        print("Extremely high accuracy (MAE < 3%)")
    elif metrics['mae'] < 5:
        print("Very high accuracy (MAE < 5%)")
    elif metrics['mae'] < 10:
        print("Moderate accuracy (MAE < 10%)")
    else:
        print("Accuracy needs improvement (MAE > 10%)")

    # Bias check
    if abs(metrics['mean_error']) < 1:
        print("No significant bias (|mean| < 1%)")
    elif metrics['mean_error'] > 1:
        print(f"Systematic overestimation ({metrics['mean_error']:.2f}%)")
    else:
        print(f"Systematic underestimation ({metrics['mean_error']:.2f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare prediction with actual progress")
    parser.add_argument("--predicted", required=True, help="Prediction CSV path")
    parser.add_argument("--actual", required=True, help="Actual progress CSV path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--actual_col", default=None,
                        help="Actual progress column name (optional, auto-detect if not specified)")
    parser.add_argument("--align_mode", choices=["none", "offset"], default="none",
                        help="Whether to do second alignment for evaluation stage: none=no alignment (default, directly use pred CSV cumulative %), offset=shift starting point to previous month's actual value")
    parser.add_argument("--bias_correction", choices=["none", "global", "linear"], default="none",
                        help="Post-hoc bias correction: none (default), global (shift decay), linear (least squares linear)")
    parser.add_argument("--report_calibrated", action="store_true",
                        help="If specified, calculate and report metrics and charts based on calibrated predictions (if not specified, report based on original predictions)")

    args = parser.parse_args()

    # Load data
    df = load_and_validate(args.predicted, args.actual, args.actual_col, align_mode=args.align_mode)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("Calculating Evaluation Metrics")
    print("=" * 70)
    pred_col = 'p50_anchored' if 'p50_anchored' in df.columns else 'p50'
    pred_raw = df[pred_col].to_numpy(dtype=float)
    act = df['actual'].to_numpy(dtype=float)

    # ========= Define a small calibration function (can be placed in main) =========
    def apply_bias_correction(x_pred, x_act, mode: str):
        if mode == "none":
            return x_pred

        x_pred = np.asarray(x_pred, dtype=float)
        x_act = np.asarray(x_act, dtype=float)

        if mode == "global":
            # Global shift: full at starting point, gradually decay toward endpoint, avoid exceeding 100%
            base = x_pred[0]
            denom = max(1e-6, (100.0 - base))
            t = np.clip((x_pred - base) / denom, 0.0, 1.0)  # 0=start, 1=end
            b = float(np.mean(x_act - x_pred))
            x_adj = x_pred + b * (1.0 - t)
            return np.clip(x_adj, 0.0, 100.0)

        if mode == "linear":
            # Least squares linear: actual ≈ a + b * pred
            A = np.vstack([np.ones_like(x_pred), x_pred]).T
            coef, *_ = np.linalg.lstsq(A, x_act, rcond=None)  # [a, b]
            a, b = float(coef[0]), float(coef[1])
            x_adj = a + b * x_pred
            return np.clip(x_adj, 0.0, 100.0)

        return x_pred

    # ========= Optional: apply calibration and decide which sequence to use for evaluation =========
    if args.report_calibrated and args.bias_correction != "none":
        pred_eval = apply_bias_correction(pred_raw, act, args.bias_correction)
        # Overwrite column used for evaluation/plotting with calibrated sequence
        df[pred_col] = pred_eval
        eval_note = f"(calibrated: {args.bias_correction})"
    else:
        pred_eval = pred_raw
        eval_note = "(raw)"

    metrics = calculate_metrics(df[pred_col].values, df['actual'].values)
    # Print summary (pass df to get correct month)
    print_metrics_summary(metrics, df)

    # Generate charts
    print("\n" + "=" * 70)
    print("Generating Comparison Charts")
    print("=" * 70)
    plot_comparison(df, metrics, args.output_dir)

    # Save metrics
    metrics_file = Path(args.output_dir) / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics: metrics.json")

    # Save aligned data
    comparison_file = Path(args.output_dir) / 'comparison_data.csv'
    df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"Saved comparison data: comparison_data.csv")

    # ========== NEW: Export comparison timeseries for governance triggers ==========
    print("\n" + "=" * 70)
    print("Exporting Comparison Timeseries for Governance Analysis")
    print("=" * 70)

    try:
        # Build comparison dataframe with aligned time series
        comparison_df = pd.DataFrame({
            "month": df['month'].values,
            "forecast_p50": df[pred_col].values,  # Use the same prediction column as evaluation
        })

        # Add P90 if available
        if 'p90' in df.columns or 'p90_anchored' in df.columns:
            p90_col = 'p90_anchored' if 'p90_anchored' in df.columns else 'p90'
            comparison_df["forecast_p90"] = df[p90_col].values

        # ========== NEW: Load DT Hybrid from Preview_progress_fusion.csv ==========
        print("Loading DT hybrid verified progress from Preview_progress_fusion.csv...")
        dt_hybrid_loaded = False

        # Try to find DT hybrid CSV in common locations
        dt_hybrid_paths = [
            Path(args.actual).parent / "Preview_progress_fusion.csv",  # Same dir as actual CSV
            Path("input_csv data") / "real_project" / "Preview_progress_fusion.csv",
        ]

        for dt_path in dt_hybrid_paths:
            if dt_path.exists():
                print(f"  Found DT hybrid at: {dt_path}")
                try:
                    dt_df = pd.read_csv(dt_path)
                    dt_df.columns = [c.strip() for c in dt_df.columns]

                    # Extract Month and APS_CumWeighted6
                    if 'Month' in dt_df.columns and 'APS_CumWeighted6' in dt_df.columns:
                        # Convert Month (M01, M02, ...) to integer
                        def extract_month_num(m):
                            if pd.isna(m):
                                return None
                            s = str(m).strip().upper()
                            match = re.search(r'\d+', s)
                            return int(match.group()) if match else None

                        dt_df['month_num'] = dt_df['Month'].apply(extract_month_num)
                        dt_df = dt_df.dropna(subset=['month_num', 'APS_CumWeighted6'])

                        # Convert APS_CumWeighted6 from string percentage to float
                        def parse_percentage(val):
                            if pd.isna(val):
                                return np.nan
                            s = str(val).strip().replace('%', '')
                            try:
                                return float(s)
                            except:
                                return np.nan

                        dt_df['APS_CumWeighted6_float'] = dt_df['APS_CumWeighted6'].apply(parse_percentage)
                        dt_df = dt_df.dropna(subset=['APS_CumWeighted6_float'])

                        # Create mapping dictionary
                        dt_dict = dict(zip(
                            dt_df['month_num'].astype(int),
                            dt_df['APS_CumWeighted6_float']
                        ))

                        # Map to comparison_df months
                        comparison_df["dt_hybrid"] = comparison_df['month'].map(dt_dict)
                        dt_hybrid_loaded = True
                        print(f"Loaded DT hybrid for {comparison_df['dt_hybrid'].notna().sum()} months")
                        break
                    else:
                        print(f"DT hybrid CSV missing required columns")
                except Exception as e:
                    print(f"Error loading DT hybrid: {e}")
                    import traceback
                    traceback.print_exc()

        if not dt_hybrid_loaded:
            print(f"DT hybrid CSV not found, using actual progress as fallback")
            comparison_df["dt_hybrid"] = df['actual'].values

        # ========== Load baseline from Chengbei_24m_work.csv ==========
        print("\nLoading baseline from Chengbei_24m_work.csv...")
        baseline_loaded = False

        # Try to find baseline CSV in common locations
        baseline_paths = [
            Path(args.actual).parent / "Chengbei_24m_work.csv",  # Same dir as actual CSV
            Path("input_csv data") / "real_project" / "Chengbei_24m_work.csv",
        ]

        baseline_df = None
        for baseline_path in baseline_paths:
            if baseline_path.exists():
                print(f"  Found baseline at: {baseline_path}")
                try:
                    baseline_df = pd.read_csv(baseline_path)
                    # Clean column names (remove BOM and whitespace)
                    baseline_df.columns = [c.strip().replace('\ufeff', '') for c in baseline_df.columns]

                    # Extract month_index and cumulative_share_pct
                    if 'month_index' in baseline_df.columns and 'cumulative_share_pct' in baseline_df.columns:
                        # CRITICAL FIX: Drop rows with NaN in either column (bottom summary rows)
                        baseline_clean = baseline_df[['month_index', 'cumulative_share_pct']].copy()
                        baseline_clean = baseline_clean.dropna(subset=['month_index', 'cumulative_share_pct'])

                        # Now safe to convert to int/float
                        baseline_clean['month_index'] = baseline_clean['month_index'].astype(int)
                        baseline_clean['cumulative_share_pct'] = baseline_clean['cumulative_share_pct'].astype(float)

                        baseline_dict = dict(zip(
                            baseline_clean['month_index'],
                            baseline_clean['cumulative_share_pct']
                        ))

                        # Map to comparison_df months
                        comparison_df["baseline"] = comparison_df['month'].map(baseline_dict)
                        baseline_loaded = True
                        print(f"Loaded baseline for {comparison_df['baseline'].notna().sum()} months")
                        break
                    else:
                        print(f"Baseline CSV missing required columns")
                except Exception as e:
                    print(f"Error loading baseline: {e}")
                    import traceback
                    traceback.print_exc()

        if not baseline_loaded:
            print(f"Baseline CSV not found or failed to load, using NaN placeholder")
            comparison_df["baseline"] = np.nan

        # Calculate differences
        comparison_df["diff_f50_vs_dt"] = comparison_df["forecast_p50"] - comparison_df["dt_hybrid"]
        comparison_df["diff_dt_vs_base"] = comparison_df["dt_hybrid"] - comparison_df["baseline"]
        comparison_df["diff_f50_vs_base"] = comparison_df["forecast_p50"] - comparison_df["baseline"]

        # Save to outputs directory
        output_dir_path = Path(args.output_dir)
        timeseries_file = output_dir_path / 'comparison_timeseries.csv'
        comparison_df.to_csv(timeseries_file, index=False, encoding='utf-8-sig', float_format='%.4f')

        print(f"\n✓ Saved comparison timeseries: comparison_timeseries.csv")
        print(f"  Columns: {list(comparison_df.columns)}")
        print(f"  Months: {len(comparison_df)} ({comparison_df['month'].min()}-{comparison_df['month'].max()})")
        if dt_hybrid_loaded:
            print(f"  DT Hybrid: Loaded from Preview_progress_fusion.csv (APS_CumWeighted6)")
        else:
            print(f"  DT Hybrid: Using actual progress as fallback")
        if baseline_loaded:
            print(f"  Baseline: Loaded from Chengbei_24m_work.csv")
        else:
            print(f"  Baseline: Not available (placeholder NaN)")

    except Exception as e:
        print(f"Warning: Failed to export comparison timeseries: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Comparison evaluation complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()