"""
Governance trigger and risk evaluation module for LSTM+DT comparison.

This module analyzes deviations between:
- LSTM forecast (P50/P90)
- DT hybrid verified progress
- Contract baseline S-curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Set font for better display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_governance_risks(
        csv_path: str,
        soft_threshold: float = 0.05,
        hard_threshold: float = 0.10,
        min_duration: int = 2,
        output_csv: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:

    """
    Evaluate governance risks from comparison timeseries data.

    This function reads aligned time series data containing:
    - forecast_p50: LSTM P50 cumulative percentage
    - forecast_p90: LSTM P90 cumulative percentage (optional)
    - dt_hybrid: DT verified hybrid progress percentage
    - baseline: Contract baseline S-curve percentage

    And identifies months requiring governance intervention based on:
    - Deviation thresholds (soft/hard)
    - Persistence duration (consecutive high-risk months)

    Args:
        csv_path: Path to comparison_timeseries.csv file
        soft_threshold: Soft warning threshold (percentage points, default 5%)
        hard_threshold: Hard risk threshold (percentage points, default 10%)
        min_duration: Minimum consecutive months for trigger (default 2)
        output_csv: Optional output path for results (default: same dir as input)

    Returns:
        tuple: (df_with_triggers, summary_dict)
            - df_with_triggers: DataFrame with risk flags and governance triggers
            - summary_dict: Dictionary containing summary statistics

    Example:
        >>> df, summary = evaluate_governance_risks(
        ...     "outputs/comparison_timeseries.csv",
        ...     soft_threshold=0.05,
        ...     hard_threshold=0.10,
        ...     min_duration=2
        ... )
        >>> print(f"Triggered months: {summary['triggered_months']}")
    """
    # --- Normalize thresholds: accept both ratio (0.05) and percent-points (5.0) ---
    if soft_threshold > 1.0:
        soft_threshold = soft_threshold / 100.0
    if hard_threshold > 1.0:
        hard_threshold = hard_threshold / 100.0
    if hard_threshold < soft_threshold:
        hard_threshold = soft_threshold

    print(f"[DEBUG] soft_threshold={soft_threshold:.4f} (ratio), hard_threshold={hard_threshold:.4f} (ratio)")

    # ==================== 1. Load Data ====================
    print("\n" + "=" * 70)
    print("GOVERNANCE RISK EVALUATION")
    print("=" * 70)
    print(f"Loading data from: {csv_path}")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['month', 'forecast_p50', 'dt_hybrid', 'diff_f50_vs_dt', 'diff_dt_vs_base']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✓ Loaded {len(df)} months (M{df['month'].min()}-M{df['month'].max()})")
    print(f"  Columns: {list(df.columns)}")

    # ==================== 2. Calculate Absolute Deviations ====================
    print("\n" + "-" * 70)
    print("Calculating Absolute Deviations")
    print("-" * 70)

    # Convert differences to absolute values (percentage points)
    df['abs_diff_f50_vs_dt'] = df['diff_f50_vs_dt'].abs()
    df['abs_diff_dt_vs_base'] = df['diff_dt_vs_base'].abs()

    # Handle NaN in baseline comparison (if baseline not available)
    baseline_available = df['baseline'].notna().any()
    if not baseline_available:
        print("Warning: Baseline data not available, dt_vs_base analysis skipped")
        df['abs_diff_dt_vs_base'] = 0.0

    print(f"  Max deviation (F50 vs DT):   {df['abs_diff_f50_vs_dt'].max():.2f}%")
    if baseline_available:
        print(f"  Max deviation (DT vs Base):  {df['abs_diff_dt_vs_base'].max():.2f}%")

    # ==================== 3. Assign Risk Flags ====================
    print("\n" + "-" * 70)
    print("Assigning Risk Levels")
    print("-" * 70)
    print(f"  Soft threshold: {soft_threshold * 100:.1f}%")
    print(f"  Hard threshold: {hard_threshold * 100:.1f}%")

    def assign_risk_flag(abs_diff: float, soft_pct: float, hard_pct: float) -> str:
        """
        Assign risk level based on absolute difference.

        Args:
            abs_diff: Absolute difference in percentage points (e.g., 18.41 for 18.41%)
            soft_pct: Soft threshold in percentage points (e.g., 5.0 for 5%)
            hard_pct: Hard threshold in percentage points (e.g., 10.0 for 10%)
        """
        if abs_diff <= soft_pct:
            return "ok"
        elif abs_diff <= hard_pct:
            return "warning"
        else:
            return "high_risk"

    # Convert thresholds to percentage points for comparison
    soft_pct = soft_threshold * 100  # 0.05 -> 5.0
    hard_pct = hard_threshold * 100  # 0.10 -> 10.0

    # Assign risk flags for both comparison dimensions
    df['risk_flag_f50_vs_dt'] = df['abs_diff_f50_vs_dt'].apply(
        lambda x: assign_risk_flag(x, soft_pct, hard_pct)
    )

    if baseline_available:
        df['risk_flag_dt_vs_base'] = df['abs_diff_dt_vs_base'].apply(
            lambda x: assign_risk_flag(x, soft_pct, hard_pct)
        )
    else:
        df['risk_flag_dt_vs_base'] = "ok"

    # Count risk levels
    risk_counts_f50 = df['risk_flag_f50_vs_dt'].value_counts().to_dict()
    print(f"\n  Risk distribution (F50 vs DT):")
    print(f"    OK:        {risk_counts_f50.get('ok', 0)} months")
    print(f"    Warning:   {risk_counts_f50.get('warning', 0)} months")
    print(f"    High Risk: {risk_counts_f50.get('high_risk', 0)} months")

    if baseline_available:
        risk_counts_base = df['risk_flag_dt_vs_base'].value_counts().to_dict()
        print(f"\n  Risk distribution (DT vs Baseline):")
        print(f"    OK:        {risk_counts_base.get('ok', 0)} months")
        print(f"    Warning:   {risk_counts_base.get('warning', 0)} months")
        print(f"    High Risk: {risk_counts_base.get('high_risk', 0)} months")

    # ==================== 4. Identify Governance Triggers ====================
    print("\n" + "-" * 70)
    print("Identifying Governance Triggers")
    print("-" * 70)
    print(f"  Min duration: {min_duration} consecutive months")

    # Initialize governance trigger column
    df['governance_trigger'] = False

    # Find consecutive high-risk periods
    def find_consecutive_high_risk(risk_series: pd.Series, min_len: int) -> List[int]:
        """Find indices of consecutive high-risk periods >= min_len"""
        trigger_indices = []
        current_streak = []

        for idx, risk in enumerate(risk_series):
            if risk == "high_risk":
                current_streak.append(idx)
            else:
                # End of streak
                if len(current_streak) >= min_len:
                    trigger_indices.extend(current_streak)
                current_streak = []

        # Check final streak
        if len(current_streak) >= min_len:
            trigger_indices.extend(current_streak)

        return trigger_indices

    # Find triggers from both dimensions
    trigger_idx_f50 = find_consecutive_high_risk(df['risk_flag_f50_vs_dt'], min_duration)
    trigger_idx_base = []
    if baseline_available:
        trigger_idx_base = find_consecutive_high_risk(df['risk_flag_dt_vs_base'], min_duration)

    # Combine triggers (union of both dimensions)
    all_trigger_idx = sorted(set(trigger_idx_f50 + trigger_idx_base))
    df.loc[all_trigger_idx, 'governance_trigger'] = True

    triggered_months = df.loc[df['governance_trigger'], 'month'].tolist()

    print(f"\n  ✓ Identified {len(triggered_months)} months requiring governance intervention")
    if triggered_months:
        print(f"    Triggered months: M{', M'.join(map(str, triggered_months))}")
    else:
        print(f"    No governance triggers identified (all deviations within acceptable range)")

    # ==================== 5. Generate Summary Statistics ====================
    summary = {
        'total_months': len(df),
        'soft_threshold': soft_threshold,
        'hard_threshold': hard_threshold,
        'min_duration': min_duration,
        'triggered_count': len(triggered_months),
        'triggered_months': triggered_months,
        'max_abs_diff_f50_vs_dt': float(df['abs_diff_f50_vs_dt'].max()),
        'mean_abs_diff_f50_vs_dt': float(df['abs_diff_f50_vs_dt'].mean()),
        'baseline_available': baseline_available,
    }

    if baseline_available:
        summary['max_abs_diff_dt_vs_base'] = float(df['abs_diff_dt_vs_base'].max())
        summary['mean_abs_diff_dt_vs_base'] = float(df['abs_diff_dt_vs_base'].mean())

    # Risk distribution
    summary['risk_distribution_f50_vs_dt'] = risk_counts_f50
    if baseline_available:
        summary['risk_distribution_dt_vs_base'] = risk_counts_base

    # ==================== 6. Save Results ====================
    if output_csv is None:
        output_csv = csv_path.parent / 'comparison_with_triggers.csv'
    else:
        output_csv = Path(output_csv)

    df.to_csv(output_csv, index=False, encoding='utf-8-sig', float_format='%.4f')
    print(f"\n✓ Saved results to: {output_csv}")

    # ==================== 7. Generate Visualizations ====================
    print("\n" + "=" * 70)
    print("Generating Governance Visualizations")
    print("=" * 70)

    try:
        output_dir = output_csv.parent
        generate_governance_visuals(
            df=df,
            output_dir=str(output_dir),
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold
        )
        print(f"\n✓ Visualizations complete")
    except Exception as e:
        print(f"\nWarning: Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70 + "\n")

    # Do NOT overwrite attrs here. Use attrs already set by evaluate_governance_risks
    # or store as percent-points consistently:
    df.attrs['soft_threshold'] = soft_threshold * 100.0 if soft_threshold <= 1.0 else soft_threshold
    df.attrs['hard_threshold'] = hard_threshold * 100.0 if hard_threshold <= 1.0 else hard_threshold

    return df, summary


def print_governance_summary(summary: Dict) -> None:
    """
    Pretty print governance risk summary.

    Args:
        summary: Summary dictionary from evaluate_governance_risks
    """
    print("\n" + "=" * 70)
    print("GOVERNANCE RISK SUMMARY")
    print("=" * 70)

    print(f"\nAnalysis Parameters:")
    print(f"  Total months analyzed:    {summary['total_months']}")
    print(f"  Soft threshold:           {summary['soft_threshold'] * 100:.1f}%")
    print(f"  Hard threshold:           {summary['hard_threshold'] * 100:.1f}%")
    print(f"  Min consecutive duration: {summary['min_duration']} months")

    print(f"\nDeviation Statistics:")
    print(f"  F50 vs DT - Max:  {summary['max_abs_diff_f50_vs_dt']:.2f}%")
    print(f"  F50 vs DT - Mean: {summary['mean_abs_diff_f50_vs_dt']:.2f}%")

    if summary['baseline_available']:
        print(f"  DT vs Base - Max:  {summary['max_abs_diff_dt_vs_base']:.2f}%")
        print(f"  DT vs Base - Mean: {summary['mean_abs_diff_dt_vs_base']:.2f}%")
    else:
        print(f"  DT vs Base: Baseline data not available")

    print(f"\nGovernance Triggers:")
    print(f"  Triggered months: {summary['triggered_count']}")
    if summary['triggered_months']:
        print(f"  Month IDs: {summary['triggered_months']}")
    else:
        print(f"  Status: No intervention required")

    print(f"\nRisk Distribution (F50 vs DT):")
    dist = summary['risk_distribution_f50_vs_dt']
    print(f"  OK:        {dist.get('ok', 0):3d} months ({dist.get('ok', 0) / summary['total_months'] * 100:5.1f}%)")
    print(
        f"  Warning:   {dist.get('warning', 0):3d} months ({dist.get('warning', 0) / summary['total_months'] * 100:5.1f}%)")
    print(
        f"  High Risk: {dist.get('high_risk', 0):3d} months ({dist.get('high_risk', 0) / summary['total_months'] * 100:5.1f}%)")

    if summary['baseline_available']:
        print(f"\nRisk Distribution (DT vs Baseline):")
        dist_base = summary['risk_distribution_dt_vs_base']
        print(
            f"  OK:        {dist_base.get('ok', 0):3d} months ({dist_base.get('ok', 0) / summary['total_months'] * 100:5.1f}%)")
        print(
            f"  Warning:   {dist_base.get('warning', 0):3d} months ({dist_base.get('warning', 0) / summary['total_months'] * 100:5.1f}%)")
        print(
            f"  High Risk: {dist_base.get('high_risk', 0):3d} months ({dist_base.get('high_risk', 0) / summary['total_months'] * 100:5.1f}%)")

    print("=" * 70 + "\n")


def plot_risk_heatmap(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    Plot risk level heatmap showing risk flags for each month.

    Args:
        df: DataFrame with risk_flag columns
        output_path: Output file path (if None, uses default location)
    """
    print("\n" + "-" * 70)
    print("Generating Risk Heatmap")
    print("-" * 70)

    # Define risk level mapping
    risk_map = {'ok': 0, 'warning': 1, 'high_risk': 2}
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

    # Prepare data for heatmap
    months = df['month'].values

    # Create risk matrix (2 rows: F50 vs DT, DT vs Base)
    risk_data = []
    row_labels = []

    if 'risk_flag_f50_vs_dt' in df.columns:
        risk_data.append([risk_map.get(x, 0) for x in df['risk_flag_f50_vs_dt']])
        row_labels.append('F50 vs DT')

    if 'risk_flag_dt_vs_base' in df.columns and df['baseline'].notna().any():
        risk_data.append([risk_map.get(x, 0) for x in df['risk_flag_dt_vs_base']])
        row_labels.append('DT vs Base')

    if not risk_data:
        print("No risk data available for heatmap")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot heatmap
    im = ax.imshow(risk_data, cmap=plt.cm.colors.ListedColormap(colors),
                   aspect='auto', vmin=0, vmax=2)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels([f'M{m}' for m in months], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Add governance trigger markers
    if 'governance_trigger' in df.columns:
        trigger_indices = np.where(df['governance_trigger'].values)[0]
        for idx in trigger_indices:
            for row in range(len(row_labels)):
                ax.add_patch(plt.Rectangle((idx - 0.5, row - 0.5), 1, 1,
                                           fill=False, edgecolor='black',
                                           linewidth=3))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['OK', 'Warning', 'High Risk'])

    # Labels and title
    ax.set_xlabel('Month', fontsize=12)
    ax.set_title('Governance Risk Heatmap (Black border = Trigger)',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.set_xticks(np.arange(len(months)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(row_labels)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path(df.attrs.get('output_dir', '.')) / 'risk_heatmap.png'

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_progress_with_triggers(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    Plot progress curves with governance trigger markers.

    Shows:
    - Forecast P50/P90
    - DT Hybrid verified progress
    - Baseline S-curve
    - Vertical bands for triggered months

    Args:
        df: DataFrame with progress data and governance_trigger column
        output_path: Output file path (if None, uses default location)
    """
    print("\n" + "-" * 70)
    print("Generating Progress Curve with Triggers")
    print("-" * 70)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})

    months = df['month'].values

    # ========== Top Panel: Progress Curves ==========

    # Plot forecast P50
    if 'forecast_p50' in df.columns:
        ax1.plot(months, df['forecast_p50'], 'b-', linewidth=2.5,
                 label='LSTM Forecast P50', marker='o', markersize=6)

    # Plot forecast P90 (uncertainty band)
    if 'forecast_p90' in df.columns:
        ax1.plot(months, df['forecast_p90'], 'b--', linewidth=1.5,
                 alpha=0.6, label='LSTM Forecast P90', marker='^', markersize=5)

    # Plot DT hybrid
    if 'dt_hybrid' in df.columns:
        ax1.plot(months, df['dt_hybrid'], 'g-', linewidth=2.5,
                 label='DT Hybrid Verified', marker='s', markersize=6)

    # Plot baseline
    if 'baseline' in df.columns and df['baseline'].notna().any():
        ax1.plot(months, df['baseline'], 'r-', linewidth=2.5,
                 label='Contract Baseline', marker='D', markersize=6)

    # Mark governance trigger months with vertical bands
    if 'governance_trigger' in df.columns:
        trigger_months = df.loc[df['governance_trigger'], 'month'].values
        if len(trigger_months) > 0:
            for month in trigger_months:
                ax1.axvspan(month - 0.5, month + 0.5, alpha=0.2, color='red',
                            zorder=0, label='Governance Trigger' if month == trigger_months[0] else '')

    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Cumulative Progress (%)', fontsize=12)
    ax1.set_title('Construction Progress Comparison with Governance Triggers',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(months.min() - 0.5, months.max() + 0.5)
    ax1.set_ylim(0, 105)

    # ========== Bottom Panel: Deviation Plot ==========

    # Plot deviations
    if 'diff_f50_vs_dt' in df.columns:
        ax2.plot(months, df['diff_f50_vs_dt'], 'b-', linewidth=2,
                 label='F50 - DT', marker='o', markersize=5)

    if 'diff_dt_vs_base' in df.columns and df['baseline'].notna().any():
        ax2.plot(months, df['diff_dt_vs_base'], 'g-', linewidth=2,
                 label='DT - Baseline', marker='s', markersize=5)

    # Add threshold lines
    soft_thresh = df.attrs.get('soft_threshold', 5.0)
    hard_thresh = df.attrs.get('hard_threshold', 10.0)

    # safety
    if hard_thresh < soft_thresh:
        hard_thresh = soft_thresh

    ax2.axhline(y=soft_thresh, color='orange', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Soft Threshold (±{soft_thresh}%)')
    ax2.axhline(y=-soft_thresh, color='orange', linestyle='--',
                linewidth=1.5, alpha=0.7)
    ax2.axhline(y=hard_thresh, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Hard Threshold (±{hard_thresh}%)')
    ax2.axhline(y=-hard_thresh, color='red', linestyle='--',
                linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Mark trigger months
    if 'governance_trigger' in df.columns:
        trigger_months = df.loc[df['governance_trigger'], 'month'].values
        for month in trigger_months:
            ax2.axvspan(month - 0.5, month + 0.5, alpha=0.2, color='red', zorder=0)

    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Deviation (%)', fontsize=12)
    ax2.set_title('Progress Deviations', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(months.min() - 0.5, months.max() + 0.5)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path(df.attrs.get('output_dir', '.')) / 'progress_with_triggers.png'

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_governance_visuals(df: pd.DataFrame,
                                output_dir: str,
                                soft_threshold: float = 0.05,
                                hard_threshold: float = 0.10) -> None:
    """
    Generate all governance visualization plots.

    Args:
        df: DataFrame with risk analysis results
        output_dir: Output directory for plots
        soft_threshold: Soft threshold for reference
        hard_threshold: Hard threshold for reference
    """
    output_dir = Path(output_dir)

    # Store metadata in df.attrs for use in plotting functions
    df.attrs['output_dir'] = str(output_dir)
    # Store thresholds in df.attrs as "percentage points" (e.g., 5, 10)
    soft_pct = soft_threshold * 100 if soft_threshold <= 1.0 else soft_threshold
    hard_pct = hard_threshold * 100 if hard_threshold <= 1.0 else hard_threshold

    df.attrs['soft_threshold'] = float(soft_pct)
    df.attrs['hard_threshold'] = float(hard_pct)

    # Generate plots
    plot_risk_heatmap(df, output_dir / 'risk_heatmap.png')
    plot_progress_with_triggers(df, output_dir / 'progress_with_triggers.png')

    print("-" * 70)


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    """
    Standalone execution example for testing and debugging.

    Usage:
        python Governance_Triggers.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Governance Risk Evaluation for LSTM+DT Comparison"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/comparison_timeseries.csv",
        help="Path to comparison_timeseries.csv (default: outputs/comparison_timeseries.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results (default: same dir as input with '_with_triggers' suffix)"
    )
    parser.add_argument(
        "--soft_threshold",
        type=float,
        default=0.05,
        help="Soft warning threshold in fraction (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--hard_threshold",
        type=float,
        default=0.10,
        help="Hard risk threshold in fraction (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=2,
        help="Minimum consecutive high-risk months for trigger (default: 2)"
    )

    args = parser.parse_args()

    # Run evaluation
    try:
        df_trigger, summary = evaluate_governance_risks(
            csv_path=args.input,
            soft_threshold=args.soft_threshold,
            hard_threshold=args.hard_threshold,
            min_duration=args.min_duration,
            output_csv=args.output
        )

        # Print summary
        print_governance_summary(summary)

        print("Governance risk evaluation complete!")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"   Please ensure comparison_timeseries.csv exists in the specified location.")
        print(f"   This file is generated by Compare_Prediction.py")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)