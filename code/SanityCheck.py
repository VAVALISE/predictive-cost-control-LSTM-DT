import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


# ===========================
# Data Loading
# ===========================
def load_generated_data(file_path="data/generated/synthetic_CN_projects.csv"):
    """
    Load generated project data

    Args:
        file_path: Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ Loaded data: {len(df)} records from {df['project_id'].nunique()} projects")
    return df


# ===========================
# 1. Data Integrity Checks
# ===========================
def check_data_integrity(df):
    """
    Verify data structure integrity

    Args:
        df: DataFrame to check

    Returns:
        dict: Dictionary of check results
    """
    print("\n" + "=" * 60)
    print("1. DATA INTEGRITY CHECKS")
    print("=" * 60)

    results = {}
    issues = []

    # Check 1.1: Continuous months for each project
    print("\n[1.1] Checking month continuity...")
    month_gaps = []
    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        months = project_data['month'].values
        expected_months = np.arange(1, len(months) + 1)

        if not np.array_equal(months, expected_months):
            month_gaps.append(pid)
            issues.append(f"Project {pid}: Non-continuous months detected")

    if len(month_gaps) == 0:
        print("All projects have continuous month sequences")
        results['month_continuity'] = True
    else
        print(f"{len(month_gaps)} projects have month gaps")
        results['month_continuity'] = False

    # Check 1.2: Final month progress = 100%
    print("\n[1.2] Checking final month progress = 100%...")
    incomplete_projects = []
    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        final_progress = project_data.iloc[-1]['progress_pct']

        if abs(final_progress - 100.0) > 0.01:  # Allow 0.01% tolerance
            incomplete_projects.append((pid, final_progress))
            issues.append(f"Project {pid}: Final progress = {final_progress:.2f}%")

    if len(incomplete_projects) == 0:
        print("All projects reach 100% completion at final month")
        results['completion_check'] = True
    else:
        print(f"{len(incomplete_projects)} projects do not reach 100%")
        results['completion_check'] = False

    # Check 1.3: NaN values
    print("\n[1.3] Checking for NaN values...")
    nan_cols = df.columns[df.isna().any()].tolist()
    if len(nan_cols) == 0:
        print("No NaN values found")
        results['no_nan'] = True
    else:
        print(f"NaN values found in columns: {nan_cols}")
        for col in nan_cols:
            issues.append(f"Column '{col}': {df[col].isna().sum()} NaN values")
        results['no_nan'] = False

    # Check 1.4: Negative values
    print("\n[1.4] Checking for negative values...")
    cost_cols = ['material_cost', 'labour_cost', 'equip_cost', 'admin_cost', 'total_cost']
    negative_found = False
    for col in cost_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_found = True
            issues.append(f"Column '{col}': {negative_count} negative values")

    if not negative_found:
        print("No negative values in cost columns")
        results['no_negatives'] = True
    else:
        print("Negative values detected in cost columns")
        results['no_negatives'] = False

    # Check 1.5: Infinite values
    print("\n[1.5] Checking for infinite values...")
    inf_found = False
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_found = True
            issues.append(f"Column '{col}': {inf_count} infinite values")

    if not inf_found:
        print("No infinite values found")
        results['no_infinites'] = True
    else:
        print("Infinite values detected")
        results['no_infinites'] = False

    return results, issues


# ===========================
# 2. Logic and Realism Checks
# ===========================
def check_logic_realism(df):
    """
    Verify logic and realism of generated data

    Args:
        df: DataFrame to check

    Returns:
        dict: Dictionary of check results
    """
    print("\n" + "=" * 60)
    print("2. LOGIC & REALISM CHECKS")
    print("=" * 60)

    results = {}
    issues = []

    # Check 2.1: S-curve characteristics
    print("\n[2.1] Checking S-curve characteristics...")
    s_curve_valid = 0
    s_curve_invalid = 0

    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        progress = project_data['progress_pct'].values

        # Check monotonic increase
        is_monotonic = all(progress[i] <= progress[i + 1] for i in range(len(progress) - 1))

        # Check acceleration (early phase) and deceleration (late phase)
        delta_progress = np.diff(progress)
        mid_point = len(delta_progress) // 2

        early_growth = np.mean(delta_progress[:mid_point])
        late_growth = np.mean(delta_progress[mid_point:])

        # S-curve should have acceleration phase followed by deceleration
        has_acceleration = len(delta_progress) > 5 and any(
            delta_progress[i + 1] > delta_progress[i] for i in range(min(5, len(delta_progress) - 1)))

        if is_monotonic and has_acceleration:
            s_curve_valid += 1
        else:
            s_curve_invalid += 1
            if not is_monotonic:
                issues.append(f"Project {pid}: Non-monotonic progress")

    print(f"{s_curve_valid}/{df['project_id'].nunique()} projects show valid S-curve")
    if s_curve_invalid > 0:
        print(f"{s_curve_invalid} projects may have irregular progress curves")
    results['s_curve_valid_ratio'] = s_curve_valid / df['project_id'].nunique()

    # Check 2.2: Cost distribution pattern
    print("\n[2.2] Checking cost distribution pattern...")
    cost_pattern_valid = 0

    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        total_costs = project_data['total_cost'].values

        # 1) Perform a 3-period rolling average of total costs to remove noise
        smoothed = pd.Series(total_costs).rolling(3, min_periods=1, center=True).mean().values

        # 2) Ignore the first month (it's prone to abnormal peaks due to prepayments/centralized purchasing)
        if len(smoothed) > 1:
            peak_idx = np.argmax(smoothed[1:]) + 1
        else:
            peak_idx = 0

        peak_position = peak_idx / len(smoothed)

        # 3) Relax the definition of "mid-segment" (20%-85%)
        if 0.20 <= peak_position <= 0.85:
            cost_pattern_valid += 1
        else:
            issues.append(f"Project {pid}: Peak at month {peak_idx} ({peak_position * 100:.1f}%)")

    print(f"{cost_pattern_valid}/{df['project_id'].nunique()} projects have mid-phase peak costs")
    results['cost_pattern_valid_ratio'] = cost_pattern_valid / df['project_id'].nunique()

    # Check 2.3: Completion phase cost reduction
    print("\n[2.3] Checking completion phase cost reduction...")
    completion_valid = 0

    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')

        # Check if material and equipment costs reduce in completion phase
        completion_data = project_data[project_data['is_completion_phase'] == True]

        if len(completion_data) > 0:
            # Material costs should decrease in completion phase
            mat_costs = completion_data['material_cost'].values
            final_mat_cost = mat_costs[-1]

            # Equipment costs should decrease
            equip_costs = completion_data['equip_cost'].values
            final_equip_cost = equip_costs[-1]

            # Final month should have very low or zero material/equipment costs
            if final_mat_cost < 1000 and final_equip_cost < 1000:
                completion_valid += 1
            else:
                issues.append(f"Project {pid}: High material/equipment costs in final month")

    print(f"{completion_valid}/{df['project_id'].nunique()} projects show proper completion phase")
    results['completion_phase_valid_ratio'] = completion_valid / df['project_id'].nunique()

    # Check 2.4: Cost ratio validation by project type
    print("\n[2.4] Checking cost ratios by project type...")

    # Switch: Use "weighted progress percentage" instead of simply summing the entire period
    USE_WEIGHTED_RATIOS = True
    # Switch: Exclude the final period (the last 2-3 months) when calculating the percentage
    EXCLUDE_COMPLETION_PHASE = True

    PROJECT_TYPES = {
        "residential": {"mat": (0.40, 0.50), "lab": (0.28, 0.38), "equip": (0.10, 0.14), "admin": (0.06, 0.10)},
        "commercial": {"mat": (0.38, 0.48), "lab": (0.26, 0.36), "equip": (0.12, 0.16), "admin": (0.06, 0.09)},
        "municipal": {"mat": (0.42, 0.52), "lab": (0.25, 0.35), "equip": (0.12, 0.18), "admin": (0.05, 0.08)},
        "industrial": {"mat": (0.35, 0.45), "lab": (0.25, 0.35), "equip": (0.15, 0.20), "admin": (0.05, 0.08)},
        "infrastructure": {"mat": (0.45, 0.55), "lab": (0.22, 0.32), "equip": (0.12, 0.18), "admin": (0.04, 0.07)}
    }

    # 为了贴近真实扰动，给区间加一点容差（±3%）
    MARGIN = 0.03
    for k in PROJECT_TYPES:
        for c in PROJECT_TYPES[k]:
            lo, hi = PROJECT_TYPES[k][c]
            PROJECT_TYPES[k][c] = (max(0.0, lo - MARGIN), min(1.0, hi + MARGIN))

    ratio_valid = 0
    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        ptype = project_data.iloc[0]['project_type']

        # 选择口径：排除收尾期
        if EXCLUDE_COMPLETION_PHASE:
            core = project_data[project_data['is_completion_phase'] == False].copy()
            if core.empty:
                core = project_data.copy()
        else:
            core = project_data.copy()

        if USE_WEIGHTED_RATIOS:
            # 用“各月成本占比”的进度加权平均，更贴近施工主干期
            prog = core['progress_pct'].values
            w = np.r_[prog[0], np.diff(prog)]
            w = np.clip(w, 0, None)
            if w.sum() == 0:
                w = np.ones_like(w)  # 兜底
            w = w / w.sum()

            mat_ratio = np.average(core['material_cost'] / core['total_cost'], weights=w)
            lab_ratio = np.average(core['labour_cost'] / core['total_cost'], weights=w)
            equip_ratio = np.average(core['equip_cost'] / core['total_cost'], weights=w)
            admin_ratio = np.average(core['admin_cost'] / core['total_cost'], weights=w)
        else:
            # 简单全周期求和（原口径）
            total_mat = core['material_cost'].sum()
            total_lab = core['labour_cost'].sum()
            total_equip = core['equip_cost'].sum()
            total_admin = core['admin_cost'].sum()
            total_all = total_mat + total_lab + total_equip + total_admin + 1e-9
            mat_ratio = total_mat / total_all
            lab_ratio = total_lab / total_all
            equip_ratio = total_equip / total_all
            admin_ratio = total_admin / total_all

        expected = PROJECT_TYPES[ptype]
        mat_ok = expected["mat"][0] <= mat_ratio <= expected["mat"][1]
        lab_ok = expected["lab"][0] <= lab_ratio <= expected["lab"][1]
        equip_ok = expected["equip"][0] <= equip_ratio <= expected["equip"][1]
        admin_ok = expected["admin"][0] <= admin_ratio <= expected["admin"][1]

        if mat_ok and lab_ok and equip_ok and admin_ok:
            ratio_valid += 1
        else:
            issues.append(
                f"Project {pid} ({ptype}): ratios out of range "
                f"[mat={mat_ratio:.3f}, lab={lab_ratio:.3f}, equip={equip_ratio:.3f}, admin={admin_ratio:.3f}]"
            )

    print(f"{ratio_valid}/{df['project_id'].nunique()} projects have valid cost ratios")
    results['cost_ratio_valid_ratio'] = ratio_valid / df['project_id'].nunique()

    return results, issues


# ===========================
# 3. Visualization
# ===========================
def visualize_projects(df, output_dir="data/validation_plots/", sample_size=6):
    """
    Create visualizations for sanity checking

    Args:
        df: DataFrame with project data
        output_dir: Directory to save plots
        sample_size: Number of sample projects to plot in detail
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("3. GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Get sample projects (stratified by type)
    sample_projects = []
    for ptype in df['project_type'].unique():
        type_projects = df[df['project_type'] == ptype]['project_id'].unique()
        n_sample = min(2, len(type_projects))
        sample_projects.extend(np.random.choice(type_projects, n_sample, replace=False))
    sample_projects = sample_projects[:sample_size]

    # Plot 1: Progress curves for sample projects
    print("\n[3.1] Plotting progress curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(sample_projects[:6]):
        project_data = df[df['project_id'] == pid].sort_values('month')
        ptype = project_data.iloc[0]['project_type_name']
        duration = project_data.iloc[0]['total_duration_months']

        ax = axes[idx]
        ax.plot(project_data['month'], project_data['progress_pct'],
                marker='o', linewidth=2, markersize=4, label='Progress')
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% Complete')
        ax.fill_between(project_data['month'], 0, project_data['progress_pct'], alpha=0.3)

        # Mark completion phase
        completion_start = project_data[project_data['is_completion_phase']]['month'].min()
        if not pd.isna(completion_start):
            ax.axvspan(completion_start, duration, alpha=0.2, color='orange', label='Completion Phase')

        ax.set_xlabel('Month')
        ax.set_ylabel('Progress (%)')
        ax.set_title(f'{pid} - {ptype}\n({duration} months)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)

    for ax in np.ravel(axes):
        ax.margins(x=0.02)
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'progress_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: progress_curves.png")
    plt.close()

    # Plot 2: Cost breakdown for sample projects
    print("\n[3.2] Plotting cost breakdowns...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(sample_projects[:6]):
        project_data = df[df['project_id'] == pid].sort_values('month')
        ptype = project_data.iloc[0]['project_type_name']

        ax = axes[idx]
        ax.plot(project_data['month'], project_data['material_cost'],
                marker='s', linewidth=2, markersize=3, label='Material', alpha=0.8)
        ax.plot(project_data['month'], project_data['labour_cost'],
                marker='^', linewidth=2, markersize=3, label='Labour', alpha=0.8)
        ax.plot(project_data['month'], project_data['equip_cost'],
                marker='d', linewidth=2, markersize=3, label='Equipment', alpha=0.8)
        ax.plot(project_data['month'], project_data['admin_cost'],
                marker='o', linewidth=2, markersize=3, label='Admin', alpha=0.8)

        # Mark completion phase
        completion_start = project_data[project_data['is_completion_phase']]['month'].min()
        if not pd.isna(completion_start):
            ax.axvspan(completion_start, project_data['month'].max(),
                       alpha=0.2, color='orange')

        ax.set_xlabel('Month')
        ax.set_ylabel('Cost')
        ax.set_title(f'{pid} - {ptype}\nMonthly Cost Breakdown')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')

    for ax in np.ravel(axes):
        ax.margins(x=0.02)
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: cost_breakdown.png")
    plt.close()

    # Plot 3: Total cost curves
    print("\n[3.3] Plotting total cost curves...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, pid in enumerate(sample_projects[:6]):
        project_data = df[df['project_id'] == pid].sort_values('month')
        ptype = project_data.iloc[0]['project_type_name']

        ax = axes[idx]
        ax.plot(project_data['month'], project_data['total_cost'],
                marker='o', linewidth=2.5, markersize=5, color='navy', label='Total Cost')
        ax.fill_between(project_data['month'], 0, project_data['total_cost'],
                        alpha=0.3, color='navy')

        # Mark completion phase
        completion_start = project_data[project_data['is_completion_phase']]['month'].min()
        if not pd.isna(completion_start):
            ax.axvspan(completion_start, project_data['month'].max(),
                       alpha=0.2, color='orange', label='Completion Phase')

        ax.set_xlabel('Month')
        ax.set_ylabel('Total Cost')
        ax.set_title(f'{pid} - {ptype}\nTotal Monthly Cost')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')

    for ax in np.ravel(axes):
        ax.margins(x=0.02)
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_cost_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: total_cost_curves.png")
    plt.close()

    # Plot 4: Cost ratio distribution by project type
    print("\n[3.4] Plotting cost ratio distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    cost_components = ['material_cost', 'labour_cost', 'equip_cost', 'admin_cost']
    component_names = ['Material', 'Labour', 'Equipment', 'Admin']

    for idx, (cost_col, name) in enumerate(zip(cost_components, component_names)):
        ax = axes[idx]

        # Calculate ratios by project and type
        ratios_by_type = {}
        for ptype in df['project_type'].unique():
            type_ratios = []
            for pid in df[df['project_type'] == ptype]['project_id'].unique():
                project_data = df[df['project_id'] == pid]
                total_cost_sum = (project_data['material_cost'].sum() +
                                  project_data['labour_cost'].sum() +
                                  project_data['equip_cost'].sum() +
                                  project_data['admin_cost'].sum())
                ratio = project_data[cost_col].sum() / total_cost_sum
                type_ratios.append(ratio)
            ratios_by_type[ptype] = type_ratios

        # Create box plot
        labels = sorted(ratios_by_type.keys())
        data = [ratios_by_type[p] for p in labels]
        try:
            ax.boxplot(data, tick_labels=labels)  # Matplotlib ≥ 3.9
        except TypeError:
            ax.boxplot(data, labels=labels)  # 兼容旧版本

        ax.set_ylabel('Cost Ratio')
        ax.set_title(f'{name} Cost Ratio by Project Type')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    for ax in np.ravel(axes):
        ax.margins(x=0.02)
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_ratio_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: cost_ratio_distribution.png")
    plt.close()

    # Plot 5: Completion phase analysis
    print("\n[3.5] Plotting completion phase analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Average cost reduction in completion phase
    completion_data_all = []
    for pid in df['project_id'].unique():
        project_data = df[df['project_id'] == pid].sort_values('month')
        completion_phase = project_data[project_data['is_completion_phase'] == True]

        if len(completion_phase) > 0:
            for _, row in completion_phase.iterrows():
                months_from_end = project_data['month'].max() - row['month']
                completion_data_all.append({
                    'months_from_end': months_from_end,
                    'material_cost': row['material_cost'],
                    'labour_cost': row['labour_cost'],
                    'equip_cost': row['equip_cost']
                })

    completion_df = pd.DataFrame(completion_data_all)

    if len(completion_df) > 0:
        # Group by months from end
        grouped = completion_df.groupby('months_from_end').mean()

        ax = axes[0]
        ax.plot(grouped.index, grouped['material_cost'], marker='o', label='Material', linewidth=2)
        ax.plot(grouped.index, grouped['labour_cost'], marker='s', label='Labour', linewidth=2)
        ax.plot(grouped.index, grouped['equip_cost'], marker='^', label='Equipment', linewidth=2)
        ax.set_xlabel('Months from Project End')
        ax.set_ylabel('Average Cost')
        ax.set_title('Cost Reduction in Completion Phase')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.ticklabel_format(style='plain', axis='y')

    # Project duration distribution
    ax = axes[1]
    duration_by_type = df.groupby(['project_type', 'project_id'])['total_duration_months'].first().reset_index()
    duration_by_type.boxplot(column='total_duration_months', by='project_type', ax=ax)
    ax.set_xlabel('Project Type')
    ax.set_ylabel('Duration (months)')
    ax.set_title('Project Duration Distribution by Type')
    ax.get_figure().suptitle('')  # Remove default title
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for ax in np.ravel(axes):
        ax.margins(x=0.02)
        ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: completion_analysis.png")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}")


# ===========================
# Main Sanity Check Function
# ===========================
def run_sanity_check(data_file="data/generated/synthetic_CN_projects.csv",
                     output_dir="data/validation_plots/"):
    """
    Run complete sanity check on generated data

    Args:
        data_file: Path to generated data CSV
        output_dir: Directory to save validation plots
    """
    print("=" * 60)
    print("SANITY CHECK FOR GENERATED PROJECT DATA")
    print("=" * 60)

    # Load data
    df = load_generated_data(data_file)

    # Run integrity checks
    integrity_results, integrity_issues = check_data_integrity(df)

    # Run logic and realism checks
    logic_results, logic_issues = check_logic_realism(df)

    # Generate visualizations
    visualize_projects(df, output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_results = {**integrity_results, **logic_results}
    all_issues = integrity_issues + logic_issues

    print(f"\nIntegrity Checks:")
    print(f"  • Month Continuity: {'PASS' if integrity_results.get('month_continuity') else 'FAIL'}")
    print(f"  • 100% Completion: {'PASS' if integrity_results.get('completion_check') else 'FAIL'}")
    print(f"  • No NaN Values: {'PASS' if integrity_results.get('no_nan') else 'FAIL'}")
    print(f"  • No Negative Values: {'PASS' if integrity_results.get('no_negatives') else 'FAIL'}")
    print(f"  • No Infinite Values: {'PASS' if integrity_results.get('no_infinites') else 'FAIL'}")

    print(f"\nRealism Checks:")
    print(f"  • S-Curve Validity: {logic_results.get('s_curve_valid_ratio', 0) * 100:.1f}% projects")
    print(f"  • Cost Pattern Validity: {logic_results.get('cost_pattern_valid_ratio', 0) * 100:.1f}% projects")
    print(f"  • Completion Phase Logic: {logic_results.get('completion_phase_valid_ratio', 0) * 100:.1f}% projects")
    print(f"  • Cost Ratio Validity: {logic_results.get('cost_ratio_valid_ratio', 0) * 100:.1f}% projects")

    if len(all_issues) > 0:
        print(f"\n{len(all_issues)} issues detected:")
        for issue in all_issues[:10]:  # Show first 10 issues
            print(issue)
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more issues")
    else:
        print("\nNo critical issues detected!")

    print(f"\n{'=' * 60}")
    print(f"Validation complete! Check plots in: {output_dir}")
    print(f"{'=' * 60}")

    return all_results


# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    run_sanity_check(
        data_file="data/generated/synthetic_CN_projects.csv",
        output_dir="data/validation_plots/"
    )