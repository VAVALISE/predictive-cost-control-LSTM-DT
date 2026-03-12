"""
LSTM Case Study - Enhanced Sensitivity Analysis
Additions for R3 revision:
  - Removed hardcoded CODE_DIR; use --project_root CLI arg (reproducibility)
  - Added cohens_d() / interpret_cohens_d() utility functions
  - Added bootstrap_condition_from_csv() to resample per-budget predictions
  - Added compute_budget_effect_sizes() reporting Cohen's d vs nominal budget
  - Bootstrap stability report now includes Cohen's d between upper/lower halves
    of the distribution to quantify within-method spread
"""
import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from scipy import stats

# ==================== Path Configuration ====================
# Paths resolved at runtime via CLI; no hardcoded machine-specific paths.

def resolve_project_root(project_root_arg: str = None) -> Path:
    """
    Resolve project root. Priority:
      1. --project_root CLI argument
      2. Three parents up from this file (LSTM/Case_Study/Sensitivity_Analysis.py)
    """
    if project_root_arg:
        p = Path(project_root_arg).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--project_root not found: {p}")
        return p
    # Auto-detect: expect this file lives at <root>/LSTM/Case_Study/
    p = Path(__file__).resolve().parents[2]
    return p


# ==================== Test Configurations ====================
BUDGET_PERTURBATIONS = [0.8, 0.9, 1.0, 1.1, 1.2]
ENSEMBLE_STRATEGIES  = ['median', 'mean', 'weighted']
FOLD_COMBINATIONS    = [
    [0, 1, 2],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4],
]
BOOTSTRAP_N_ITERATIONS = 1000
BOOTSTRAP_SEED         = 42


# ==================== Effect Size Utilities (NEW for R3) ====================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent samples.

    d = (mean_a - mean_b) / pooled_std

    Pooled SD uses the unbiased (n-1) denominator (Hedges 1981).
    Returns 0.0 if pooled SD is effectively zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d magnitude following Cohen (1988) conventions.
    |d| < 0.2  → negligible
    |d| < 0.5  → small
    |d| < 0.8  → medium
    |d| ≥ 0.8  → large
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_condition_from_csv(
        pred_file: Path,
        n_bootstrap: int = 1000,
        seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Bootstrap resample a comparison_data.csv file to obtain distributions of
    MAE, RMSE, and R² for a single experimental condition.

    This avoids re-running the LSTM model and enables Cohen's d computation
    across budget perturbation conditions without additional model evaluations.

    Args:
        pred_file : Path to comparison_data.csv produced by Compare_Prediction.py
        n_bootstrap : Number of bootstrap iterations
        seed : Random seed for reproducibility

    Returns:
        dict with keys 'mae', 'rmse', 'r2' each containing an array of length n_bootstrap
    """
    df = pd.read_csv(pred_file)

    # Identify actual and predicted columns
    actual_col = None
    for c in ['actual', 'Actual', 'actual_pct']:
        if c in df.columns:
            actual_col = c
            break
    pred_col = None
    for c in ['p50', 'p50_anchored', 'predicted', 'Predicted']:
        if c in df.columns:
            pred_col = c
            break

    if actual_col is None or pred_col is None:
        raise ValueError(
            f"Cannot find actual/pred columns in {pred_file.name}. "
            f"Available: {list(df.columns)}"
        )

    y_true = df[actual_col].dropna().values
    y_pred = df[pred_col].dropna().values
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    rng = np.random.default_rng(seed)
    mae_boot, rmse_boot, r2_boot = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mae_boot.append(np.mean(np.abs(yt - yp)))
        rmse_boot.append(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2_boot.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)

    return {
        'mae':  np.array(mae_boot),
        'rmse': np.array(rmse_boot),
        'r2':   np.array(r2_boot),
    }


def compute_budget_effect_sizes(
        budget_results_df: pd.DataFrame,
        parent_run_id: str,
        project_root: Path,
        n_bootstrap: int = 1000,
        seed: int = 42) -> pd.DataFrame:
    """
    For each budget perturbation condition, bootstrap resample the saved
    comparison_data.csv and compute Cohen's d vs the nominal (0%) condition.

    Reports:
      - Cohen's d (MAE)  : effect of budget perturbation on MAE
      - Cohen's d (R²)   : effect of budget perturbation on R²
      - Interpretation   : negligible / small / medium / large

    This directly responds to Reviewer 3's request for effect size reporting
    in the sensitivity analysis.

    Args:
        budget_results_df : DataFrame from test_budget_sensitivity()
        parent_run_id     : Parent run directory name
        project_root      : Project root Path
        n_bootstrap       : Bootstrap iterations per condition
        seed              : Base random seed

    Returns:
        DataFrame with per-condition effect sizes
    """
    print("\n" + "=" * 70)
    print("COHEN'S d EFFECT SIZE: BUDGET PERTURBATION vs NOMINAL")
    print("(Answers: does budget uncertainty materially affect forecast quality?)")
    print("=" * 70)

    runs_base = project_root / "outputs" / "runs" / parent_run_id

    # Collect bootstrap distributions keyed by perturbation_pct
    dist_by_pct: Dict[float, Dict[str, np.ndarray]] = {}

    anchor_rows = budget_results_df[budget_results_df.get('anchor_mode', 'with_anchor') != 'no_anchor'] \
        if 'anchor_mode' in budget_results_df.columns \
        else budget_results_df

    for _, row in anchor_rows.iterrows():
        pct = float(row.get('perturbation_pct', 0))
        run_note = str(row.get('run_note', ''))
        pred_file = runs_base / run_note / "comparison_data.csv"

        if not pred_file.exists():
            print(f"  [SKIP] comparison_data.csv not found for run: {run_note}")
            continue

        try:
            dist_by_pct[pct] = bootstrap_condition_from_csv(
                pred_file, n_bootstrap=n_bootstrap, seed=seed + int(abs(pct))
            )
            print(f"  Bootstrapped pct={pct:+.0f}%  (n={n_bootstrap}, file={pred_file.name})")
        except Exception as e:
            print(f"  [ERROR] pct={pct}: {e}")

    if 0.0 not in dist_by_pct:
        print("  WARNING: Nominal (0%) condition not found; cannot compute Cohen's d")
        return pd.DataFrame()

    nominal = dist_by_pct[0.0]
    rows = []

    for pct in sorted(dist_by_pct.keys()):
        dist = dist_by_pct[pct]

        d_mae  = cohens_d(dist['mae'],  nominal['mae'])
        d_r2   = cohens_d(dist['r2'],   nominal['r2'])

        rows.append({
            'perturbation_pct': pct,
            'mae_mean':         float(np.mean(dist['mae'])),
            'mae_ci95_lower':   float(np.percentile(dist['mae'], 2.5)),
            'mae_ci95_upper':   float(np.percentile(dist['mae'], 97.5)),
            'r2_mean':          float(np.mean(dist['r2'])),
            'r2_ci95_lower':    float(np.percentile(dist['r2'], 2.5)),
            'r2_ci95_upper':    float(np.percentile(dist['r2'], 97.5)),
            'cohens_d_mae':     round(d_mae, 4),
            'cohens_d_r2':      round(d_r2, 4),
            'effect_size_mae':  interpret_cohens_d(d_mae),
            'effect_size_r2':   interpret_cohens_d(d_r2),
        })

    df_out = pd.DataFrame(rows)

    print("\nEffect size summary (vs nominal budget):")
    print(f"{'Perturbation':>14}  {'MAE mean':>9}  {'d(MAE)':>8}  {'Interp':>12}  "
          f"{'R² mean':>8}  {'d(R²)':>7}  {'Interp'}")
    print("-" * 80)
    for _, r in df_out.iterrows():
        print(f"{r['perturbation_pct']:>+13.0f}%  "
              f"{r['mae_mean']:>9.3f}  "
              f"{r['cohens_d_mae']:>+8.3f}  "
              f"{r['effect_size_mae']:>12s}  "
              f"{r['r2_mean']:>8.4f}  "
              f"{r['cohens_d_r2']:>+7.3f}  "
              f"{r['effect_size_r2']}")

    print("\nInterpretation: |d| < 0.2 = negligible, < 0.5 = small, "
          "< 0.8 = medium, ≥ 0.8 = large  (Cohen 1988)")

    return df_out


# ==================== Utility Functions ====================

def run_case_study(budget: float,
                   run_note: str,
                   project_root: Path,
                   ensemble_strategy: str = "median",
                   exclude_folds: str = "",
                   cv_model_dir: str = None,
                   parent_run_id: str = None,
                   anchor_mode: str = "csv",
                   return_predictions: bool = False) -> Dict:
    """
    Run a single case study and return results.

    Args:
        project_root      : Resolved project root Path
        ensemble_strategy : "median", "mean", or "weighted"
        anchor_mode       : "csv" or "none"
        return_predictions: If True, include path to comparison_data.csv

    Returns:
        dict with performance metrics and optionally 'prediction_file'
    """
    data_root   = project_root / "input_csv_data"
    input_csv   = data_root / "model_project" / "Preview_case_input_for_LSTM.csv"
    actual_csv  = data_root / "real_project" / "Chengbei_24m_work.csv"
    anchor_csv  = actual_csv

    if cv_model_dir is None:
        cv_model_dir = str(project_root / "models" / "cv5_seeds5_stratified")

    if parent_run_id:
        out_dir = project_root / "outputs" / "runs" / parent_run_id / run_note
    else:
        out_dir = project_root / "outputs" / "runs" / run_note
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(project_root / "LSTM" / "Case_Study" / "Run_Case_Study.py"),
        "--input_csv",    str(input_csv),
        "--total_budget", str(budget),
        "--run_note",     run_note,
        "--anchor_mode",  anchor_mode,
        "--cv_model_dir", cv_model_dir,
        "--output_dir",   str(out_dir),
        "--actual_csv",   str(actual_csv),
    ]

    if anchor_mode == "csv":
        cmd.extend(["--anchor_csv", str(anchor_csv)])
    if exclude_folds:
        cmd.extend(["--exclude_folds", exclude_folds])

    print(f"\n{'='*70}")
    print(f"Running: {run_note}")
    print(f"Budget: {budget:,.0f}, Ensemble: {ensemble_strategy}, "
          f"Anchor: {anchor_mode}, Exclude: {exclude_folds}")
    print(f"{'='*70}")

    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='ignore')

    if proc.returncode != 0:
        print(f"ERROR: Case study failed (rc={proc.returncode})")
        print(proc.stderr[-500:])
        return None

    latest_dir = out_dir
    metrics_file = latest_dir / "metrics.json"

    if not metrics_file.exists():
        print(f"ERROR: metrics.json not found in {latest_dir}")
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)

    print(f"✓ Complete: MAE={metrics['mae']:.2f}%, R²={metrics['r2']:.4f}")

    result_dict = {
        'run_note':          run_note,
        'budget':            budget,
        'ensemble_strategy': ensemble_strategy,
        'exclude_folds':     exclude_folds,
        **metrics
    }

    if return_predictions:
        pred_file = latest_dir / "comparison_data.csv"
        if pred_file.exists():
            result_dict['prediction_file'] = str(pred_file)
        else:
            print(f"WARNING: comparison_data.csv not found in {latest_dir}")

    return result_dict


# ==================== Test 1: Budget Sensitivity ====================

def test_budget_sensitivity(base_budget: float,
                            project_root: Path,
                            cv_model_dir: str = None,
                            parent_run_id: str = None,
                            test_no_anchor: bool = False) -> pd.DataFrame:
    """
    Test how model performance changes with budget perturbations (±10%, ±20%).
    Saves comparison_data.csv per condition for downstream Cohen's d analysis.
    """
    print("\n" + "=" * 70)
    print("TEST 1: BUDGET SENSITIVITY ANALYSIS")
    print("=" * 70)

    results = []

    for perturbation in BUDGET_PERTURBATIONS:
        budget = base_budget * perturbation
        pct = (perturbation - 1) * 100
        run_note = f"budget_sens_p{pct:.0f}pct" if pct >= 0 else f"budget_sens_n{abs(pct):.0f}pct"

        metrics = run_case_study(
            budget=budget,
            run_note=run_note,
            project_root=project_root,
            ensemble_strategy="median",
            cv_model_dir=cv_model_dir,
            parent_run_id=parent_run_id,
            anchor_mode="csv",
            return_predictions=True,   # required for Cohen's d downstream
        )

        if metrics:
            metrics['perturbation_pct'] = pct
            metrics['anchor_mode']      = 'with_anchor'
            results.append(metrics)

        if test_no_anchor:
            metrics_na = run_case_study(
                budget=budget,
                run_note=f"{run_note}_no_anchor",
                project_root=project_root,
                ensemble_strategy="median",
                cv_model_dir=cv_model_dir,
                parent_run_id=parent_run_id,
                anchor_mode="none",
            )
            if metrics_na:
                metrics_na['perturbation_pct'] = pct
                metrics_na['anchor_mode']      = 'no_anchor'
                results.append(metrics_na)

    df = pd.DataFrame(results)
    output_base = project_root / "sensitivity_analysis"
    output_base.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_base / "budget_sensitivity_results.csv", index=False)
    print(f"\n✓ Results saved to: {output_base / 'budget_sensitivity_results.csv'}")
    return df


# ==================== Test 2: Ensemble Strategy ====================

def test_ensemble_strategies(base_budget: float,
                              project_root: Path,
                              cv_model_dir: str = None,
                              parent_run_id: str = None) -> pd.DataFrame:
    """Compare ensemble strategies: median / mean / weighted."""
    print("\n" + "=" * 70)
    print("TEST 2: ENSEMBLE STRATEGY COMPARISON")
    print("=" * 70)

    results = []

    for strategy in ENSEMBLE_STRATEGIES:
        metrics = run_case_study(
            budget=base_budget,
            run_note=f"ensemble_{strategy}",
            project_root=project_root,
            ensemble_strategy=strategy,
            cv_model_dir=cv_model_dir,
            parent_run_id=parent_run_id,
        )
        if metrics:
            metrics['strategy'] = strategy
            results.append(metrics)

    df = pd.DataFrame(results)
    output_base = project_root / "sensitivity_analysis"
    output_base.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_base / "ensemble_comparison_results.csv", index=False)
    print(f"\n✓ Results saved to: {output_base / 'ensemble_comparison_results.csv'}")
    return df


# ==================== Test 3: Bootstrap Stability ====================

def test_bootstrap_stability(base_budget: float,
                             project_root: Path,
                             cv_model_dir: str = None,
                             parent_run_id: str = None,
                             n_bootstrap: int = BOOTSTRAP_N_ITERATIONS,
                             seed: int = BOOTSTRAP_SEED) -> Dict:
    """
    Bootstrap stability analysis without model re-training.
    Added (R3): reports Cohen's d between the lower and upper halves of each
    bootstrap distribution as a measure of within-distribution effect size,
    and between-seed variability interpretation.
    """
    print("\n" + "=" * 70)
    print("TEST 3: BOOTSTRAP STABILITY ANALYSIS")
    print(f"Bootstrap iterations: {n_bootstrap}, Seed: {seed}")
    print("=" * 70)

    # Step 1: baseline prediction
    baseline_metrics = run_case_study(
        budget=base_budget,
        run_note="bootstrap_baseline",
        project_root=project_root,
        ensemble_strategy="median",
        cv_model_dir=cv_model_dir,
        parent_run_id=parent_run_id,
        return_predictions=True,
    )

    if not baseline_metrics or 'prediction_file' not in baseline_metrics:
        print("ERROR: Failed to get prediction file from baseline run")
        return None

    pred_file = Path(baseline_metrics['prediction_file'])
    if not pred_file.exists():
        print(f"ERROR: Prediction file not found: {pred_file}")
        return None

    df_pred = pd.read_csv(pred_file)
    y_true = df_pred["actual"].values

    if "p50_anchored" in df_pred.columns:
        y_pred = df_pred["p50_anchored"].values
    elif "p50" in df_pred.columns:
        y_pred = df_pred["p50"].values
    else:
        raise ValueError(f"No p50 column found. Available: {df_pred.columns.tolist()}")

    print(f"✓ Loaded {len(df_pred)} predictions")

    # Step 2: Bootstrap
    rng = np.random.default_rng(seed)
    boot = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
    n_samples = len(y_true)

    for i in range(n_bootstrap):
        idx     = rng.integers(0, n_samples, size=n_samples)
        yt, yp  = y_true[idx], y_pred[idx]
        mae     = np.mean(np.abs(yt - yp))
        rmse    = np.sqrt(np.mean((yt - yp) ** 2))
        ss_res  = np.sum((yt - yp) ** 2)
        ss_tot  = np.sum((yt - np.mean(yt)) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        mask    = yt > 0.1
        mape    = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100 if mask.any() else np.nan

        boot['mae'].append(mae)
        boot['rmse'].append(rmse)
        boot['r2'].append(r2)
        boot['mape'].append(mape)

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_bootstrap}")

    # Step 3: Summary statistics + Cohen's d (NEW R3)
    summary = {}
    print("\n" + "=" * 70)
    print("BOOTSTRAP STABILITY + COHEN'S d RESULTS")
    print("=" * 70)

    print("\nBaseline (single run with 5 folds):")
    for k in ['mae', 'rmse', 'r2', 'mape']:
        v = baseline_metrics.get(k)
        if v is not None:
            print(f"  {k.upper():5s}: {v:.4f}")

    for metric in ['mae', 'rmse', 'r2', 'mape']:
        vals = np.array(boot[metric])
        vals = vals[~np.isnan(vals)]

        # Split distribution: lower vs upper half by median
        # Cohen's d between halves quantifies internal spread (large d = skewed dist)
        lower_half = vals[vals <= np.median(vals)]
        upper_half = vals[vals >  np.median(vals)]
        d_halves   = cohens_d(upper_half, lower_half)

        summary[metric] = {
            'mean':          float(np.mean(vals)),
            'std':           float(np.std(vals, ddof=1)),
            'median':        float(np.median(vals)),
            'ci_lower_95':   float(np.percentile(vals, 2.5)),
            'ci_upper_95':   float(np.percentile(vals, 97.5)),
            'ci_lower_90':   float(np.percentile(vals, 5.0)),
            'ci_upper_90':   float(np.percentile(vals, 95.0)),
            'cv':            float(np.std(vals, ddof=1) / np.mean(vals) * 100)
                             if np.mean(vals) > 1e-9 else 0.0,
            # R3: effect size within distribution
            'cohens_d_halves':        round(d_halves, 4),
            'cohens_d_interpretation': interpret_cohens_d(d_halves),
        }

        print(f"\n{metric.upper()}:")
        print(f"  Mean ± SD  : {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")
        print(f"  95% CI     : [{summary[metric]['ci_lower_95']:.4f}, "
              f"{summary[metric]['ci_upper_95']:.4f}]")
        print(f"  CV         : {summary[metric]['cv']:.2f}%")
        print(f"  Cohen's d (upper vs lower half): "
              f"{d_halves:+.3f}  [{interpret_cohens_d(d_halves)}]")
        print(f"  → {'Symmetric distribution' if abs(d_halves) < 0.2 else 'Skewed distribution'}")

    # Stability rating
    mae_cv = summary['mae']['cv']
    stability = ("EXCELLENT" if mae_cv < 5 else
                 "GOOD"      if mae_cv < 10 else
                 "MODERATE"  if mae_cv < 15 else "POOR")
    print(f"\nMAE CV = {mae_cv:.2f}%  →  Stability: {stability}")

    summary['baseline'] = {k: baseline_metrics.get(k) for k in ['mae', 'rmse', 'r2', 'mape']}
    summary['bootstrap_config'] = {
        'n_iterations':    n_bootstrap,
        'seed':            seed,
        'n_samples':       n_samples,
        'stability_rating': stability,
    }

    output_base = project_root / "sensitivity_analysis"
    output_base.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(boot).to_csv(output_base / "bootstrap_distributions.csv", index=False)
    with open(output_base / "bootstrap_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Bootstrap distributions: {output_base / 'bootstrap_distributions.csv'}")
    print(f"✓ Summary (incl. Cohen's d): {output_base / 'bootstrap_summary.json'}")
    return summary


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="LSTM Sensitivity Analysis with Cohen's d effect sizes (R3 revision)"
    )
    parser.add_argument("--total_budget",      type=float, required=True,
                        help="Base project budget (CNY)")
    parser.add_argument("--project_root",      type=str,   default=None,
                        help="Project root directory (auto-detected if omitted)")
    parser.add_argument("--cv_model_dir",      type=str,   default=None,
                        help="CV model directory (default: <root>/models/cv5_seeds5_stratified)")
    parser.add_argument("--tests",             type=str,   default="all",
                        help="Comma-separated tests: all, budget, ensemble, bootstrap, effect_sizes")
    parser.add_argument("--bootstrap_iterations", type=int, default=BOOTSTRAP_N_ITERATIONS,
                        help=f"Bootstrap iterations (default: {BOOTSTRAP_N_ITERATIONS})")
    parser.add_argument("--bootstrap_seed",    type=int,   default=BOOTSTRAP_SEED,
                        help=f"Bootstrap seed (default: {BOOTSTRAP_SEED})")
    parser.add_argument("--test_no_anchor",    action='store_true',
                        help="Test budget sensitivity with/without anchor")
    parser.add_argument("--effect_size_n_boot", type=int,  default=1000,
                        help="Bootstrap iterations for Cohen's d effect sizes (default: 1000)")

    args = parser.parse_args()

    # Resolve project root (no hardcoded paths)
    project_root = resolve_project_root(args.project_root)
    output_base  = project_root / "sensitivity_analysis"
    output_base.mkdir(parents=True, exist_ok=True)

    cv_model_dir = args.cv_model_dir or str(
        project_root / "models" / "cv5_seeds5_stratified"
    )

    tests_to_run = [t.strip() for t in args.tests.lower().split(',')]
    run_all      = 'all' in tests_to_run

    parent_run_id = time.strftime("%Y%m%d_%H%M") + "_sensitivity_analysis"

    print("\n" + "=" * 70)
    print("LSTM SENSITIVITY ANALYSIS (R3 revision — Cohen's d)")
    print("=" * 70)
    print(f"Project Root   : {project_root}")
    print(f"Base Budget    : {args.total_budget:,.0f} CNY")
    print(f"Parent Run ID  : {parent_run_id}")
    print(f"Output Dir     : {output_base}")
    print(f"Tests          : {args.tests}")
    print(f"Bootstrap Seed : {args.bootstrap_seed}")
    print("=" * 70)

    # Save run config for reproducibility
    config = {
        'project_root':      str(project_root),
        'total_budget':      args.total_budget,
        'cv_model_dir':      cv_model_dir,
        'tests':             args.tests,
        'bootstrap_iterations': args.bootstrap_iterations,
        'bootstrap_seed':    args.bootstrap_seed,
        'effect_size_n_boot': args.effect_size_n_boot,
        'parent_run_id':     parent_run_id,
        'timestamp':         time.strftime("%Y-%m-%d %Human:%M:%S"),
    }
    with open(output_base / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Run config saved: {output_base / 'run_config.json'}")

    results       = {}
    budget_df     = None

    # Test 1: Budget Sensitivity
    if run_all or 'budget' in tests_to_run:
        budget_df = test_budget_sensitivity(
            args.total_budget, project_root, cv_model_dir,
            parent_run_id, test_no_anchor=args.test_no_anchor
        )
        results['budget'] = budget_df

    # Test 2: Ensemble Strategy
    if run_all or 'ensemble' in tests_to_run:
        results['ensemble'] = test_ensemble_strategies(
            args.total_budget, project_root, cv_model_dir, parent_run_id
        )

    # Test 3: Bootstrap Stability
    if run_all or 'bootstrap' in tests_to_run:
        results['bootstrap'] = test_bootstrap_stability(
            args.total_budget, project_root, cv_model_dir,
            parent_run_id,
            n_bootstrap=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )

    # Cohen's d Effect Sizes (NEW R3 — runs after budget test)
    if (run_all or 'effect_sizes' in tests_to_run or 'budget' in tests_to_run):
        if budget_df is not None and len(budget_df) > 0:
            es_df = compute_budget_effect_sizes(
                budget_df, parent_run_id, project_root,
                n_bootstrap=args.effect_size_n_boot,
                seed=args.bootstrap_seed,
            )
            if es_df is not None and len(es_df) > 0:
                es_df.to_csv(output_base / "budget_cohens_d.csv", index=False)
                print(f"\n✓ Cohen's d table saved: {output_base / 'budget_cohens_d.csv'}")
                results['effect_sizes'] = es_df
        else:
            print("\nINFO: Budget sensitivity results not available; "
                  "skipping Cohen's d computation.")

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All results saved to: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()