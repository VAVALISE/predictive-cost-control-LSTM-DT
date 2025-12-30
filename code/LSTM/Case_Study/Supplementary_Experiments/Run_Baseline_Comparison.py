"""
Run_Baseline_Comparison.py - Experiment2 + Experiment3
============================================
Experiment2：Stronger baseline comparison（EVM/CPI, ETS, Prophet）
Experiment3：trigger rule sensitivity（tolerance × duration grid）

Metrics：
- point forecast：MAE, R²
- Probabilisty forecast：pinball loss, interval score, coverage
- Sensitivity：trigger count, earliest detection, FA rate, stability
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Baseline_Models import (
    NaiveBaseline, ARIMAModel, EVMForecast, ETSModel, ProphetModel,
    calculate_point_metrics, calculate_probabilistic_metrics,
    run_baseline_comparison
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_lstm_from_case_study(case_dir: str, anchor: int, horizon: int):
    """
    Load LSTM quantiles (P10/P50/P90) from a case-study output folder.
    Preferred source: pre_progress.csv
    Returns: (p10, p50, p90) as numpy arrays (length ~= horizon)
    """
    # 1) prefer CSV if exists
    csv_path = os.path.join(case_dir, "pred_progress.csv")
    xlsx_path = os.path.join(case_dir, "pred_progress.xlsx")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ts_path = csv_path
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
        ts_path = xlsx_path
    else:
        raise FileNotFoundError(
            f"Neither pred_progress.csv nor pred_progress.xlsx found in: {case_dir}"
        )

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_p10 = pick("forecast_p10", "p10", "lstm_p10", "yhat_p10")
    c_p50 = pick("forecast_p50", "p50", "lstm_p50", "yhat_p50")
    c_p90 = pick("forecast_p90", "p90", "lstm_p90", "yhat_p90")

    if c_p50 is None:
        raise KeyError(f"Cannot find P50 column in {ts_path}. Columns={list(df.columns)}")

    c_m = pick("month", "m", "time", "t")
    if c_m is not None:
        m = df[c_m].astype(str).str.extract(r"(\d+)")[0].astype(float)
        mask = m > anchor
        sub = df.loc[mask].copy()
    else:
        sub = df.tail(horizon).copy()

    p50 = pd.to_numeric(sub[c_p50], errors="coerce").to_numpy()
    p10 = pd.to_numeric(sub[c_p10], errors="coerce").to_numpy() if c_p10 else None
    p90 = pd.to_numeric(sub[c_p90], errors="coerce").to_numpy() if c_p90 else None

    L = min(horizon, len(p50))
    p50 = p50[:L]
    if p10 is not None: p10 = p10[:L]
    if p90 is not None: p90 = p90[:L]

    p50 = pd.Series(p50).interpolate(limit_direction="both").to_numpy()
    p50 = np.maximum.accumulate(np.clip(p50, 0, 100))

    if p10 is None: p10 = p50 * 0.9
    if p90 is None: p90 = p50 * 1.1

    p10 = np.clip(p10, 0, p50)
    p90 = np.clip(p90, p50, 150)

    return p10, p50, p90


# ===========================================================================================
# PART 1: Experiment2 - baselineratio
# ===========================================================================================

class BaselineComparisonExperiment:
    """
    Experiment 2: Stronger Baseline Control
    Objective: To address the criticism that "naïve baselines are too weak"
    Baselines:
    A. EVM/CPI-based forecast (engineering standard)
    B. ETS/ARIMA/Prophet (time series standard)
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.exp_dir = os.path.join(output_dir, "baseline_comparison")
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"✓ Experiment 2 output: {self.exp_dir}")

    def load_data(self, data_path: str) -> dict:
        df = pd.read_csv(data_path)
        df = df.dropna(how='all')

        if 'cumulative_share_pct' in df.columns:
            s = df['cumulative_share_pct']
            s = s.astype(str).str.replace('%', '', regex=False)
            s = pd.to_numeric(s, errors='coerce')
            cost_share = s.values
        elif 'cumulative_cost' in df.columns:
            s = pd.to_numeric(df['cumulative_cost'], errors='coerce')
            last = s.dropna().iloc[-1]
            cost_share = (s / last) * 100
        else:
            raise ValueError("No cost column found")

        cost_share = pd.Series(cost_share).interpolate(limit_direction='both').ffill().bfill().values
        cost_share = np.maximum.accumulate(cost_share)

        # ------------------------------------------------------------
        # enforce 24-month case length (to match 24-month case study)
        # Some files include an extra point (e.g., Month 0 or Month 25).
        target_n = 24
        if len(cost_share) == 25:
            # if first point looks like Month 0 (near 0), drop it; otherwise drop the last
            if cost_share[0] <= 1.0:
                cost_share = cost_share[1:]
            else:
                cost_share = cost_share[:target_n]
        elif len(cost_share) > target_n:
            cost_share = cost_share[:target_n]

        anchor = 12

        return {
            'train': cost_share[:anchor],
            'test': cost_share[anchor:],
            'baseline': np.linspace(cost_share[0], 100, anchor),
            'full_data': cost_share
        }


    def run_experiment(self, train: np.ndarray, test: np.ndarray,
                        baseline: np.ndarray,
                        lstm_p50: np.ndarray = None,
                        lstm_p10: np.ndarray = None,
                        lstm_p90: np.ndarray = None) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: BASELINE COMPARISON")
        print("=" * 70)
        print(f"Training: {len(train)} months")
        print(f"Testing: {len(test)} months")

        print("\n[Running Baselines]")
        results_df = run_baseline_comparison(
            train, test, baseline,
            lstm_p10, lstm_p50, lstm_p90
        )

        results_df.to_csv(os.path.join(self.exp_dir, 'comparison_table.csv'), index=False)

        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        display_df = results_df.copy()
        for col in ['MAE', 'RMSE', 'MAPE', 'Pinball_Avg', 'IntervalScore_80', 'MeanWidth_80']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        for col in ['R2', 'Coverage_80']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)

        print(display_df.to_string(index=False))

        return results_df

    def plot_comparison_curves(self, train: np.ndarray, test: np.ndarray,
                                baseline: np.ndarray, lstm_p50: np.ndarray = None):
        """
        Caption template:
        "Comparison of cumulative cost forecasts. Training period (M1-12) shown in gray.
        Test period (M13-24) compares actual values against LSTM, EVM/CPI, and ETS predictions."
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        train_months = np.arange(1, len(train) + 1)
        test_months = np.arange(len(train) + 1, len(train) + len(test) + 1)

        ax.plot(train_months, train, 'o-', color='gray', lw=2, alpha=0.7, label='Training Data')
        ax.plot(test_months, test, 'ko-', lw=2.5, ms=8, label='Actual', zorder=5)

        if lstm_p50 is not None:
            ax.plot(test_months[:len(lstm_p50)], lstm_p50, 'b-', lw=2.5, marker='s', ms=6, label='LSTM (Ours)')

        # EVM/CPI
        evm = EVMForecast('cpi_trend')
        pred_evm = evm.predict(train, baseline, len(test))
        pred_evm = np.maximum.accumulate(np.clip(pred_evm, 0, 100))
        ax.plot(test_months[:len(pred_evm)], pred_evm, 'r--', lw=2, marker='^', ms=5, label='EVM/CPI', alpha=0.8)

        # ETS
        ets = ETSModel(damped=True)
        pred_ets = ets.predict(train, len(test))
        pred_ets = np.maximum.accumulate(np.clip(pred_ets, 0, 100))
        ax.plot(test_months[:len(pred_ets)], pred_ets, 'g-.', lw=2, marker='D', ms=5, label='ETS Holt', alpha=0.8)

        ax.axvline(len(train) + 0.5, color='red', ls=':', lw=2, alpha=0.5)
        ax.text(len(train) + 0.7, 20, 'Forecast Start', color='red', fontsize=10, rotation=90)

        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Cumulative Cost Share (%)', fontsize=12)
        ax.set_title('Baseline Comparison: LSTM vs EVM/CPI vs ETS', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(train) + len(test) + 1)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, 'comparison_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_metrics_bar(self, results_df: pd.DataFrame):
        """
        Caption template:
        "Performance comparison of forecasting methods. (a) Point prediction metrics (MAE, RMSE).
        (b) Probabilistic prediction metrics (Pinball loss, Interval score).
        Lower values indicate better performance."
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        methods = results_df['Method'].values
        x = np.arange(len(methods))

        # (a) oint prediction metrics
        ax1 = axes[0]
        width = 0.35
        ax1.bar(x - width/2, results_df['MAE'], width, label='MAE (pp)', color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, results_df['RMSE'], width, label='RMSE (pp)', color='#e74c3c', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Error (percentage points)', fontsize=11)
        ax1.set_title('(a) Point Prediction Metrics', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # (b) Probabilistic prediction metrics
        ax2 = axes[1]
        if 'Pinball_Avg' in results_df.columns:
            ax2.bar(x - width/2, results_df['Pinball_Avg'], width, label='Pinball Avg', color='#2ecc71', alpha=0.8)
        if 'IntervalScore_80' in results_df.columns:
            ax2.bar(x + width/2, results_df['IntervalScore_80'], width, label='Interval Score', color='#9b59b6', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('(b) Probabilistic Prediction Metrics', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")



# ===========================================================================================
# PART 2: Experiment 3 - Sensitivity Analysis
# ===========================================================================================

class SensitivityAnalysisExperiment:
    """
    Parametric mesh:
    - tolerance: ±1%, ±2%, ±3%
    - duration: 2, 3, 4 months

    Metrics：
    - Trigger count
    - Earliest detection month
    - False alarm rate
    - Stability
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.exp_dir = os.path.join(output_dir, "tolerance_duration")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.tolerance_grid = [1, 2, 3, 5]
        self.duration_grid = [1, 2, 3, 4]

        print(f"Experiment 3 output: {self.exp_dir}")

    def detect_with_params(self, deviation: np.ndarray,
                           tolerance: float, duration: int) -> list:
        n = len(deviation)
        triggered = []
        i = 0

        while i < n:
            if abs(deviation[i]) > tolerance:
                start = i
                length = 1
                j = i + 1
                while j < n and abs(deviation[j]) > tolerance:
                    length += 1
                    j += 1

                if length >= duration:
                    triggered.extend(range(start + 1, start + length + 1))
                i = j
            else:
                i += 1

        return sorted(set(triggered))


    def calculate_metrics(self, triggered: list, drift_mask: np.ndarray,
                          onset_month: int, total: int) -> dict:
        trigger_set = set(triggered)

        tp = fp = fn = tn = 0
        for m in range(1, total + 1):
            idx = m - 1
            is_triggered = m in trigger_set
            has_drift = drift_mask[idx] if idx < len(drift_mask) else False

            if is_triggered and has_drift:
                tp += 1
            elif is_triggered and not has_drift:
                fp += 1
            elif not is_triggered and has_drift:
                fn += 1
            else:
                tn += 1

        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        first = min(triggered) if triggered else None
        lead = onset_month - first if first else None

        return {
            'trigger_count': len(triggered),
            'first_trigger': first,
            'lead_time': lead,
            'false_alarm_rate': fa_rate,
            'tp': tp, 'fp': fp, 'fn': fn
        }

    def run_grid_analysis(self, deviation: np.ndarray, drift_mask: np.ndarray,
                          onset_month: int) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: SENSITIVITY ANALYSIS")
        print("=" * 70)

        results = []
        total = len(deviation)

        for tol in self.tolerance_grid:
            for dur in self.duration_grid:
                triggered = self.detect_with_params(deviation, tol, dur)
                metrics = self.calculate_metrics(triggered, drift_mask, onset_month, total)

                results.append({
                    'Tolerance (%)': tol,
                    'Duration (months)': dur,
                    **metrics
                })

                print(f"  tol=±{tol}%, dur={dur}: triggers={metrics['trigger_count']}, "
                      f"FA={metrics['false_alarm_rate']:.1%}")

        return pd.DataFrame(results)

    def calculate_stability(self, results_df: pd.DataFrame) -> dict:
        """Compute stability statistics across the parameter grid."""
        stability = {}

        for metric in ['trigger_count', 'lead_time', 'false_alarm_rate']:
            values = results_df[metric].dropna().values.astype(float)
            if len(values) > 1:
                mu = float(np.mean(values))
                sigma = float(np.std(values))
                stability[metric] = {
                    'mean': mu,
                    'std': sigma,
                    'range': float(np.max(values) - np.min(values)),
                    'cv': float(sigma / (abs(mu) + 1e-9))
                }

        return stability


    def find_pareto_optimal(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Find Pareto-optimal configurations.

        Objectives:
          - maximise lead_time (earlier detection is better)
          - minimise false_alarm_rate
        Notes:
          - rows with trigger_count == 0 are excluded (no actionable detection)
          - NaN lead_time is treated as worst (-inf) to avoid NaN becoming 'always Pareto'
        """

        def _lt(x):
            return float(x) if pd.notna(x) else float("-inf")

        # Exclude configs that never trigger (lead_time is undefined / not actionable)
        if 'trigger_count' in results_df.columns:
            valid = results_df[results_df['trigger_count'] > 0].copy()
        else:
            valid = results_df.copy()

        if valid.empty:
            return valid  # return empty DF (caller can handle)

        pareto_idx = []
        for idx, row in valid.iterrows():
            dominated = False
            lt_r = _lt(row.get('lead_time', np.nan))
            fa_r = float(row.get('false_alarm_rate', np.nan))

            for jdx, other in valid.iterrows():
                if jdx == idx:
                    continue

                lt_o = _lt(other.get('lead_time', np.nan))
                fa_o = float(other.get('false_alarm_rate', np.nan))

                lt_better = lt_o >= lt_r
                fa_better = fa_o <= fa_r
                strictly_better = (lt_o > lt_r) or (fa_o < fa_r)

                if lt_better and fa_better and strictly_better:
                    dominated = True
                    break

            if not dominated:
                pareto_idx.append(idx)

        return valid.loc[pareto_idx]

    def plot_heatmaps(self, results_df: pd.DataFrame):
        """
        Drawing a sensitivity heatmap

        Caption template:
        "Sensitivity of governance triggers to parameter settings.
        (a) Trigger count. (b) False alarm rate. (c) Detection lead time.
        Darker colors indicate higher values."
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Trigger Rule Sensitivity: Tolerance × Duration', fontsize=14, fontweight='bold')

        metrics = [
            ('trigger_count', 'Trigger Count', 'Blues'),
            ('false_alarm_rate', 'False Alarm Rate', 'Reds'),
            ('lead_time', 'Lead Time (months)', 'Greens')
        ]

        for ax, (metric, title, cmap) in zip(axes, metrics):
            pivot = results_df.pivot(
                index='Duration (months)',
                columns='Tolerance (%)',
                values=metric
            )

            fmt = '.0f' if metric in ['trigger_count', 'lead_time'] else '.2%'
            if metric == 'false_alarm_rate':
                pivot = pivot * 100
                fmt = '.1f'

            sns.heatmap(pivot, annot=True, fmt=fmt.replace('%', ''), cmap=cmap, ax=ax)
            ax.set_title(f'({chr(97 + metrics.index((metric, title, cmap)))}) {title}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, 'sensitivity_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def plot_tradeoff(self, results_df: pd.DataFrame, pareto_df: pd.DataFrame):
        """
        Caption template:
        "Trade-off between detection lead time and false alarm rate.
        Star markers indicate Pareto-optimal configurations.
        The recommended configuration (tolerance=±2%, duration=2)
        balances early detection with low false alarms."
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        for dur in self.duration_grid:
            subset = results_df[results_df['Duration (months)'] == dur]
            lt = subset['lead_time'].fillna(0).values
            fa = subset['false_alarm_rate'].values * 100
            ax.plot(fa, lt, 'o-', ms=10, label=f'Duration={dur}')

        # Pareto
        if len(pareto_df) > 0:
            lt = pareto_df['lead_time'].fillna(0).values
            fa = pareto_df['false_alarm_rate'].values * 100
            ax.scatter(fa, lt, s=300, c='red', marker='*', zorder=5, label='Pareto Optimal')

        ax.set_xlabel('False Alarm Rate (%)', fontsize=12)
        ax.set_ylabel('Detection Lead Time (months)', fontsize=12)
        ax.set_title('Trade-off: Early Detection vs False Alarms', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, 'tradeoff_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def generate_recommendation(self,
                                results_df: pd.DataFrame,
                                pareto_df: pd.DataFrame,
                                stability: dict) -> str:
        # --- derive tested grids dynamically (avoid hard-coded text) ---
        tols = sorted(results_df['Tolerance (%)'].dropna().unique().tolist())
        durs = sorted(results_df['Duration (months)'].dropna().unique().tolist())

        # keep only configs that actually trigger at least once (lead_time meaningful)
        valid = results_df[(results_df['trigger_count'] > 0)].copy()

        # --- Recommendation A: Early-warning (balanced sensitivity vs alert burden) ---
        # Prefer FA <= 30% (engineering-friendly), then maximise lead_time.
        early = valid[(valid['false_alarm_rate'] <= 0.30) &
                      (valid['Duration (months)'] >= 2)].copy()

        if len(early) > 0:
            early_best = early.sort_values(['lead_time', 'false_alarm_rate'],
                                           ascending=[False, True]).iloc[0]
        else:
            # fall back to Pareto front (largest lead_time)
            early_best = pareto_df.sort_values(['lead_time', 'false_alarm_rate'],
                                               ascending=[False, True]).iloc[0]

        # --- Recommendation B: Management-review (conservative, low false-alarm) ---
        # Prefer tol = 5% if available; otherwise lowest FA in valid set.
        mgmt = valid.copy()
        if 5 in tols:
            mgmt = mgmt[mgmt['Tolerance (%)'] == 5].copy()
        if len(mgmt) > 0:
            # prefer a non-trivial persistence window (dur>=2)
            mgmt2 = mgmt[mgmt['Duration (months)'] >= 2].copy()
            pick_df = mgmt2 if len(mgmt2) > 0 else mgmt

            mgmt_best = pick_df.sort_values(['false_alarm_rate', 'trigger_count', 'Duration (months)'],
                                            ascending=[True, True, True]).iloc[0]

        else:
            mgmt_best = valid.sort_values(['false_alarm_rate', 'trigger_count'],
                                          ascending=[True, True]).iloc[0]

        default = results_df[(results_df['Tolerance (%)'] == 2) &
                             (results_df['Duration (months)'] == 2)]
        has_default = len(default) > 0
        if has_default:
            default_row = default.iloc[0]

        lead_cv = stability.get('lead_time', {}).get('cv', 0)
        fa_range = stability.get('false_alarm_rate', {}).get('range', 0) * 100

        conclusion = "Stable" if lead_cv < 0.30 else "Moderately sensitive"

        rec = f"""
    PARAMETER RECOMMENDATION

    1. GRID SUMMARY
       - Tested tolerance levels (%): {tols}
       - Tested duration windows (months): {durs}
       - Total configurations: {len(results_df)}
       - Pareto-optimal configurations: {len(pareto_df)}

    2. STABILITY CHECK
       - Lead-time coefficient of variation (CV): {lead_cv:.2f}
       - False-alarm rate range across grid: {fa_range:.1f}%
       - Overall conclusion: {conclusion}

    3. RECOMMENDED THRESHOLDS (two-tier governance)
       (A) Early-warning band (more sensitive; higher alert burden is expected)
           → tol=±{early_best['Tolerance (%)']}%, dur={int(early_best['Duration (months)'])} months
           → Lead={early_best['lead_time']}, FA={early_best['false_alarm_rate']:.1%}

       (B) Management-review band (conservative; prioritises low false-alarm)
           → tol=±{mgmt_best['Tolerance (%)']}%, dur={int(mgmt_best['Duration (months)'])} months
           → Lead={mgmt_best['lead_time']}, FA={mgmt_best['false_alarm_rate']:.1%}
    """

        if has_default:
            rec += f"""
       (Reference) Default setting used in the paper
           → tol=±2%, dur=2 months
           → Lead={default_row['lead_time']}, FA={default_row['false_alarm_rate']:.1%}
    """

        rec += "\n4. PARETO-OPTIMAL CONFIGURATIONS\n"
        for _, row in pareto_df.iterrows():
            rec += (
                f"   • tol=±{row['Tolerance (%)']}%, dur={int(row['Duration (months)'])} months: "
                f"Lead={row['lead_time']}, FA={row['false_alarm_rate']:.1%}\n"
            )

        return rec


# ===========================================================================================
# PART 3: main function
# ===========================================================================================

def run_experiment2(data_path: str, output_dir: str, case_dir: str = None):
    exp = BaselineComparisonExperiment(output_dir)
    data = exp.load_data(data_path)

    train = data['train']
    test = data['test']
    baseline = data['baseline']

    lstm_p10 = lstm_p50 = lstm_p90 = None

    print(f"[LSTM import] case_dir = {case_dir}")
    if case_dir:
        try:
            lstm_p10, lstm_p50, lstm_p90 = load_lstm_from_case_study(
                case_dir, anchor=len(train), horizon=len(test)
            )
            print(f"Loaded LSTM: len={len(lstm_p50)}, p50 head={lstm_p50[:3]}")
        except Exception as e:
            print(f"LSTM import failed: {e}")

    results_df = exp.run_experiment(
        train, test, baseline,
        lstm_p50=lstm_p50, lstm_p10=lstm_p10, lstm_p90=lstm_p90
    )

    exp.plot_comparison_curves(train, test, baseline, lstm_p50=lstm_p50)
    exp.plot_metrics_bar(results_df)
    return results_df



def run_experiment3(forecast: np.ndarray, baseline: np.ndarray,
                     drift_mask: np.ndarray, onset_month: int,
                     output_dir: str):

    exp = SensitivityAnalysisExperiment(output_dir)

    deviation = np.where(baseline > 0, (forecast - baseline) / baseline * 100, 0)

    results_df = exp.run_grid_analysis(deviation, drift_mask, onset_month)
    results_df.to_csv(os.path.join(exp.exp_dir, 'sensitivity_results.csv'), index=False)

    pareto_df = exp.find_pareto_optimal(results_df)
    pareto_df.to_csv(os.path.join(exp.exp_dir, 'pareto_optimal.csv'), index=False)

    stability = exp.calculate_stability(results_df)

    exp.plot_heatmaps(results_df)
    exp.plot_tradeoff(results_df, pareto_df)

    rec = exp.generate_recommendation(results_df, pareto_df, stability)
    rec = rec.replace("•", "-")
    with open(os.path.join(exp.exp_dir, 'recommendation.txt'), 'w', encoding='utf-8') as f:
        f.write(rec)
    print(rec)

    return results_df, pareto_df


def main():
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    CASE_DIR = BASE_DIR / "outputs" / "case_study_chengbei_v5_20251215_154436"
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_DIR = os.path.join(OUTPUT_DIR, f"run_baseline_comparison_{run_ts}")
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"✓ Output root: {RUN_DIR}")
    DATA_PATH = os.path.join(BASE_DIR, "input_csv data", "real_project", "Chengbei_24m_work.csv")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['2', '3', 'both'], default='both')
    parser.add_argument('--case_dir', default=None,
                        help='Case study output folder containing pre_progress.csv')
    args = parser.parse_args()

    if args.experiment in ['2', 'both']:
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT 2: BASELINE COMPARISON")
        print("=" * 70)
        if os.path.exists(DATA_PATH):
            run_experiment2(DATA_PATH, RUN_DIR, case_dir=CASE_DIR)
        else:
            print(f"Data file not found: {DATA_PATH}")

            n = 24
            t = np.arange(1, n + 1)
            data = 100 / (1 + np.exp(-0.3 * (t - 12)))
            data = data / data[-1] * 100 + np.random.normal(0, 2, n)

            exp = BaselineComparisonExperiment(RUN_DIR)
            results = exp.run_experiment(data[:12], data[12:], np.linspace(data[0], 100, 12))
            exp.plot_metrics_bar(results)

    if args.experiment in ['3', 'both']:
        print("\n" + "=" * 70)
        print("RUNNING EXPERIMENT 3: SENSITIVITY ANALYSIS")
        print("=" * 70)

        n = 24
        t = np.arange(1, n + 1)


        baseline = 100 / (1 + np.exp(-0.25 * (t - 12)))
        baseline = baseline / baseline[-1] * 100
        baseline = np.maximum.accumulate(baseline)

        rng = np.random.default_rng(7)

        forecast = baseline + rng.normal(0, 0.6, n)

        forecast[5:7] += baseline[5:7] * 0.025

        for i in range(10, n):
            ramp = (i - 10) / (n - 10)  # 0..1
            forecast[i] += baseline[i] * (0.08 * ramp)

        dev_pct = (forecast - baseline) / np.maximum(baseline, 1e-6) * 100

        drift_mask = dev_pct > 5.0
        onset_month = int(np.argmax(drift_mask)) + 1

        run_experiment3(forecast, baseline, drift_mask, onset_month, RUN_DIR)


if __name__ == "__main__":
    main()