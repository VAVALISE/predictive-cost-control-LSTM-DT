"""
Core design：
- Full-EW: forecast OR DT either exceeds threshold → early warning（earlier than single source）
- Full-MR: forecast AND DT both confirm → management escalation（fewer false alarms than single source）

ThusFulltruly demonstrates"fusion"value：
- Full-EW ≠ Forecast-only
- Full-MR ≠ DT-only

Scene design：
- A: Sustained overrun
- B: Cost-leading
- C: Progress-lag
- D1: DT transient spike
- D2: Forecast transient spike
- E: Sustained control
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ===========================================================================================
# PART 1: drift scenario definition
# ===========================================================================================

DRIFT_SCENARIOS = {
    # ========== actual drift scenario  ==========

    "A1_overrun_mild": {
        "name": "Sustained Overrun (mild)",
        "type": "sustained_overrun",
        "drift_rate_pp": 3.0,
        "onset_month": 14,
        "forecast_lead": 2,
        "is_true_drift": True,
    },
    "A2_overrun_moderate": {
        "name": "Sustained Overrun (moderate)",
        "type": "sustained_overrun",
        "drift_rate_pp": 6.0,
        "onset_month": 14,
        "forecast_lead": 2,
        "is_true_drift": True,
    },

    "B_cost_leading": {
        "name": "Cost-Leading Drift",
        "type": "cost_leading",
        "drift_rate_pp": 6.0,
        "onset_month": 14,
        "forecast_lead": 3,
        "dt_lag": 2,
        "is_true_drift": True,
    },

    "C_progress_lag": {
        "name": "Progress-Lag Drift",
        "type": "progress_lag",
        "lag_rate_pp": 6.0,
        "onset_month": 14,
        "is_true_drift": True,
    },

    "E_sustained_control": {
        "name": "Sustained (D1/D2 control)",
        "type": "sustained_overrun",
        "drift_rate_pp": 6.0,
        "onset_month": 15,
        "forecast_lead": 2,
        "is_true_drift": True,
    },

    # ========== Negative Control ==========

    "D1_spike_dt_only": {
        "name": "Spike in DT only",
        "type": "spike_dt_only",
        "spike_months": [15, 16, 17],
        "spike_magnitude_pp": 12.0,
        "is_true_drift": False,
    },

    "D2_spike_forecast_only": {
        "name": "Spike in Forecast only",
        "type": "spike_forecast_only",
        "spike_months": [15, 16, 17],
        "spike_magnitude_pp": 12.0,
        "is_true_drift": False,
    },
}


# ===========================================================================================
# PART 2: Data generation and drift injection
# ===========================================================================================

def generate_baseline_data(total_months: int = 24, seed: int = 42) -> Dict:

    np.random.seed(seed)
    t = np.arange(1, total_months + 1)

    baseline = 100 / (1 + np.exp(-0.4 * (t - total_months/2)))
    baseline = baseline / baseline[-1] * 100

    noise_std = 0.3
    dt_original = baseline + np.random.normal(0, noise_std, total_months)
    dt_original = np.maximum.accumulate(np.clip(dt_original, 0, 100))

    forecast_p50 = baseline + np.random.normal(0, noise_std, total_months)
    forecast_p50 = np.maximum.accumulate(np.clip(forecast_p50, 0, 100))

    return {
        'baseline': baseline,
        'dt_original': dt_original,
        'forecast_p50': forecast_p50,
        'total_months': total_months
    }


def inject_drift(baseline: np.ndarray,
                 dt_original: np.ndarray,
                 forecast_p50: np.ndarray,
                 scenario_key: str,
                 noise_std: float = 0.5) -> Dict:

    scenario = DRIFT_SCENARIOS[scenario_key]
    n = len(baseline)

    drifted_dt = dt_original.copy()
    drifted_fc = forecast_p50.copy()

    is_true_drift = scenario.get('is_true_drift', True)
    drift_mask = np.zeros(n, dtype=bool)

    onset = scenario.get('onset_month', 14)
    onset_idx = onset - 1

    scenario_type = scenario['type']

    if scenario_type == 'sustained_overrun':
        rate = scenario['drift_rate_pp']
        lead = scenario.get('forecast_lead', 2)


        fc_start = max(0, onset_idx - lead)
        for i in range(fc_start, n):
            months_since = i - fc_start + 1
            drifted_fc[i] = baseline[i] + rate * months_since

        for i in range(onset_idx, n):
            months_since = i - onset_idx + 1
            drifted_dt[i] = baseline[i] + rate * months_since

        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif scenario_type == 'cost_leading':
        rate = scenario['drift_rate_pp']
        fc_lead = scenario.get('forecast_lead', 3)
        dt_lag = scenario.get('dt_lag', 2)

        fc_start = max(0, onset_idx - fc_lead)
        for i in range(fc_start, n):
            months_since = i - fc_start + 1
            drifted_fc[i] = baseline[i] + rate * months_since

        dt_start = onset_idx + dt_lag
        for i in range(dt_start, n):
            months_since = i - dt_start + 1
            drifted_dt[i] = baseline[i] + rate * months_since

        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif scenario_type == 'progress_lag':
        rate = scenario['lag_rate_pp']

        for i in range(onset_idx, n):
            months_since = i - onset_idx + 1
            drifted_dt[i] = baseline[i] - rate * months_since

        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif scenario_type == 'spike_dt_only':

        spike_months = scenario.get('spike_months', [15, 16, 17])
        spike_mag = scenario['spike_magnitude_pp']

        for m in spike_months:
            idx = m - 1
            if idx < n:
                drifted_dt[idx] = baseline[idx] + spike_mag


    elif scenario_type == 'spike_forecast_only':

        spike_months = scenario.get('spike_months', [15, 16, 17])
        spike_mag = scenario['spike_magnitude_pp']

        for m in spike_months:
            idx = m - 1
            if idx < n:
                drifted_fc[idx] = baseline[idx] + spike_mag


    drifted_dt += np.random.normal(0, noise_std, n)
    drifted_fc += np.random.normal(0, noise_std, n)

    drifted_dt = np.maximum.accumulate(np.clip(drifted_dt, 0, 150))
    drifted_fc = np.maximum.accumulate(np.clip(drifted_fc, 0, 150))

    return {
        'baseline': baseline,
        'dt_original': dt_original,
        'drifted_dt': drifted_dt,
        'forecast_p50': drifted_fc,
        'drift_mask': drift_mask,
        'onset_month': onset,
        'is_true_drift': is_true_drift,
        'scenario_key': scenario_key,
        'scenario_info': scenario
    }


# ===========================================================================================
# PART 3: Detector - Core Improvements
# ===========================================================================================

class SimpleDetector:
    def __init__(self, tolerance: float = 5.0, duration: int = 2):
        self.tolerance = tolerance
        self.duration = duration

    def _find_sustained_abs(self, deviation: np.ndarray) -> list[tuple[int,int]]:
        n = len(deviation)
        intervals = []
        i = 0
        while i < n:
            if abs(deviation[i]) > self.tolerance:
                start = i
                j = i + 1
                while j < n and abs(deviation[j]) > self.tolerance:
                    j += 1
                if (j - start) >= self.duration:
                    intervals.append((start, j - 1))
                i = j
            else:
                i += 1
        return intervals

    def detect(self, data: np.ndarray, baseline: np.ndarray) -> list[int]:
        dev = data - baseline
        triggered = set()
        for s, e in self._find_sustained_abs(dev):
            triggered.update(range(s + 1, e + 2))
        return sorted(triggered)



class FusionGovernanceDetector:
    """
    fusion governance detector - truly demonstrates"fusion"value

    Full-EW (Early Warning): forecast OR DT either exceeds thresholdcontinuousdur → alarm
        → earlier than single source

    Full-MR (Management Review): forecast AND DT All confirmed within window → improvement
        → fewer false alarms than single source
    """

    def __init__(self, tolerance: float = 5.0, duration: int = 2, mr_window: int = 3):
        self.tolerance = tolerance
        self.duration = duration
        self.mr_window = mr_window

        # DT-lag safeguard: Stricter than EW/MR, to avoid false triggering due to short-term fluctuations.
        self.dt_lag_mr_duration = max(duration + 1, 3)


    def _find_sustained_above(self, deviation: np.ndarray, threshold: float, min_len: int) -> List[Tuple[int, int]]:
        "Identify intervals where the deviation consistently exceeds the threshold."
        n = len(deviation)
        intervals = []
        i = 0
        while i < n:
            if deviation[i] > threshold:
                start = i
                j = i + 1
                while j < n and deviation[j] > threshold:
                    j += 1
                if j - start >= min_len:
                    intervals.append((start, j - 1))
                i = j
            else:
                i += 1
        return intervals


    def _find_sustained_below(self, deviation: np.ndarray, threshold: float, min_len: int) -> List[Tuple[int, int]]:

        n = len(deviation)
        intervals = []
        i = 0
        while i < n:
            if deviation[i] < threshold:
                start = i
                j = i + 1
                while j < n and deviation[j] < threshold:
                    j += 1
                if j - start >= min_len:
                    intervals.append((start, j - 1))
                i = j
            else:
                i += 1
        return intervals

    def detect_early_warning(self, forecast: np.ndarray, dt: np.ndarray,
                             baseline: np.ndarray) -> List[int]:
        """
        Full-EW: Forecast OR DT (Dual Thread) - Continuous out-of-band dur monitoring from either source within ±tolerance range → Warning
        Simultaneous coverage:
          - Overrun / leading: dev > +tol
          - Lag / underrun:   dev < -tol
        """
        dev_fc = forecast - baseline
        dev_dt = dt - baseline

        triggered = set()

        # ===== 1) Override threshold: dev > +tol =====
        fc_above = self._find_sustained_above(dev_fc, +self.tolerance, self.duration)
        dt_above = self._find_sustained_above(dev_dt, +self.tolerance, self.duration)

        for s, e in fc_above:
            triggered.update(range(s + 1, e + 2))
        for s, e in dt_above:
            triggered.update(range(s + 1, e + 2))

        # ===== 2) Downpass threshold: dev < -tol =====
        fc_below = self._find_sustained_below(dev_fc, -self.tolerance, self.duration)
        dt_below = self._find_sustained_below(dev_dt, -self.tolerance, self.duration)

        for s, e in fc_below:
            triggered.update(range(s + 1, e + 2))
        for s, e in dt_below:
            triggered.update(range(s + 1, e + 2))

        return sorted(triggered)


    def detect_management_review(self, forecast: np.ndarray, dt: np.ndarray,
                                 baseline: np.ndarray) -> List[int]:
        """
        Full-MR (Management Review): conservative escalation with AND confirmation.

        - Positive drift (overrun / cost-leading): require BOTH forecast and DT
          to exceed +tolerance for sustained durations (direction-consistent).
        - Negative drift (progress-lag): allow DT-only safeguard (sustained below -tolerance),
          because forecast may not capture lag-type deviations.
        """
        dev_fc = forecast - baseline
        dev_dt = dt - baseline
        n = len(baseline)

        triggered = set()

        # ---------------------------
        # (1) Positive drift: AND confirmation (direction-consistent)
        # ---------------------------
        # Build intervals using "above +tol" only (NOT abs)
        fc_pos = self._find_sustained_above(dev_fc, +self.tolerance, self.duration)
        dt_pos = self._find_sustained_above(dev_dt, +self.tolerance, self.duration)

        fc_neg = self._find_sustained_below(dev_fc, -self.tolerance, self.duration)
        dt_neg = self._find_sustained_below(dev_dt, -self.tolerance, self.duration)

        for fc_start, fc_end in fc_pos:
            # DT confirmation must occur near the forecast interval
            check_start = max(0, fc_start - self.mr_window)
            check_end = min(n - 1, fc_end + self.mr_window)

            for dt_start, dt_end in dt_pos:
                # Require DT interval intersects the expanded window
                if not (dt_start <= check_end and dt_end >= check_start):
                    continue

                # Prefer actual overlap as MR "confirmed" region
                overlap_start = max(fc_start, dt_start)
                overlap_end = min(fc_end, dt_end)

                if overlap_end >= overlap_start:
                    # confirmed region: months (1-indexed output)
                    triggered.update(range(overlap_start + 1, overlap_end + 2))
                else:
                    # no overlap but within window: trigger from DT confirmation interval
                    triggered.update(range(dt_start + 1, dt_end + 2))

        # ---------------------------
        # (2) Negative drift safeguard: progress-lag should still trigger MR
        # ---------------------------
        # Require longer persistence for lag if you want to be stricter:
        # lag_dur = max(self.duration, self.dt_lag_mr_duration)
        lag_dur = getattr(self, "dt_lag_mr_duration", self.duration)

        lag_intervals = self._find_sustained_below(dev_dt, -self.tolerance, lag_dur)
        for s, e in lag_intervals:
            triggered.update(range(s + 1, e + 2))

        return sorted(triggered)


    def detect_full(self, forecast: np.ndarray, dt: np.ndarray,
                    baseline: np.ndarray) -> Tuple[List[int], List[int]]:

        ew = self.detect_early_warning(forecast, dt, baseline)
        mr = self.detect_management_review(forecast, dt, baseline)
        return ew, mr


# ===========================================================================================
# PART 4: Metrics calculation
# ===========================================================================================

def calculate_metrics(triggered: List[int], onset_month: int,
                      drift_mask: np.ndarray, total_months: int,
                      is_true_drift: bool) -> Dict:

    trigger_set = set(triggered)

    tp = fp = fn = tn = 0

    for m in range(1, total_months + 1):
        idx = m - 1
        is_triggered = m in trigger_set

        if is_true_drift:
            has_drift = drift_mask[idx] if idx < len(drift_mask) else False
        else:
            has_drift = False

        if is_triggered and has_drift:
            tp += 1
        elif is_triggered and not has_drift:
            fp += 1
        elif not is_triggered and has_drift:
            fn += 1
        else:
            tn += 1

    # For negative control, F1 is meaningless (no TP is possible).
    if is_true_drift:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = None
        recall = None
        f1 = None

    fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    first_trigger = min(triggered) if triggered else None

    if is_true_drift and first_trigger is not None:
        lead_time = onset_month - first_trigger
    else:
        lead_time = None

    return {
        'first_trigger': first_trigger,
        'lead_time': lead_time,
        'onset_month': onset_month,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_alarm_rate': fa_rate,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total_triggers': len(triggered),
        'is_true_drift': is_true_drift
    }


# ===========================================================================================
# PART 5: Main Experiment Class
# ===========================================================================================

class GovernanceValueExperiment:

    def __init__(self, output_dir: str,
                 tolerance: float = 5.0,
                 duration: int = 2,
                 n_seeds: int = 30):

        self.output_dir = output_dir
        self.tolerance = tolerance
        self.duration = duration
        self.n_seeds = n_seeds

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(output_dir, f"run_ablation_study_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        print(f"✓ Experiment 1 output: {self.exp_dir}")
        print(f"  Tolerance: ±{tolerance}pp, Duration: {duration} months, Seeds: {n_seeds}")

    def run_single(self, data: Dict, scenario_key: str, seed: int = 42) -> Dict:

        np.random.seed(seed)

        drift_data = inject_drift(
            data['baseline'].copy(),
            data['dt_original'].copy(),
            data['forecast_p50'].copy(),
            scenario_key,
            noise_std=0.5
        )

        is_true_drift = drift_data['is_true_drift']
        onset = drift_data['onset_month']
        n = len(drift_data['baseline'])

        results = {}

        # 1. Full Framework (Fusion)
        full_detector = FusionGovernanceDetector(self.tolerance, self.duration)
        ew_triggers, mr_triggers = full_detector.detect_full(
            drift_data['forecast_p50'],
            drift_data['drifted_dt'],
            drift_data['baseline']
        )

        results['full_ew'] = calculate_metrics(
            ew_triggers, onset, drift_data['drift_mask'], n, is_true_drift
        )
        results['full_ew']['mode'] = 'Full-EW'

        results['full_mr'] = calculate_metrics(
            mr_triggers, onset, drift_data['drift_mask'], n, is_true_drift
        )
        results['full_mr']['mode'] = 'Full-MR'

        # 2. DT-only
        dt_detector = SimpleDetector(self.tolerance, self.duration)
        dt_triggers = dt_detector.detect(drift_data['drifted_dt'], drift_data['baseline'])
        results['dt_only'] = calculate_metrics(
            dt_triggers, onset, drift_data['drift_mask'], n, is_true_drift
        )
        results['dt_only']['mode'] = 'DT-only'

        # 3. Forecast-only
        fc_detector = SimpleDetector(self.tolerance, self.duration)
        fc_triggers = fc_detector.detect(drift_data['forecast_p50'], drift_data['baseline'])
        results['forecast_only'] = calculate_metrics(
            fc_triggers, onset, drift_data['drift_mask'], n, is_true_drift
        )
        results['forecast_only']['mode'] = 'Forecast-only'

        return {
            'scenario': scenario_key,
            'results': results,
            'drift_data': drift_data
        }

    def run_with_seeds(self, data: Dict, scenario_key: str) -> Dict:

        all_runs = []

        for seed in range(self.n_seeds):
            seed_data = generate_baseline_data(total_months=24, seed=seed * 100)
            result = self.run_single(seed_data, scenario_key, seed)
            all_runs.append(result)

        is_true_drift = DRIFT_SCENARIOS[scenario_key].get('is_true_drift', True)
        summary = {}

        for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
            lead_times = []
            f1s = []
            fa_rates = []
            precisions = []
            recalls = []

            for run in all_runs:
                m = run['results'][mode]
                if m['lead_time'] is not None:
                    lead_times.append(m['lead_time'])
                if m['f1'] is not None:
                    f1s.append(m['f1'])
                fa_rates.append(m['false_alarm_rate'])
                if m['precision'] is not None:
                    precisions.append(m['precision'])
                if m['recall'] is not None:
                    recalls.append(m['recall'])

            summary[mode] = {
                'lead_time_mean': np.mean(lead_times) if lead_times else None,
                'lead_time_std': np.std(lead_times) if len(lead_times) > 1 else 0,
                'f1_mean': np.mean(f1s) if f1s else None,
                'f1_std': np.std(f1s) if len(f1s) > 1 else 0,
                'fa_rate_mean': np.mean(fa_rates),
                'fa_rate_std': np.std(fa_rates),
                'precision_mean': np.mean(precisions) if precisions else None,
                'recall_mean': np.mean(recalls) if recalls else None,
                'n_runs': len(all_runs),
                'is_true_drift': is_true_drift
            }

        return {'scenario': scenario_key, 'summary': summary, 'all_runs': all_runs}

    def run_all_scenarios(self, data: Dict) -> Dict:

        print("\n" + "=" * 80)
        print("EXPERIMENT 1: GOVERNANCE VALUE VERIFICATION")
        print("Full-EW: OR logic (earliest detection)")
        print("Full-MR: AND logic (lowest false alarm)")
        print("=" * 80)

        all_results = {}

        for scenario_key in DRIFT_SCENARIOS:
            scenario = DRIFT_SCENARIOS[scenario_key]
            is_true = scenario.get('is_true_drift', True)
            print(f"\n[{scenario_key}] {scenario['name']}")
            print(f"  Type: {'TRUE DRIFT' if is_true else 'NEGATIVE CONTROL'}")

            result = self.run_with_seeds(data, scenario_key)
            all_results[scenario_key] = result

            s = result['summary']
            for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
                m = s[mode]

                if is_true:
                    lt = f"{m['lead_time_mean']:+.1f}±{m['lead_time_std']:.1f}" if m['lead_time_mean'] is not None else "N/A"
                    f1 = f"{m['f1_mean']:.1%}±{m['f1_std']:.1%}" if m['f1_mean'] is not None else "N/A"
                else:
                    lt = "N/A"
                    f1 = "N/A"

                fa = f"{m['fa_rate_mean']:.1%}±{m['fa_rate_std']:.1%}"
                print(f"  {mode:15s}: Lead={lt:>12s}, F1={f1:>14s}, FA={fa}")

        return all_results

    def generate_summary_table(self, all_results: Dict) -> pd.DataFrame:

        rows = []

        for scenario_key, result in all_results.items():
            scenario = DRIFT_SCENARIOS[scenario_key]
            is_true = scenario.get('is_true_drift', True)
            s = result['summary']

            for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
                m = s[mode]

                if is_true:
                    lt_str = f"{m['lead_time_mean']:+.1f}±{m['lead_time_std']:.1f}" if m['lead_time_mean'] is not None else "N/A"
                    f1_str = f"{m['f1_mean']:.1%}±{m['f1_std']:.1%}" if m['f1_mean'] is not None else "N/A"
                    prec_str = f"{m['precision_mean']:.1%}" if m['precision_mean'] is not None else "N/A"
                    rec_str = f"{m['recall_mean']:.1%}" if m['recall_mean'] is not None else "N/A"
                else:
                    lt_str = "N/A"
                    f1_str = "N/A"
                    prec_str = "N/A"
                    rec_str = "N/A"

                rows.append({
                    'Scenario': scenario['name'],
                    'Is_True_Drift': is_true,
                    'Mode': mode.replace('_', '-').title(),
                    'Lead_Time': lt_str,
                    'F1': f1_str,
                    'FA_Rate': f"{m['fa_rate_mean']:.1%}±{m['fa_rate_std']:.1%}",
                    'Precision': prec_str,
                    'Recall': rec_str,
                })

        return pd.DataFrame(rows)

    def plot_scenario(self, scenario_result: Dict, save_path: str = None):

        example = scenario_result['all_runs'][0]
        d = example['drift_data']
        results = example['results']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={'height_ratios': [2, 1]})

        months = np.arange(1, len(d['baseline']) + 1)
        onset = d['onset_month']

        ax1 = axes[0]
        ax1.plot(months, d['baseline'], 'k--', lw=2, label='Baseline', alpha=0.7)
        ax1.plot(months, d['forecast_p50'], 'b-', lw=2, marker='o', ms=4,
                label='Forecast', alpha=0.8)
        ax1.plot(months, d['drifted_dt'], 'r-', lw=2, marker='s', ms=4,
                label='DT Verified', alpha=0.8)

        if d['is_true_drift']:
            drift_months = months[d['drift_mask']]
            if len(drift_months) > 0:
                ax1.axvspan(drift_months[0]-0.5, drift_months[-1]+0.5,
                           alpha=0.15, color='green', label='True Drift Period')

        colors = {'full_ew': '#f39c12', 'full_mr': '#27ae60',
                  'dt_only': '#e74c3c', 'forecast_only': '#3498db'}
        labels_map = {'full_ew': 'Full-EW', 'full_mr': 'Full-MR',
                      'dt_only': 'DT-only', 'forecast_only': 'FC-only'}

        for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
            m = results[mode]
            if m['first_trigger']:
                ax1.axvline(m['first_trigger'], color=colors[mode], ls='-', lw=2.5, alpha=0.7,
                           label=f"{labels_map[mode]} (M{m['first_trigger']})")

        if d['is_true_drift']:
            ax1.axvline(onset, color='black', ls=':', lw=2, label=f'Onset (M{onset})')

        ax1.set_ylabel('Cumulative Progress (%)', fontsize=11)
        title = f"Scenario: {d['scenario_info']['name']}"
        if not d['is_true_drift']:
            title += " [NEGATIVE CONTROL]"
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        dev_fc = d['forecast_p50'] - d['baseline']
        dev_dt = d['drifted_dt'] - d['baseline']

        ax2.plot(months, dev_fc, 'b-', lw=2, label='Forecast dev', alpha=0.8)
        ax2.plot(months, dev_dt, 'r-', lw=2, label='DT dev', alpha=0.8)

        ax2.axhline(self.tolerance, color='orange', ls='--', lw=1.5)
        ax2.axhline(-self.tolerance, color='orange', ls='--', lw=1.5,
                   label=f'±{self.tolerance}pp')
        ax2.axhline(0, color='black', lw=0.5)

        ax2.set_xlabel('Month', fontsize=11)
        ax2.set_ylabel('Deviation (pp)', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(self.exp_dir, f"scenario_{scenario_result['scenario']}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_summary(self, all_results: Dict):

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Two-tier Governance: Full-EW (OR) vs Full-MR (AND) vs Single-source',
                    fontsize=14, fontweight='bold')

        true_drift = [k for k, v in DRIFT_SCENARIOS.items() if v.get('is_true_drift', True)]
        neg_control = [k for k, v in DRIFT_SCENARIOS.items() if not v.get('is_true_drift', True)]

        labels = [DRIFT_SCENARIOS[s]['name'][:18] for s in true_drift]
        x = np.arange(len(true_drift))
        width = 0.2

        modes = ['full_ew', 'full_mr', 'dt_only', 'forecast_only']
        colors = ['#f39c12', '#27ae60', '#e74c3c', '#3498db']
        mode_labels = ['Full-EW', 'Full-MR', 'DT-only', 'FC-only']

        # (a) Lead Time
        ax1 = axes[0, 0]
        for i, (mode, c, ml) in enumerate(zip(modes, colors, mode_labels)):
            vals = []
            errs = []
            for s in true_drift:
                m = all_results[s]['summary'][mode]
                vals.append(m['lead_time_mean'] if m['lead_time_mean'] is not None else 0)
                errs.append(m['lead_time_std'] if m['lead_time_mean'] is not None else 0)
            ax1.bar(x + i*width - 1.5*width, vals, width, yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_title('(a) Detection Lead Time (TRUE DRIFT)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Lead Time (months, +ve=early)')
        ax1.axhline(0, color='black', lw=0.5)
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # (b) F1 Score
        ax2 = axes[0, 1]
        for i, (mode, c, ml) in enumerate(zip(modes, colors, mode_labels)):
            vals = []
            errs = []
            for s in true_drift:
                m = all_results[s]['summary'][mode]
                vals.append(m['f1_mean'] * 100 if m['f1_mean'] is not None else 0)
                errs.append(m['f1_std'] * 100 if m['f1_mean'] is not None else 0)
            ax2.bar(x + i*width - 1.5*width, vals, width, yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_title('(b) F1 Score (TRUE DRIFT)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 (%)')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        # (c) FA Rate for true drift
        ax3 = axes[1, 0]
        for i, (mode, c, ml) in enumerate(zip(modes, colors, mode_labels)):
            vals = [all_results[s]['summary'][mode]['fa_rate_mean'] * 100 for s in true_drift]
            errs = [all_results[s]['summary'][mode]['fa_rate_std'] * 100 for s in true_drift]
            ax3.bar(x + i*width - 1.5*width, vals, width, yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax3.set_title('(c) False Alarm Rate (TRUE DRIFT)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('FA Rate (%)')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

        # (d) Negative Control comparison
        ax4 = axes[1, 1]
        nc_labels = [DRIFT_SCENARIOS[s]['name'][:20] for s in neg_control]
        x_nc = np.arange(len(neg_control))

        for i, (mode, c, ml) in enumerate(zip(modes, colors, mode_labels)):
            vals = [all_results[s]['summary'][mode]['fa_rate_mean'] * 100 for s in neg_control]
            errs = [all_results[s]['summary'][mode]['fa_rate_std'] * 100 for s in neg_control]
            ax4.bar(x_nc + i*width - 1.5*width, vals, width, yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)

        ax4.set_xticks(x_nc)
        ax4.set_xticklabels(nc_labels, rotation=45, ha='right', fontsize=9)
        ax4.set_title('(d) FA Rate in NEGATIVE CONTROL\n(Full-MR should be lowest)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('FA Rate (%)')
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.exp_dir, 'summary_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {save_path}")

    def generate_key_findings(self, all_results: Dict) -> str:

        lines = []
        lines.append("=" * 70)
        lines.append("KEY FINDINGS")
        lines.append("=" * 70)

        # 1. Full-EW early warning (earlier than both Forecast-only and DT-only)
        lines.append("\n1. EARLY WARNING (Full-EW uses OR logic → earliest detection):")
        for s in ['B_cost_leading', 'A2_overrun_moderate']:
            if s in all_results:
                ew_lead = all_results[s]['summary']['full_ew']['lead_time_mean']
                fc_lead = all_results[s]['summary']['forecast_only']['lead_time_mean']
                dt_lead = all_results[s]['summary']['dt_only']['lead_time_mean']

                if ew_lead is not None:
                    fc_adv = ew_lead - (fc_lead or 0)
                    dt_adv = ew_lead - (dt_lead or 0)
                    lines.append(f"   {DRIFT_SCENARIOS[s]['name']}: Full-EW Lead={ew_lead:+.1f}")
                    lines.append(f"      vs FC-only: {fc_adv:+.1f} months advantage")
                    lines.append(f"      vs DT-only: {dt_adv:+.1f} months advantage")

        # 2. Full-MR suppresses false alarms (lower than single-source).
        lines.append("\n2. FALSE ALARM SUPPRESSION (Full-MR uses AND logic → lowest FA):")
        for s in ['D1_spike_dt_only', 'D2_spike_forecast_only']:
            if s in all_results:
                mr_fa = all_results[s]['summary']['full_mr']['fa_rate_mean']
                dt_fa = all_results[s]['summary']['dt_only']['fa_rate_mean']
                fc_fa = all_results[s]['summary']['forecast_only']['fa_rate_mean']

                lines.append(f"   {DRIFT_SCENARIOS[s]['name']}:")
                lines.append(f"      Full-MR FA = {mr_fa:.1%}")
                lines.append(f"      DT-only FA = {dt_fa:.1%}")
                lines.append(f"      FC-only FA = {fc_fa:.1%}")

        # 3. DT's Unique Value
        lines.append("\n3. DT UNIQUE VALUE (Progress-Lag scenario):")
        if 'C_progress_lag' in all_results:
            dt_f1 = all_results['C_progress_lag']['summary']['dt_only']['f1_mean']
            fc_f1 = all_results['C_progress_lag']['summary']['forecast_only']['f1_mean']
            lines.append(f"   DT-only F1 = {dt_f1:.1%}")
            lines.append(f"   FC-only F1 = {fc_f1:.1%}" if fc_f1 else "   FC-only F1 = N/A (cannot detect)")

        return "\n".join(lines)

    def run_full_experiment(self, data: Dict):

        all_results = self.run_all_scenarios(data)

        df = self.generate_summary_table(all_results)
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(self.exp_dir, 'summary_table.csv'), index=False)

        print("\n[Generating Plots]")
        for key in ['A2_overrun_moderate', 'B_cost_leading', 'C_progress_lag',
                    'D1_spike_dt_only', 'D2_spike_forecast_only']:
            if key in all_results:
                self.plot_scenario(all_results[key])
        self.plot_summary(all_results)

        findings = self.generate_key_findings(all_results)
        print("\n" + findings)

        with open(os.path.join(self.exp_dir, 'key_findings.txt'), 'w') as f:
            f.write(findings)

        print(f"\n✓ Experiment 1 completed! Results in: {self.exp_dir}")
        return all_results


# ===========================================================================================
# PART 6: Main function
# ===========================================================================================

def main():
    from pathlib import Path

    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    data = generate_baseline_data(total_months=24, seed=42)

    print(f"\nBaseline data: {data['baseline'][0]:.1f}% → {data['baseline'][-1]:.1f}%")

    exp = GovernanceValueExperiment(
        OUTPUT_DIR,
        tolerance=5.0,
        duration=2,
        n_seeds=30
    )
    exp.run_full_experiment(data)


if __name__ == "__main__":
    main()