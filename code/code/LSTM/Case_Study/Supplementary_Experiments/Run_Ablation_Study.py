"""
Run_Ablation_Study.py - Two-tier Governance Ablation Experiment

Reproducibility enhancements (revision):
  - Removed hardcoded BASE_DIR; use --output_dir CLI argument
  - Added full argparse: --output_dir, --n_seeds, --base_seed,
                         --tolerance, --duration, --scenarios
  - Complete experiment config (including all seeds) saved to JSON at startup
  - inject_drift() noise seeded per (scenario, seed) pair → deterministic
  - run_with_seeds() generates n_seeds using base_seed + i*100 (documented)
  - All random calls route through numpy Generator (default_rng) for reproducibility

Core design:
  - Full-EW: forecast OR DT either exceeds threshold → early warning
  - Full-MR: forecast AND DT both confirm → management escalation
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ===========================================================================================
# PART 1: drift scenario definitions
# ===========================================================================================

DRIFT_SCENARIOS = {
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
    """
    Generate synthetic baseline progress data.

    Uses numpy.random.default_rng(seed) for fully reproducible, independent
    random streams.  Every call with the same seed produces identical output.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(1, total_months + 1)

    baseline = 100 / (1 + np.exp(-0.4 * (t - total_months / 2)))
    baseline = baseline / baseline[-1] * 100

    noise_std = 0.3
    dt_original = baseline + rng.normal(0, noise_std, total_months)
    dt_original = np.maximum.accumulate(np.clip(dt_original, 0, 100))

    forecast_p50 = baseline + rng.normal(0, noise_std, total_months)
    forecast_p50 = np.maximum.accumulate(np.clip(forecast_p50, 0, 100))

    return {
        'baseline':     baseline,
        'dt_original':  dt_original,
        'forecast_p50': forecast_p50,
        'total_months': total_months,
        'seed':         seed,          # stored for traceability
    }


def inject_drift(baseline:     np.ndarray,
                 dt_original:  np.ndarray,
                 forecast_p50: np.ndarray,
                 scenario_key: str,
                 noise_std:    float = 0.5,
                 seed:         int   = 0) -> Dict:
    """
    Inject a drift pattern into the baseline data.

    Reproducibility: noise added after drift injection is seeded via
    numpy.random.default_rng(seed).  seed should encode both the scenario
    and the outer seed index so every (scenario, seed) combination is
    independent and deterministic.
    """
    scenario = DRIFT_SCENARIOS[scenario_key]
    n        = len(baseline)
    rng      = np.random.default_rng(seed)   # deterministic per (scenario, seed)

    drifted_dt = dt_original.copy()
    drifted_fc = forecast_p50.copy()

    is_true_drift = scenario.get('is_true_drift', True)
    drift_mask    = np.zeros(n, dtype=bool)

    onset      = scenario.get('onset_month', 14)
    onset_idx  = onset - 1
    stype      = scenario['type']

    if stype == 'sustained_overrun':
        rate     = scenario['drift_rate_pp']
        lead     = scenario.get('forecast_lead', 2)
        fc_start = max(0, onset_idx - lead)

        for i in range(fc_start, n):
            drifted_fc[i] = baseline[i] + rate * (i - fc_start + 1)
        for i in range(onset_idx, n):
            drifted_dt[i] = baseline[i] + rate * (i - onset_idx + 1)
        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif stype == 'cost_leading':
        rate     = scenario['drift_rate_pp']
        fc_lead  = scenario.get('forecast_lead', 3)
        dt_lag   = scenario.get('dt_lag', 2)
        fc_start = max(0, onset_idx - fc_lead)
        dt_start = onset_idx + dt_lag

        for i in range(fc_start, n):
            drifted_fc[i] = baseline[i] + rate * (i - fc_start + 1)
        for i in range(dt_start, n):
            drifted_dt[i] = baseline[i] + rate * (i - dt_start + 1)
        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif stype == 'progress_lag':
        rate = scenario['lag_rate_pp']
        for i in range(onset_idx, n):
            drifted_dt[i] = baseline[i] - rate * (i - onset_idx + 1)
        if is_true_drift:
            drift_mask[onset_idx:] = True

    elif stype == 'spike_dt_only':
        spike_mag = scenario['spike_magnitude_pp']
        for m in scenario.get('spike_months', []):
            idx = m - 1
            if 0 <= idx < n:
                drifted_dt[idx] = baseline[idx] + spike_mag

    elif stype == 'spike_forecast_only':
        spike_mag = scenario['spike_magnitude_pp']
        for m in scenario.get('spike_months', []):
            idx = m - 1
            if 0 <= idx < n:
                drifted_fc[idx] = baseline[idx] + spike_mag

    # Add observation noise using the per-(scenario, seed) RNG
    drifted_dt += rng.normal(0, noise_std, n)
    drifted_fc += rng.normal(0, noise_std, n)

    drifted_dt = np.maximum.accumulate(np.clip(drifted_dt, 0, 150))
    drifted_fc = np.maximum.accumulate(np.clip(drifted_fc, 0, 150))

    return {
        'baseline':     baseline,
        'dt_original':  dt_original,
        'drifted_dt':   drifted_dt,
        'forecast_p50': drifted_fc,
        'drift_mask':   drift_mask,
        'onset_month':  onset,
        'is_true_drift': is_true_drift,
        'scenario_key': scenario_key,
        'scenario_info': scenario,
        'inject_seed':  seed,          # traceability
    }


# ===========================================================================================
# PART 3: Detectors
# ===========================================================================================

class SimpleDetector:
    def __init__(self, tolerance: float = 5.0, duration: int = 2):
        self.tolerance = tolerance
        self.duration  = duration

    def _find_sustained_abs(self, deviation: np.ndarray) -> List[Tuple[int, int]]:
        n, intervals, i = len(deviation), [], 0
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

    def detect(self, data: np.ndarray, baseline: np.ndarray) -> List[int]:
        dev = data - baseline
        triggered = set()
        for s, e in self._find_sustained_abs(dev):
            triggered.update(range(s + 1, e + 2))
        return sorted(triggered)


class FusionGovernanceDetector:
    """
    Two-tier fusion governance detector.

    Full-EW (Early Warning) : forecast OR DT exceeds threshold → OR logic
    Full-MR (Management Review): forecast AND DT both confirm  → AND logic
    """

    def __init__(self, tolerance: float = 5.0, duration: int = 2, mr_window: int = 3):
        self.tolerance          = tolerance
        self.duration           = duration
        self.mr_window          = mr_window
        self.dt_lag_mr_duration = max(duration + 1, 3)

    def _find_sustained_above(self, dev: np.ndarray, thr: float,
                               min_len: int) -> List[Tuple[int, int]]:
        n, intervals, i = len(dev), [], 0
        while i < n:
            if dev[i] > thr:
                start = i
                j = i + 1
                while j < n and dev[j] > thr:
                    j += 1
                if j - start >= min_len:
                    intervals.append((start, j - 1))
                i = j
            else:
                i += 1
        return intervals

    def _find_sustained_below(self, dev: np.ndarray, thr: float,
                               min_len: int) -> List[Tuple[int, int]]:
        n, intervals, i = len(dev), [], 0
        while i < n:
            if dev[i] < thr:
                start = i
                j = i + 1
                while j < n and dev[j] < thr:
                    j += 1
                if j - start >= min_len:
                    intervals.append((start, j - 1))
                i = j
            else:
                i += 1
        return intervals

    def detect_early_warning(self, forecast: np.ndarray, dt: np.ndarray,
                              baseline: np.ndarray) -> List[int]:
        dev_fc  = forecast - baseline
        dev_dt  = dt - baseline
        triggered = set()

        for s, e in self._find_sustained_above(dev_fc, +self.tolerance, self.duration):
            triggered.update(range(s + 1, e + 2))
        for s, e in self._find_sustained_above(dev_dt, +self.tolerance, self.duration):
            triggered.update(range(s + 1, e + 2))
        for s, e in self._find_sustained_below(dev_fc, -self.tolerance, self.duration):
            triggered.update(range(s + 1, e + 2))
        for s, e in self._find_sustained_below(dev_dt, -self.tolerance, self.duration):
            triggered.update(range(s + 1, e + 2))

        return sorted(triggered)

    def detect_management_review(self, forecast: np.ndarray, dt: np.ndarray,
                                  baseline: np.ndarray) -> List[int]:
        dev_fc = forecast - baseline
        dev_dt = dt - baseline
        n      = len(baseline)
        triggered = set()

        fc_pos = self._find_sustained_above(dev_fc, +self.tolerance, self.duration)
        dt_pos = self._find_sustained_above(dev_dt, +self.tolerance, self.duration)

        for fc_s, fc_e in fc_pos:
            check_s = max(0, fc_s - self.mr_window)
            check_e = min(n - 1, fc_e + self.mr_window)
            for dt_s, dt_e in dt_pos:
                if not (dt_s <= check_e and dt_e >= check_s):
                    continue
                ov_s = max(fc_s, dt_s)
                ov_e = min(fc_e, dt_e)
                if ov_e >= ov_s:
                    triggered.update(range(ov_s + 1, ov_e + 2))
                else:
                    triggered.update(range(dt_s + 1, dt_e + 2))

        lag_dur = getattr(self, 'dt_lag_mr_duration', self.duration)
        for s, e in self._find_sustained_below(dev_dt, -self.tolerance, lag_dur):
            triggered.update(range(s + 1, e + 2))

        return sorted(triggered)

    def detect_full(self, forecast: np.ndarray, dt: np.ndarray,
                    baseline: np.ndarray) -> Tuple[List[int], List[int]]:
        ew = self.detect_early_warning(forecast, dt, baseline)
        mr = self.detect_management_review(forecast, dt, baseline)
        return ew, mr


# ===========================================================================================
# PART 4: Metrics
# ===========================================================================================

def calculate_metrics(triggered: List[int], onset_month: int,
                      drift_mask: np.ndarray, total_months: int,
                      is_true_drift: bool) -> Dict:
    trigger_set = set(triggered)
    tp = fp = fn = tn = 0

    for m in range(1, total_months + 1):
        idx          = m - 1
        is_triggered = m in trigger_set
        has_drift    = drift_mask[idx] if (is_true_drift and idx < len(drift_mask)) else False

        if is_triggered and has_drift:       tp += 1
        elif is_triggered and not has_drift: fp += 1
        elif not is_triggered and has_drift: fn += 1
        else:                                tn += 1

    if is_true_drift:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.0
    else:
        precision = recall = f1 = None

    fa_rate      = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    first_trigger = min(triggered) if triggered else None
    lead_time     = (onset_month - first_trigger) \
                    if (is_true_drift and first_trigger is not None) else None

    return {
        'first_trigger': first_trigger,
        'lead_time':     lead_time,
        'onset_month':   onset_month,
        'precision':     precision,
        'recall':        recall,
        'f1':            f1,
        'false_alarm_rate': fa_rate,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'total_triggers': len(triggered),
        'is_true_drift':  is_true_drift,
    }


# ===========================================================================================
# PART 5: Experiment class
# ===========================================================================================

class GovernanceValueExperiment:

    def __init__(self, output_dir: str, tolerance: float = 5.0,
                 duration: int = 2, n_seeds: int = 30, base_seed: int = 0):
        """
        Args:
            output_dir : Base output directory (no hardcoded paths)
            tolerance  : Governance threshold in percentage points
            duration   : Minimum consecutive months for trigger
            n_seeds    : Number of random seeds per scenario
            base_seed  : Base seed; per-scenario seeds = base_seed + i*100
        """
        self.output_dir = output_dir
        self.tolerance  = tolerance
        self.duration   = duration
        self.n_seeds    = n_seeds
        self.base_seed  = base_seed

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(output_dir, f"run_ablation_study_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Save full config immediately for reproducibility
        self.config = {
            'output_dir':  output_dir,
            'exp_dir':     self.exp_dir,
            'tolerance':   tolerance,
            'duration':    duration,
            'n_seeds':     n_seeds,
            'base_seed':   base_seed,
            'seed_formula': 'seed_i = base_seed + i * 100  (i = 0 .. n_seeds-1)',
            'timestamp':   timestamp,
        }
        config_path = os.path.join(self.exp_dir, 'experiment_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✓ Experiment output: {self.exp_dir}")
        print(f"  Tolerance: ±{tolerance}pp, Duration: {duration} months, "
              f"Seeds: {n_seeds}, Base seed: {base_seed}")
        print(f"  Config saved: {config_path}")

    def run_single(self, data: Dict, scenario_key: str, seed: int = 42) -> Dict:
        """Run one scenario with one seed."""
        drift_data = inject_drift(
            data['baseline'].copy(),
            data['dt_original'].copy(),
            data['forecast_p50'].copy(),
            scenario_key,
            noise_std=0.5,
            seed=seed,          # fully deterministic per (scenario, seed)
        )

        is_true_drift = drift_data['is_true_drift']
        onset         = drift_data['onset_month']
        n             = len(drift_data['baseline'])
        results       = {}

        full_det = FusionGovernanceDetector(self.tolerance, self.duration)
        ew, mr   = full_det.detect_full(
            drift_data['forecast_p50'],
            drift_data['drifted_dt'],
            drift_data['baseline'],
        )
        results['full_ew'] = {**calculate_metrics(ew, onset, drift_data['drift_mask'],
                                                   n, is_true_drift), 'mode': 'Full-EW'}
        results['full_mr'] = {**calculate_metrics(mr, onset, drift_data['drift_mask'],
                                                   n, is_true_drift), 'mode': 'Full-MR'}

        dt_det  = SimpleDetector(self.tolerance, self.duration)
        results['dt_only'] = {
            **calculate_metrics(
                dt_det.detect(drift_data['drifted_dt'], drift_data['baseline']),
                onset, drift_data['drift_mask'], n, is_true_drift),
            'mode': 'DT-only',
        }

        fc_det  = SimpleDetector(self.tolerance, self.duration)
        results['forecast_only'] = {
            **calculate_metrics(
                fc_det.detect(drift_data['forecast_p50'], drift_data['baseline']),
                onset, drift_data['drift_mask'], n, is_true_drift),
            'mode': 'Forecast-only',
        }

        return {'scenario': scenario_key, 'results': results, 'drift_data': drift_data}

    def run_with_seeds(self, data: Dict, scenario_key: str) -> Dict:
        """
        Run a scenario across n_seeds seeds.

        Seed formula: seed_i = base_seed + i * 100
        This ensures independent random streams across seeds and across scenarios
        (since inject_drift uses a separate RNG per seed value).
        Each seed_i is also used as the data-generation seed so the baseline
        data is varied per seed, providing genuine Monte Carlo coverage.
        """
        all_runs = []
        used_seeds = []

        for i in range(self.n_seeds):
            seed_i = self.base_seed + i * 100
            used_seeds.append(seed_i)
            seed_data = generate_baseline_data(total_months=24, seed=seed_i)
            result    = self.run_single(seed_data, scenario_key, seed=seed_i)
            all_runs.append(result)

        is_true_drift = DRIFT_SCENARIOS[scenario_key].get('is_true_drift', True)
        summary = {}

        for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
            lead_times, f1s, fa_rates, precisions, recalls = [], [], [], [], []

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
                'lead_time_mean':  np.mean(lead_times) if lead_times else None,
                'lead_time_std':   np.std(lead_times)  if len(lead_times) > 1 else 0.0,
                'f1_mean':         np.mean(f1s)        if f1s else None,
                'f1_std':          np.std(f1s)         if len(f1s) > 1 else 0.0,
                'fa_rate_mean':    np.mean(fa_rates),
                'fa_rate_std':     np.std(fa_rates)    if len(fa_rates) > 1 else 0.0,
                'precision_mean':  np.mean(precisions) if precisions else None,
                'recall_mean':     np.mean(recalls)    if recalls else None,
                'n_runs':          len(all_runs),
                'is_true_drift':   is_true_drift,
            }

        return {
            'scenario':    scenario_key,
            'summary':     summary,
            'all_runs':    all_runs,
            'used_seeds':  used_seeds,   # full seed list in output
        }

    def run_all_scenarios(self, data: Dict,
                          scenarios: Optional[List[str]] = None) -> Dict:
        """
        Run all (or selected) scenarios.

        Args:
            scenarios: list of scenario keys; defaults to all DRIFT_SCENARIOS
        """
        if scenarios is None:
            scenarios = list(DRIFT_SCENARIOS.keys())

        print("\n" + "=" * 80)
        print("EXPERIMENT: GOVERNANCE VALUE ABLATION")
        print("Full-EW: OR logic (earliest detection)")
        print("Full-MR: AND logic (lowest false alarm)")
        print("=" * 80)

        all_results = {}
        for key in scenarios:
            if key not in DRIFT_SCENARIOS:
                print(f"[SKIP] Unknown scenario key: {key}")
                continue

            sc     = DRIFT_SCENARIOS[key]
            is_true = sc.get('is_true_drift', True)
            print(f"\n[{key}] {sc['name']}  "
                  f"({'TRUE DRIFT' if is_true else 'NEGATIVE CONTROL'})")

            result = self.run_with_seeds(data, key)
            all_results[key] = result

            s = result['summary']
            for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
                m   = s[mode]
                lt  = (f"{m['lead_time_mean']:+.1f}±{m['lead_time_std']:.1f}"
                       if (is_true and m['lead_time_mean'] is not None) else "N/A")
                f1  = (f"{m['f1_mean']:.1%}±{m['f1_std']:.1%}"
                       if (is_true and m['f1_mean'] is not None) else "N/A")
                fa  = f"{m['fa_rate_mean']:.1%}±{m['fa_rate_std']:.1%}"
                print(f"  {mode:15s}: Lead={lt:>12s}, F1={f1:>14s}, FA={fa}")

        # Save seed log
        seed_log = {}
        for key, res in all_results.items():
            seed_log[key] = res.get('used_seeds', [])
        with open(os.path.join(self.exp_dir, 'seed_log.json'), 'w') as f:
            json.dump(seed_log, f, indent=2)
        print(f"\n✓ Seed log saved: {self.exp_dir}/seed_log.json")

        return all_results

    def generate_summary_table(self, all_results: Dict) -> pd.DataFrame:
        rows = []
        for key, result in all_results.items():
            sc     = DRIFT_SCENARIOS[key]
            is_true = sc.get('is_true_drift', True)
            s      = result['summary']

            for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
                m = s[mode]
                if is_true:
                    lt_str   = (f"{m['lead_time_mean']:+.1f}±{m['lead_time_std']:.1f}"
                                if m['lead_time_mean'] is not None else "N/A")
                    f1_str   = (f"{m['f1_mean']:.1%}±{m['f1_std']:.1%}"
                                if m['f1_mean'] is not None else "N/A")
                    prec_str = f"{m['precision_mean']:.1%}" if m['precision_mean'] is not None else "N/A"
                    rec_str  = f"{m['recall_mean']:.1%}"    if m['recall_mean']    is not None else "N/A"
                else:
                    lt_str = f1_str = prec_str = rec_str = "N/A"

                rows.append({
                    'Scenario':      sc['name'],
                    'Is_True_Drift': is_true,
                    'Mode':          mode.replace('_', '-').title(),
                    'Lead_Time':     lt_str,
                    'F1':            f1_str,
                    'FA_Rate':       f"{m['fa_rate_mean']:.1%}±{m['fa_rate_std']:.1%}",
                    'Precision':     prec_str,
                    'Recall':        rec_str,
                })
        return pd.DataFrame(rows)

    def plot_scenario(self, scenario_result: Dict, save_path: str = None):
        example = scenario_result['all_runs'][0]
        d       = example['drift_data']
        results = example['results']

        fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={'height_ratios': [2, 1]})
        months = np.arange(1, len(d['baseline']) + 1)
        onset  = d['onset_month']

        ax1 = axes[0]
        ax1.plot(months, d['baseline'],     'k--', lw=2, label='Baseline', alpha=0.7)
        ax1.plot(months, d['forecast_p50'], 'b-',  lw=2, marker='o', ms=4,
                 label='Forecast', alpha=0.8)
        ax1.plot(months, d['drifted_dt'],   'r-',  lw=2, marker='s', ms=4,
                 label='DT Verified', alpha=0.8)

        if d['is_true_drift']:
            dm = months[d['drift_mask']]
            if len(dm):
                ax1.axvspan(dm[0] - 0.5, dm[-1] + 0.5, alpha=0.15,
                            color='green', label='True Drift Period')

        colors     = {'full_ew': '#f39c12', 'full_mr': '#27ae60',
                      'dt_only': '#e74c3c', 'forecast_only': '#3498db'}
        lbl_map    = {'full_ew': 'Full-EW', 'full_mr': 'Full-MR',
                      'dt_only': 'DT-only', 'forecast_only': 'FC-only'}

        for mode in ['full_ew', 'full_mr', 'dt_only', 'forecast_only']:
            ft = results[mode]['first_trigger']
            if ft:
                ax1.axvline(ft, color=colors[mode], ls='-', lw=2.5, alpha=0.7,
                            label=f"{lbl_map[mode]} (M{ft})")
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
        ax2.plot(months, d['forecast_p50'] - d['baseline'], 'b-',
                 lw=2, label='Forecast dev', alpha=0.8)
        ax2.plot(months, d['drifted_dt']   - d['baseline'], 'r-',
                 lw=2, label='DT dev',      alpha=0.8)
        ax2.axhline(+self.tolerance, color='orange', ls='--', lw=1.5)
        ax2.axhline(-self.tolerance, color='orange', ls='--', lw=1.5,
                    label=f'±{self.tolerance}pp')
        ax2.axhline(0, color='black', lw=0.5)
        ax2.set_xlabel('Month', fontsize=11)
        ax2.set_ylabel('Deviation (pp)', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(
                self.exp_dir, f"scenario_{scenario_result['scenario']}.png"
            )
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")

    def plot_summary(self, all_results: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Two-tier Governance: Full-EW (OR) vs Full-MR (AND) vs Single-source',
                     fontsize=14, fontweight='bold')

        true_drift  = [k for k in all_results if DRIFT_SCENARIOS[k].get('is_true_drift', True)]
        neg_control = [k for k in all_results if not DRIFT_SCENARIOS[k].get('is_true_drift', True)]

        labels  = [DRIFT_SCENARIOS[s]['name'][:18] for s in true_drift]
        x       = np.arange(len(true_drift))
        width   = 0.2
        modes   = ['full_ew', 'full_mr', 'dt_only', 'forecast_only']
        colors  = ['#f39c12', '#27ae60', '#e74c3c', '#3498db']
        mlabels = ['Full-EW', 'Full-MR', 'DT-only', 'FC-only']

        def _bar(ax, metric_fn, err_fn, ylabel, title):
            for i, (m, c, ml) in enumerate(zip(modes, colors, mlabels)):
                vals = [metric_fn(all_results[s]['summary'][m]) for s in true_drift]
                errs = [err_fn(all_results[s]['summary'][m])   for s in true_drift]
                ax.bar(x + i * width - 1.5 * width, vals, width,
                       yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        _bar(axes[0, 0],
             lambda m: (m['lead_time_mean'] or 0),
             lambda m: (m['lead_time_std']  if m['lead_time_mean'] is not None else 0),
             'Lead Time (months, +ve=early)',
             '(a) Detection Lead Time (TRUE DRIFT)')
        axes[0, 0].axhline(0, color='black', lw=0.5)

        _bar(axes[0, 1],
             lambda m: (m['f1_mean'] * 100 if m['f1_mean'] is not None else 0),
             lambda m: (m['f1_std']  * 100 if m['f1_mean'] is not None else 0),
             'F1 (%)',
             '(b) F1 Score (TRUE DRIFT)')

        _bar(axes[1, 0],
             lambda m: m['fa_rate_mean'] * 100,
             lambda m: m['fa_rate_std']  * 100,
             'FA Rate (%)',
             '(c) False Alarm Rate (TRUE DRIFT)')

        # Negative control
        nc_labels = [DRIFT_SCENARIOS[s]['name'][:20] for s in neg_control]
        x_nc = np.arange(len(neg_control))
        for i, (m, c, ml) in enumerate(zip(modes, colors, mlabels)):
            vals = [all_results[s]['summary'][m]['fa_rate_mean'] * 100 for s in neg_control]
            errs = [all_results[s]['summary'][m]['fa_rate_std']  * 100 for s in neg_control]
            axes[1, 1].bar(x_nc + i * width - 1.5 * width, vals, width,
                           yerr=errs, label=ml, color=c, alpha=0.8, capsize=3)
        axes[1, 1].set_xticks(x_nc)
        axes[1, 1].set_xticklabels(nc_labels, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_title('(d) FA Rate in NEGATIVE CONTROL\n(Full-MR should be lowest)',
                              fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('FA Rate (%)')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.exp_dir, 'summary_comparison.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {path}")

    def generate_key_findings(self, all_results: Dict) -> str:
        lines = ["=" * 70, "KEY FINDINGS", "=" * 70]

        lines.append("\n1. EARLY WARNING (Full-EW — OR logic → earliest detection):")
        for s in ['B_cost_leading', 'A2_overrun_moderate']:
            if s in all_results:
                sm = all_results[s]['summary']
                ew_lead = sm['full_ew']['lead_time_mean']
                fc_lead = sm['forecast_only']['lead_time_mean']
                dt_lead = sm['dt_only']['lead_time_mean']
                if ew_lead is not None:
                    lines.append(f"   {DRIFT_SCENARIOS[s]['name']}: Full-EW Lead={ew_lead:+.1f}")
                    lines.append(f"      vs FC-only: {ew_lead - (fc_lead or 0):+.1f} months advantage")
                    lines.append(f"      vs DT-only: {ew_lead - (dt_lead or 0):+.1f} months advantage")

        lines.append("\n2. FALSE ALARM SUPPRESSION (Full-MR — AND logic → lowest FA):")
        for s in ['D1_spike_dt_only', 'D2_spike_forecast_only']:
            if s in all_results:
                sm = all_results[s]['summary']
                lines.append(f"   {DRIFT_SCENARIOS[s]['name']}:")
                lines.append(f"      Full-MR FA = {sm['full_mr']['fa_rate_mean']:.1%}")
                lines.append(f"      DT-only FA = {sm['dt_only']['fa_rate_mean']:.1%}")
                lines.append(f"      FC-only FA = {sm['forecast_only']['fa_rate_mean']:.1%}")

        lines.append("\n3. DT UNIQUE VALUE (Progress-Lag scenario):")
        if 'C_progress_lag' in all_results:
            sm = all_results['C_progress_lag']['summary']
            dt_f1 = sm['dt_only']['f1_mean']
            fc_f1 = sm['forecast_only']['f1_mean']
            lines.append(f"   DT-only F1 = {dt_f1:.1%}" if dt_f1 else "   DT-only F1 = N/A")
            lines.append(f"   FC-only F1 = {fc_f1:.1%}" if fc_f1 else "   FC-only F1 = N/A (cannot detect)")

        return "\n".join(lines)

    def run_full_experiment(self, data: Dict,
                             scenarios: Optional[List[str]] = None):
        all_results = self.run_all_scenarios(data, scenarios=scenarios)

        df = self.generate_summary_table(all_results)
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(self.exp_dir, 'summary_table.csv'), index=False)

        plot_keys = ['A2_overrun_moderate', 'B_cost_leading', 'C_progress_lag',
                     'D1_spike_dt_only', 'D2_spike_forecast_only']
        for key in plot_keys:
            if key in all_results:
                self.plot_scenario(all_results[key])
        self.plot_summary(all_results)

        findings = self.generate_key_findings(all_results)
        print("\n" + findings)
        with open(os.path.join(self.exp_dir, 'key_findings.txt'), 'w') as f:
            f.write(findings)

        print(f"\n✓ Ablation study complete! Results in: {self.exp_dir}")
        return all_results


# ===========================================================================================
# PART 6: CLI entry point
# ===========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Governance Ablation Study  (reproducibility revision)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Base output directory. Defaults to <project_root>/outputs. "
             "Auto-detected from script location if omitted."
    )
    parser.add_argument(
        "--n_seeds", type=int, default=30,
        help="Number of random seeds per scenario (default: 30)"
    )
    parser.add_argument(
        "--base_seed", type=int, default=0,
        help="Base random seed. Seed i = base_seed + i*100 (default: 0)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=5.0,
        help="Governance deviation threshold in pp (default: 5.0)"
    )
    parser.add_argument(
        "--duration", type=int, default=2,
        help="Min consecutive months for trigger (default: 2)"
    )
    parser.add_argument(
        "--scenarios", type=str, default="all",
        help="Comma-separated scenario keys, or 'all' (default: all)"
    )

    args = parser.parse_args()

    # Resolve output_dir without hardcoded paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-detect: three parents up from LSTM/Case_Study/Run_Ablation_Study.py
        script_dir  = Path(__file__).resolve().parents[3]
        output_dir  = str(script_dir / "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Scenario selection
    if args.scenarios.lower() == "all":
        scenarios = None   # run_all_scenarios defaults to all
    else:
        scenarios = [s.strip() for s in args.scenarios.split(',')]

    print("\n" + "=" * 70)
    print("GOVERNANCE ABLATION STUDY  (reproducibility revision)")
    print("=" * 70)
    print(f"Output dir   : {output_dir}")
    print(f"N seeds      : {args.n_seeds}")
    print(f"Base seed    : {args.base_seed}  "
          f"(seeds = {args.base_seed}, {args.base_seed+100}, "
          f"{args.base_seed+200}, ...)")
    print(f"Tolerance    : ±{args.tolerance} pp")
    print(f"Duration     : {args.duration} months")
    print(f"Scenarios    : {args.scenarios}")
    print("=" * 70)

    # Generate initial baseline (seed=base_seed for the initial data object)
    data = generate_baseline_data(total_months=24, seed=args.base_seed)
    print(f"\nBaseline data: {data['baseline'][0]:.1f}% → {data['baseline'][-1]:.1f}%")

    exp = GovernanceValueExperiment(
        output_dir=output_dir,
        tolerance=args.tolerance,
        duration=args.duration,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
    )
    exp.run_full_experiment(data, scenarios=scenarios)


if __name__ == "__main__":
    from pathlib import Path
    main()