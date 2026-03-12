"""
Run_Case_Study.py - Multi-seed Case Study Automation
Automate the complete Case Study workflow (multi-seed version)
Uses old filenames: Prediction_CS.py and Combine_Ensemble_CS.py

Steps:
1. Prediction_CS.py - Predict using 25 models
2. Combine_Ensemble_CS.py - Merge P10/P50/P90
3. Compare_Prediction.py - Compare predictions with actuals
4. Governance_Triggers.py - Governance trigger analysis
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LSTM_DIR = PROJECT_ROOT / "LSTM"


def validate_paths(args):
    print("\n" + "="*70)
    print("Validating Path Configuration")
    print("="*70)

    errors = []

    # Check project root directory
    if PROJECT_ROOT.exists():
        print(f"[OK] Project root directory: {PROJECT_ROOT}")
    else:
        errors.append(f"Project root not found: {PROJECT_ROOT}")

    # Check script files (using old names)
    scripts = {
        "Prediction_CS": LSTM_DIR / "Case_Study" / "Prediction_CS.py",
        "Combine_Ensemble_CS": LSTM_DIR / "Case_Study" / "Combine_Ensemble_CS.py",
        "Compare_Prediction": LSTM_DIR / "Case_Study" / "Compare_Prediction.py",
        "Governance_Triggers": LSTM_DIR / "Case_Study" / "Governance_Triggers.py"
    }

    for name, path in scripts.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            errors.append(f"{name} script not found: {path}")
            print(f"[MISSING] {name}: {path}")

    # Check model directory
    cv_model_dir = Path(args.cv_model_dir)
    if cv_model_dir.exists():
        print(f"[OK] CV model directory: {cv_model_dir}")
    else:
        errors.append(f"CV model directory not found: {cv_model_dir}")

    # Check input CSV
    input_csv = Path(args.input_csv)
    if input_csv.exists():
        print(f"[OK] Input CSV: {input_csv}")
    else:
        errors.append(f"Input CSV not found: {input_csv}")

    # Check actual progress CSV
    if args.actual_csv:
        actual_csv = Path(args.actual_csv)
        if actual_csv.exists():
            print(f"[OK] Actual progress CSV: {actual_csv}")
        else:
            errors.append(f"Actual CSV not found: {actual_csv}")

    # Check anchor CSV (if used)
    if args.anchor_mode == "csv" and args.anchor_csv:
        anchor_csv = Path(args.anchor_csv)
        if anchor_csv.exists():
            print(f"[OK] Anchor CSV: {anchor_csv}")
        else:
            errors.append(f"Anchor CSV not found: {anchor_csv}")

    # Check multi-seed model files
    print("\nChecking multi-seed model files...")
    all_models_found = True
    for fold in range(args.n_folds):
        fold_complete = True
        missing = []

        # Check scaler (one per fold)
        scaler_path = cv_model_dir / f"scaler_fold{fold}.pkl"
        if not scaler_path.exists():
            fold_complete = False
            missing.append("scaler")

        # Check each seed's model and config
        for seed in range(args.n_seeds):
            config_path = cv_model_dir / f"model_config_fold{fold}_seed{seed}.json"
            model_path = cv_model_dir / f"best_model_fold{fold}_seed{seed}.pt"

            if not config_path.exists():
                fold_complete = False
                missing.append(f"config_seed{seed}")
            if not model_path.exists():
                fold_complete = False
                missing.append(f"model_seed{seed}")

        if fold_complete:
            print(f"[OK] Fold {fold}: All {args.n_seeds} seeds + scaler present")
        else:
            print(f"[MISSING] Fold {fold}: Missing {', '.join(missing)}")
            all_models_found = False

    if not all_models_found:
        errors.append("Some multi-seed model files missing!")

    # If there are errors, print and exit
    if errors:
        print("\n" + "="*70)
        print("VALIDATION ERRORS")
        print("="*70)
        for error in errors:
            print(f"{error}")
        print("\nPlease fix the above errors before running.")
        return False

    print("\n" + "="*70)
    print("All paths validated successfully!")
    print("="*70)
    return True


def run_step(step_name, command, description):
    """Run a single step"""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)
    print(f"Description: {description}")
    print(f"Command: {' '.join(command)}")
    print("-"*70)

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
            cwd=PROJECT_ROOT
        )

        print("-"*70)
        print(f"{step_name} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print("-"*70)
        print(f"{step_name} failed with error:")
        print(f"   {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed Case Study Automation"
    )

    # Model parameters
    parser.add_argument("--cv_model_dir", type=str, required=True,
                        help="CV model directory containing multi-seed models")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds (default: 5)")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds per fold (default: 5)")

    # Input/output
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input case study CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--run_note", type=str, default="multiseed_cs",
                        help="Run note for output directory naming")

    # Project parameters
    parser.add_argument("--total_budget", type=float, required=True,
                        help="Total project budget (CNY)")
    parser.add_argument("--industry_config", type=str, default="",
                        help="Industry config file (optional)")

    # Anchoring parameters
    parser.add_argument("--anchor_mode", type=str, default="default",
                        choices=["default", "csv", "none"],
                        help="Anchoring mode")
    parser.add_argument("--anchor_csv", type=str, default=None,
                        help="CSV with actual data for anchoring")

    # Actual data (for comparison and calibration)
    parser.add_argument("--actual_csv", type=str, default=None,
                        help="CSV with actual progress data")

    # Governance trigger parameters
    parser.add_argument("--trigger_soft", type=float, default=5.0,
                        help="Soft trigger threshold (pp)")
    parser.add_argument("--trigger_hard", type=float, default=10.0,
                        help="Hard trigger threshold (pp)")
    parser.add_argument("--min_duration", type=int, default=2,
                        help="Minimum consecutive months for trigger")

    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda/cpu)")

    # Control steps
    parser.add_argument("--skip_prediction", action="store_true",
                        help="Skip prediction step (if already done)")
    parser.add_argument("--skip_combine", action="store_true",
                        help="Skip combine step")
    parser.add_argument("--skip_compare", action="store_true",
                        help="Skip compare step")
    parser.add_argument("--skip_governance", action="store_true",
                        help="Skip governance step")

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "outputs" / f"case_study_{args.run_note}_{timestamp}")

    output_dir = Path(args.output_dir)

    # Validate paths
    if not validate_paths(args):
        sys.exit(1)

    # Start execution
    print("\n" + "="*70)
    print("MULTI-SEED CASE STUDY AUTOMATION")
    print("="*70)
    print(f"CV Model Dir:    {args.cv_model_dir}")
    print(f"Input CSV:       {args.input_csv}")
    print(f"Output Dir:      {output_dir}")
    print(f"Total Budget:    {args.total_budget:,.0f} CNY")
    print(f"N Folds:         {args.n_folds}")
    print(f"N Seeds:         {args.n_seeds}")
    print(f"Total Models:    {args.n_folds * args.n_seeds}")
    print(f"Anchor Mode:     {args.anchor_mode}")
    if args.actual_csv:
        print(f"Actual CSV:      {args.actual_csv}")
    print("="*70)

    success_count = 0
    total_steps = 4

    # Step 1: Prediction
    if not args.skip_prediction:
        cmd = [
            sys.executable,
            "-m", "LSTM.Case_Study.Prediction_CS",
            "--cv_model_dir", args.cv_model_dir,
            "--input_csv", args.input_csv,
            "--output_dir", str(output_dir),
            "--total_budget", str(args.total_budget),
            "--n_folds", str(args.n_folds),
            "--n_seeds", str(args.n_seeds),
            "--device", args.device,
            "--anchor_mode", args.anchor_mode
        ]

        if args.anchor_csv:
            cmd.extend(["--anchor_csv", args.anchor_csv])

        if args.industry_config:
            cmd.extend(["--industry_config", args.industry_config])

        if run_step(
            "1. Multi-seed Prediction",
            cmd,
            f"Generate {args.n_folds * args.n_seeds} predictions using all models"
        ):
            success_count += 1
        else:
            print("\nPrediction failed! Stopping execution.")
            sys.exit(1)
    else:
        print("\n[SKIPPED] Step 1: Prediction")
        success_count += 1

    # Step 2: Combine
    if not args.skip_combine:
        cmd = [
            sys.executable,
            "-m", "LSTM.Case_Study.Combine_Ensemble_CS",
            "--case_dir", str(output_dir),
            "--n_folds", str(args.n_folds),
            "--n_seeds", str(args.n_seeds)
        ]

        if args.actual_csv:
            cmd.extend(["--actual_file", args.actual_csv])

        if run_step(
            "2. Combine Ensemble",
            cmd,
            "Merge predictions to generate P10/P50/P90 + calibration metrics"
        ):
            success_count += 1
        else:
            print("\nCombine failed! Continuing anyway...")
    else:
        print("\n[SKIPPED] Step 2: Combine")
        success_count += 1

    # Step 3: Compare
    if not args.skip_compare and args.actual_csv:
        cmd = [
            sys.executable,
            "-m", "LSTM.Case_Study.Compare_Prediction",
            "--predicted", str(output_dir / "pred_progress.csv"),
            "--actual", args.actual_csv,
            "--output_dir", str(output_dir)
        ]

        if run_step(
            "3. Compare Prediction",
            cmd,
            "Compare predictions with actual progress"
        ):
            success_count += 1
        else:
            print("\nCompare failed! Continuing anyway...")
    else:
        if args.skip_compare:
            print("\n[SKIPPED] Step 3: Compare")
        else:
            print("\n[SKIPPED] Step 3: Compare (no actual_csv provided)")
        success_count += 1

    # Step 4: Governance Triggers
    if not args.skip_governance and args.actual_csv:
        cmd = [
            sys.executable,
            "-m", "LSTM.Case_Study.Governance_Triggers",
            "--input", str(output_dir / "comparison_timeseries.csv"),
            "--output", str(output_dir / "governance_triggers.json"),
            "--soft_threshold", str(args.trigger_soft),
            "--hard_threshold", str(args.trigger_hard),
            "--min_duration", str(args.min_duration)
        ]

        if run_step(
            "4. Governance Triggers",
            cmd,
            "Analyze governance trigger points"
        ):
            success_count += 1
        else:
            print("\nGovernance analysis failed! Continuing anyway...")
    else:
        if args.skip_governance:
            print("\n[SKIPPED] Step 4: Governance")
        else:
            print("\n[SKIPPED] Step 4: Governance (no actual_csv provided)")
        success_count += 1

    # Final summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"Steps completed: {success_count}/{total_steps}")
    print(f"Output directory: {output_dir}")

    # List generated files
    if output_dir.exists():
        print("\nGenerated files:")
        key_files = [
            "pred_progress.csv",
            "calibration_metrics.json",
            "pred_progress_plot.png",
            "metrics.json",
            "governance_triggers.json"
        ]
        for filename in key_files:
            filepath = output_dir / filename
            if filepath.exists():
                print(f"{filename}")
            else:
                print(f"{filename} (not found)")

    print("="*70)

    if success_count == total_steps:
        print("\n🎉 All steps completed successfully!")
        print("\nNext steps:")
        print("1. Check pred_progress_plot.png for visualization")
        print("2. Review calibration_metrics.json for quality assessment")
        print("3. Examine governance_triggers.json for risk points")
        return 0
    else:
        print(f"\n{total_steps - success_count} step(s) had issues")
        print("Check the output above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())