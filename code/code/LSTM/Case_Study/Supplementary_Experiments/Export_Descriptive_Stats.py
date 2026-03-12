"""
Export_Descriptive_Stats.py
============================
Generates Table X (Descriptive Statistics of Synthetic Training Dataset)
for the manuscript revision, addressing Reviewer 2 Comment #9.

Reviewer request:
    "The authors should provide descriptive statistics (max, min, mean, SD)
     for all variables, including project_type, total_duration_months,
     progress_pct, mat_index, cpi_index, material_cost, labour_cost,
     equip_cost, admin_cost, and total_cost."

Outputs (all in --output_dir):
    descriptive_stats.csv           – main table: N / Mean / SD / Min / Max
    project_type_distribution.csv  – project type counts and proportions
    descriptive_stats_paper.txt    – LaTeX-ready table fragment
    data_summary.json              – machine-readable metadata

Usage:
    python Export_Descriptive_Stats.py \
        --data_csv  data/synthetic_CN_projects.csv \
        --output_dir outputs/descriptive_stats
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Variables requested by Reviewer 2 #9  (in display order)
# ---------------------------------------------------------------------------
NUMERIC_VARS = [
    ("total_duration_months", "Project Duration (months)"),
    ("progress_pct",          "Monthly Progress (%)"),
    ("mat_index",             "Material Price Index"),
    ("cpi_index",             "Consumer Price Index (CPI)"),
    ("lab_index",             "Labour Cost Index"),
    ("material_cost",         "Monthly Material Cost (CNY)"),
    ("labour_cost",           "Monthly Labour Cost (CNY)"),
    ("equip_cost",            "Monthly Equipment Cost (CNY)"),
    ("admin_cost",            "Monthly Admin/Other Cost (CNY)"),
    ("total_cost",            "Monthly Total Cost (CNY)"),
]

CATEGORICAL_VARS = [
    ("project_type", "Project Type"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Column name harmonisation
    rename = {
        "cumulative_cost_pct":  "progress_pct",
        "month_index":          "month",
        "equipment_cost":       "equip_cost",
        "admin_ratio":          "admin_cost_ratio",
        "labor_cost":           "labour_cost",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure progress_pct is in 0-100 range
    if "progress_pct" in df.columns:
        pct = pd.to_numeric(df["progress_pct"], errors="coerce")
        if pct.dropna().max() <= 1.5:
            df["progress_pct"] = pct * 100.0

    # Derive total_cost if absent
    if "total_cost" not in df.columns:
        cost_cols = [c for c in ["material_cost", "labour_cost", "equip_cost", "admin_cost"]
                     if c in df.columns]
        if cost_cols:
            df["total_cost"] = df[cost_cols].sum(axis=1)

    # Derive admin_cost if absent but admin_ratio present
    if "admin_cost" not in df.columns and "admin_cost_ratio" in df.columns \
            and "total_cost" in df.columns:
        df["admin_cost"] = df["total_cost"] * df["admin_cost_ratio"]

    return df


def _numeric_stats(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return {"N": 0, "Mean": np.nan, "SD": np.nan, "Min": np.nan,
                "Median": np.nan, "Max": np.nan}
    return {
        "N":      int(len(s)),
        "Mean":   float(s.mean()),
        "SD":     float(s.std(ddof=1)),
        "Min":    float(s.min()),
        "Median": float(s.median()),
        "Max":    float(s.max()),
    }


def _format_value(v, decimals=2, is_cost=False):
    if pd.isna(v) or not np.isfinite(v):
        return "—"
    if is_cost:
        if abs(v) >= 1e6:
            return f"{v/1e6:.{decimals}f}M"
        elif abs(v) >= 1e3:
            return f"{v/1e3:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def build_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in NUMERIC_VARS:
        if col not in df.columns:
            rows.append({
                "Variable": label,
                "Column":   col,
                "N":        "—",
                "Mean":     "—",
                "SD":       "—",
                "Min":      "—",
                "Median":   "—",
                "Max":      "—",
                "Note":     "Column not found in dataset",
            })
            continue

        stats  = _numeric_stats(df, col)
        is_cost = "cost" in col.lower()

        # Choose decimal precision
        if is_cost:
            dec = 0
        elif "index" in col.lower():
            dec = 4
        elif col == "progress_pct":
            dec = 2
        else:
            dec = 2

        rows.append({
            "Variable": label,
            "Column":   col,
            "N":        stats["N"],
            "Mean":     _format_value(stats["Mean"], dec, is_cost),
            "SD":       _format_value(stats["SD"],   dec, is_cost),
            "Min":      _format_value(stats["Min"],  dec, is_cost),
            "Median":   _format_value(stats["Median"], dec, is_cost),
            "Max":      _format_value(stats["Max"],  dec, is_cost),
            "Note":     "",
        })

    return pd.DataFrame(rows)


def build_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if "project_type" not in df.columns:
        return pd.DataFrame(columns=["Project Type", "Project Count", "Proportion (%)"])

    # Count unique projects per type
    if "project_id" in df.columns:
        type_counts = (
            df.drop_duplicates("project_id")["project_type"]
            .value_counts()
            .reset_index()
        )
    else:
        type_counts = df["project_type"].value_counts().reset_index()

    type_counts.columns = ["Project Type", "Project Count"]
    total = type_counts["Project Count"].sum()
    type_counts["Proportion (%)"] = (type_counts["Project Count"] / total * 100).round(1)
    type_counts = type_counts.sort_values("Project Count", ascending=False).reset_index(drop=True)
    return type_counts


def build_latex_table(stats_df: pd.DataFrame, n_projects: int, n_rows_total: int) -> str:
    """Generate LaTeX table fragment for the paper."""
    lines = []
    lines.append("% -------------------------------------------------------")
    lines.append("% Descriptive Statistics Table (auto-generated)")
    lines.append(f"% Dataset: {n_projects} synthetic projects, {n_rows_total:,} monthly records")
    lines.append("% -------------------------------------------------------")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Descriptive statistics of the synthetic training dataset "
                 r"(\textit{N} = monthly observations). "
                 r"Cost variables reported in CNY; index variables normalised to "
                 r"a common base period. M = million CNY; K = thousand CNY.}")
    lines.append(r"\label{tab:descriptive_stats}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Variable} & \textbf{N} & \textbf{Mean} & \textbf{SD} "
                 r"& \textbf{Min} & \textbf{Median} & \textbf{Max} \\")
    lines.append(r"\hline")

    for _, row in stats_df.iterrows():
        var   = row["Variable"].replace("&", r"\&").replace("%", r"\%")
        n_val = row["N"] if row["N"] != "—" else "—"
        line  = (f"{var} & {n_val} & {row['Mean']} & {row['SD']} "
                 f"& {row['Min']} & {row['Median']} & {row['Max']} \\\\")
        lines.append(line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics table for paper (R2 #9)"
    )
    parser.add_argument(
        "--data_csv", type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "data" / "generated" / "synthetic_CN_projects.csv"
        ),
        help="Path to synthetic_CN_projects.csv",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(
            Path(__file__).resolve().parents[3]
            / "outputs" / "descriptive_stats"
        ),
    )
    args = parser.parse_args()

    csv_path   = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: Data file not found: {csv_path}")
        print("Please specify the correct path with --data_csv")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DESCRIPTIVE STATISTICS EXPORT")
    print(f"{'='*60}")
    print(f"Input : {csv_path}")
    print(f"Output: {output_dir}")

    # Load
    df = _load_data(csv_path)
    n_rows = len(df)
    n_proj = df["project_id"].nunique() if "project_id" in df.columns else "N/A"
    print(f"\nLoaded: {n_rows:,} rows, {n_proj} unique projects")
    print(f"Columns: {list(df.columns)}")

    # ── 1. Numeric statistics table ──────────────────────────────────────
    stats_df = build_stats_table(df)
    stats_csv = output_dir / "descriptive_stats.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ Saved: {stats_csv}")

    # Pretty print to console
    print(f"\n{'─'*80}")
    print(f"{'Variable':<35} {'N':>8} {'Mean':>12} {'SD':>12} {'Min':>10} {'Max':>12}")
    print(f"{'─'*80}")
    for _, row in stats_df.iterrows():
        note = f"  [{row['Note']}]" if row["Note"] else ""
        print(f"{row['Variable']:<35} {str(row['N']):>8} {str(row['Mean']):>12} "
              f"{str(row['SD']):>12} {str(row['Min']):>10} {str(row['Max']):>12}{note}")
    print(f"{'─'*80}")

    # ── 2. Project type distribution ─────────────────────────────────────
    type_df = build_type_distribution(df)
    type_csv = output_dir / "project_type_distribution.csv"
    type_df.to_csv(type_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ Saved: {type_csv}")
    print(f"\nProject type distribution:")
    print(type_df.to_string(index=False))

    # ── 3. LaTeX table ───────────────────────────────────────────────────
    latex_str = build_latex_table(
        stats_df,
        n_projects=int(n_proj) if str(n_proj).isdigit() else 0,
        n_rows_total=n_rows,
    )
    latex_path = output_dir / "descriptive_stats_paper.txt"
    latex_path.write_text(latex_str, encoding="utf-8")
    print(f"\n✓ LaTeX table saved: {latex_path}")

    # ── 4. JSON metadata ─────────────────────────────────────────────────
    meta = {
        "generated_at":   datetime.now().isoformat(),
        "source_file":    str(csv_path),
        "n_projects":     int(n_proj) if isinstance(n_proj, (int, np.integer)) else str(n_proj),
        "n_monthly_rows": n_rows,
        "columns_found":  [col for col, _ in NUMERIC_VARS if col in df.columns],
        "columns_missing":[col for col, _ in NUMERIC_VARS if col not in df.columns],
        "reviewer_note":  "Addresses Reviewer 2 Comment #9 – Descriptive Statistics",
    }
    json_path = output_dir / "data_summary.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved: {json_path}")

    # ── 5. Training-sequence count estimate ──────────────────────────────
    # This directly addresses R2 #6/#7/#8 – report SEQUENCES not just projects
    if "total_duration_months" in df.columns and "project_id" in df.columns:
        dur = (
            df.drop_duplicates("project_id")["total_duration_months"]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
    elif "project_id" in df.columns and "month" in df.columns:
        dur = df.groupby("project_id")["month"].max()
    else:
        dur = None

    if dur is not None:
        SEQ_LEN = 12
        HORIZON = 3
        windows = dur.apply(lambda d: max(0, int(d) - SEQ_LEN - HORIZON + 1))
        total_seq = int(windows.sum())
        n_folds   = 5
        avg_train_seq_per_fold = int(total_seq * (1 - 1 / n_folds))

        print(f"\n{'='*60}")
        print("TRAINING SEQUENCE ANALYSIS")
        print(f"(For manuscript Section 3 – addresses R2 #6, #7, #8)")
        print(f"{'='*60}")
        print(f"  Total projects              : {len(dur):,}")
        print(f"  Avg project duration        : {dur.mean():.1f} months")
        print(f"  Input window                : {SEQ_LEN} months")
        print(f"  Prediction horizon          : {HORIZON} months")
        print(f"  Sliding windows per project : avg {windows.mean():.1f}")
        print(f"  Total training sequences    : {total_seq:,}")
        print(f"  Avg sequences per fold      : ~{avg_train_seq_per_fold:,}")
        print(f"\n  → Report in paper: 'The {len(dur)} synthetic projects yield")
        print(f"    approximately {total_seq:,} sliding-window training sequences")
        print(f"    in total, with ~{avg_train_seq_per_fold:,} sequences available")
        print(f"    per training fold.'")
        print(f"{'='*60}")

        meta["training_sequence_analysis"] = {
            "seq_len":                SEQ_LEN,
            "horizon":                HORIZON,
            "total_sliding_windows":  total_seq,
            "avg_windows_per_project": float(windows.mean()),
            "avg_train_seq_per_fold": avg_train_seq_per_fold,
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n✓ All outputs in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()