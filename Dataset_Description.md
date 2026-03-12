# LSTM-DT Construction Cost Forecasting Dataset

## Overview

This dataset accompanies the paper "A predictive–comparative framework for construction cost control using long short-term memory and digital twin technologies." It supports a predictive cost governance system for construction projects that couples LSTM-based cost forecasting with Digital Twin (DT) progress verification through Autodesk Construction Cloud (ACC) and Autodesk Platform Services (APS).

The dataset supports researchers and practitioners in:
- Training LSTM models for multi-month construction cost forecasting
- Validating predictions against DT-verified progress data
- Implementing two-tier governance triggers for project cost risk management
- Comparing forecasting methods (LSTM vs. statistical and engineering baselines)
- Reproducing the experiments reported in the accompanying paper

---

## Dataset Contents

### 1. Synthetic Training Data

**File**: `data/generated/synthetic_CN_projects.csv` (3,183 records, 120 projects)

A synthetic multi-project time-series dataset constructed for this study to enable controlled and reproducible evaluation where large multi-project cost records are not openly available in practice. The dataset follows a simulation-driven synthetic-data design with component-level cost decomposition and channel-aligned exogenous indices.

- **Project types**: Residential, Commercial, Municipal, Industrial, Infrastructure
- **Duration range**: 18–36 months per project
- **Monthly records**: Cost breakdowns (material, labour, equipment, administration)
- **Economic indices**: CPI, PPI, MPPI, Labour cost index (chained from Aug-2022 = 100)
- **Progress tracking**: S-curve-based cumulative completion percentages
- **Data split**: 24 projects held out as test set; 96 projects used for cross-validation

**Key design principles**:
- Component-level cost decomposition with nationally representative cost-composition ratios
- Channel-aligned exogenous indices constraining simulated trajectories to plausible cost dynamics
- Project-level GroupKFold cross-validation to prevent data leakage across projects
- Five-fold CV with five seeds per fold (25 models total) to reduce optimisation variance

### 2. Case Study Data

**File**: `data/input_csv_data/model_project/Preview_case_input_for_LSTM.csv`

Real construction project input data for out-of-sample case study instantiation:
- **Project**: CN001 (Chengbei Construction Project, 24-month building project)
- **Input window**: 12 months of historical data (Month 1–12)
- **Purpose**: Anchor point for LSTM rolling prediction in the integrated workflow
- **Contains**: Cost share percentages, component ratios, cumulative progress

**File**: `data/input_csv_data/real_project/Preview_progress_fusion.csv`

DT hybrid verified progress data combining:
- BIM model analysis (component counts from Revit models via APS Viewer)
- Contract-weighted stage completion (cost-weighted earned-value proxy)
- Month-by-month hybrid DT progress percentages (APS_CumWeighted6)

**File**: `data/input_csv_data/real_project/Chengbei_24m_work.csv`

Contract baseline curve with planned progress:
- Monthly planned progress percentages
- Work breakdown by construction phase (foundation, superstructure, MEP, interior, outdoor, handover)
- Serves as contractual reference for deviation analysis in the governance layer

### 3. Model Files

**Directory**: `code/models/cv5_seeds5_stratified/`

- `model_config_example.json`: Representative per-run configuration for fold 0, seed 0. Documents model architecture (hidden size: 128, layers: 2, dropout: 0.2), input features (13 historical features, 7 future features), training hyperparameters (Adam optimiser, lr=0.001, batch size=32, MixedLoss with MSE weight=0.7 and MAE weight=0.3), and GroupKFold validation split.
- `cv_summary.json`: Aggregated cross-validation summary across all 25 fold–seed runs, including per-fold best validation loss, per-run training details, and overall statistics (mean val_loss=0.0201).
- `test_projects.json`: Held-out test set project IDs (n=24), split mode (stratified_random), and random seed (2025) for reproducibility verification.

### 4. Configuration Files

**File**: `industry_config.json`

Economic index parameters for Chinese construction industry simulation:
- CPI, PPI, MPPI, Labour wage index base values, drift, and volatility parameters
- Consistent with Table 3 in the accompanying paper

### 5. Code Files

**Core Training** (`code/LSTM/`):
- `LSTM_Model.py`: Seq2Seq encoder–decoder LSTM architecture
- `Train.py`: Single model training with early stopping and learning-rate scheduling
- `Train_CV.py`: Five-fold GroupKFold cross-validation with multi-seed training (5 seeds per fold = 25 models)
- `Prediction.py`: Single model inference
- `Prediction_Ensemble.py`: Ensemble evaluation (median, mean, weighted aggregation)

**Case Study Pipeline** (`code/LSTM/Case_Study/`):
- `Prediction_CS.py`: Multi-model prediction on real project data
- `Combine_Ensemble_CS.py`: Ensemble aggregation with P10/P50/P90 quantiles
- `Compare_Prediction.py`: Three-way comparison (LSTM vs. DT vs. baseline)
- `Governance_Triggers.py`: Two-tier tolerance–duration governance logic
- `Run_Case_Study.py`: End-to-end automation script
- `Sensitivity_Analysis.py`: Budget scaling and governance parameter robustness tests

**Supplementary Experiments** (`code/LSTM/Case_Study/Supplementary_Experiments/`):
- `Baseline_Models.py`: Statistical and engineering baselines (EVM+CPI, ARIMA, ETS, Prophet, Naïve, VAR)
- `Run_Baseline_Comparison.py`: Baseline comparison experiment (Experiment 2)
- `Run_Ablation_Study.py`: Governance layer ablation study (Experiment 1)
- `Export_Descriptive_Stats.py`: Generates descriptive statistics for the synthetic training dataset (Table 1 in the accompanying paper)
- `Run_Learning_Curve.py`: Generates training and validation learning curves (Supplementary Figure S1)

**BIM Integration** (`code/LSTM/Case_Study/Progress_Data_Received/`):
- `Progress_Adapter.py`: DT progress extraction and cost-weighted aggregation
- `Progress_Viewer.py`: APS Viewer integration for BIM component analysis
- `ACC_File_Tool.py`: ACC file management utilities
- `urn_mapper.py`: URN mapping for Autodesk model identification
- `WEB Viewer/`: Local web viewer for APS model inspection (local_server.py, viewer.html, model_links.html)

**Data Generation** (`code/`):
- `ModelGenerator.py`: Synthetic project data generator
- `SanityCheck.py`: Data quality validation
- `generate_requirements.py`: Scans project source files and generates requirements.txt automatically

---

## Technical Requirements

### Software Dependencies

See `requirements.txt` for exact pinned versions. Key dependencies:

- **Python**: 3.10
- **PyTorch**: 2.8.0
- **scikit-learn**: 1.7.2
- **pandas**: 2.3.3
- **numpy**: 2.2.5
- **statsmodels**: 0.14.6
- **prophet**: 1.2.1
- **matplotlib / seaborn**: Visualisation

**Tested environment**: Python 3.10, Windows 11

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **Minimum**: CPU-only, 8 GB RAM
- **Recommended**: NVIDIA GPU with CUDA support, 16 GB+ RAM
- **Training time (25 models)**:
  - CPU: approximately 4–8 hours
  - GPU: approximately 30–60 minutes

### BIM Integration (Optional)

The core LSTM forecasting and governance functionality operates independently without BIM integration. DT verification provides the progress evidence layer but requires:

- Autodesk Construction Cloud (ACC) subscription
- APS/Forge API credentials with OAuth 3-legged authentication
- Access to Revit models (monthly snapshots M01–M24)
- Network access to Autodesk API endpoints

---

## Quick Start

### 1. Train Models

```bash
# Cross-validation training (25 models)
python code/LSTM/Train_CV.py --n_folds 5 --n_seeds 5
```

### 2. Run Case Study

```bash
python code/LSTM/Case_Study/Run_Case_Study.py \
    --cv_model_dir code/models/cv5_seeds5_stratified \
    --input_csv data/input_csv_data/model_project/Preview_case_input_for_LSTM.csv \
    --actual_csv data/input_csv_data/real_project/Preview_progress_fusion.csv \
    --total_budget 40000000 \
    --n_folds 5 \
    --n_seeds 5
```

### 3. Reproduce Experiments

```bash
# Baseline comparison
python code/LSTM/Case_Study/Supplementary_Experiments/Run_Baseline_Comparison.py \
    --case_dir outputs/case_study_latest

# Ablation study
python code/LSTM/Case_Study/Supplementary_Experiments/Run_Ablation_Study.py

# Descriptive statistics (Table 1)
python code/LSTM/Case_Study/Supplementary_Experiments/Export_Descriptive_Stats.py \
    --data_csv data/generated/synthetic_CN_projects.csv
```

### 4. Generate Synthetic Data

```bash
python code/ModelGenerator.py
python code/SanityCheck.py
```

---

## Data Formats

### synthetic_CN_projects.csv

| Column | Description |
|--------|-------------|
| `project_id` | Unique identifier (P001–P120) |
| `project_type` | Category (residential/commercial/municipal/industrial/infrastructure) |
| `total_duration_months` | Project duration (18–36 months) |
| `month` | Current month index |
| `progress_pct` | Cumulative completion percentage (0–100%) |
| `mat_index`, `cpi_index`, `lab_index` | Economic indices (chained, normalised) |
| `material_cost`, `labour_cost`, `equip_cost`, `admin_cost` | Monthly cost components (CNY) |
| `total_cost` | Sum of cost components (CNY) |

### Preview_case_input_for_LSTM.csv

| Column | Description |
|--------|-------------|
| `project_id` | Project identifier (CN001) |
| `month_index` | Sequential month number (1–12) |
| `labour_ratio`, `material_ratio`, `equipment_ratio`, `admin_ratio` | Cost component proportions |
| `cumulative_cost_pct` | Cumulative cost share percentage |

### Preview_progress_fusion.csv

| Column | Description |
|--------|-------------|
| `Month` | Month identifier (M01–M24) |
| `APS_CumWeighted6` | DT hybrid cumulative progress (cost-weighted earned-value proxy, %) |

### Chengbei_24m_work.csv

| Column | Description |
|--------|-------------|
| `month_index` | Month number (1–24) |
| `cumulative_share_pct` | Planned cumulative progress (contractual baseline) |
| `monthly_total_share_pct` | Planned monthly progress increment |

---

## Reproducibility Notes

- The held-out test set (24 projects) is recorded in `test_projects.json` with the random seed (2025) and split mode used for the data partition reported in the paper.
- The `model_config_example.json` documents the architecture and training configuration for a representative single run (fold 0, seed 0). All 25 runs share the same architecture; fold index and random seed are the only run-specific parameters.
- Full cross-validation results across all 25 runs, including per-fold validation loss and convergence details, are provided in `cv_summary.json`.
