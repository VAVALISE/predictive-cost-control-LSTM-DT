# LSTM-BIM Digital Twin Construction Cost Forecasting Dataset

## Overview

This dataset accompanies the research paper on LSTM-based construction cost forecasting integrated with Digital Twin (DT) technology and Building Information Modeling (BIM). It provides a curated implementation of a predictive cost management system for construction projects, combining deep learning models with real-time BIM verification through Autodesk Construction Cloud (ACC).

The dataset enables researchers and practitioners to:
- Train LSTM models for multi-month construction cost prediction
- Validate predictions against BIM-verified progress data
- Implement governance triggers for project risk management
- Compare forecasting methods (LSTM vs. traditional baselines)
- Reproduce all experiments from the accompanying research paper

---

## Dataset Contents

### 1. Training Data

**File**: `data/generated/synthetic_CN_projects.csv` (838 records, 30 projects)

Synthetically generated construction project data following realistic Chinese construction industry patterns:
- **Project types**: Residential, Commercial, Municipal, Industrial, Infrastructure
- **Duration range**: 18-36 months per project
- **Monthly records**: Cost breakdowns (material, labor, equipment, administration)
- **Economic indices**: CPI, PPI, MPPI, Labor cost indices
- **Progress tracking**: S-curve based completion percentages
- **Completion phases**: Realistic modeling of project final months with reduced material/equipment costs

**Key features**:
- Realistic cost distributions by project type
- Completion phase modeling (last 2-3 months with acceptance activities)
- Time-aware features (normalized time, remaining months)
- Multi-component cost breakdown

### 2. Case Study Data

**File**: `data/real_project/Preview_case_input_for_LSTM.csv` (25 records, 12 months)

Real construction project input data for case study validation:
- **Project**: CN001 (Chengbei Construction Project)
- **Input window**: 12 months of historical data (Month 1-12)
- **Purpose**: Anchor point for LSTM rolling prediction
- **Contains**: Cost share percentages, component ratios, cumulative progress

**File**: `data/real_project/Preview_progress_fusion.csv`

DT hybrid verified progress combining:
- BIM model analysis (component counts from Revit models)
- On-site progress verification
- Weighted progress calculation (APS_CumWeighted6)
- Month-by-month actual completion percentages

**File**: `data/real_project/Chengbei_24m_work.csv` (27 records, 24 months)

Contract baseline curve with planned progress:
- Monthly planned progress percentages
- Work breakdown by construction phase
- Foundation, superstructure, MEP, interior, outdoor, handover stages
- Serves as contractual reference for deviation analysis

### 3. Configuration Files

**File**: `industry_config.json` (project root)

Economic index parameters for Chinese construction industry:
```json
{
  "CN": {
    "CPI": {"base_value": 1.0, "avg_growth": 0.002, "volatility": 0.005},
    "PPI": {"base_value": 1.0, "avg_growth": 0.003, "volatility": 0.008},
    "MPPI": {"base_value": 1.0, "avg_growth": 0.004, "volatility": 0.010},
    "Labor": {"base_value": 1.0, "avg_growth": 0.005, "volatility": 0.006}
  }
}
```

**File**: `models/model_config.json`

Model architecture and training configuration:
- Sequence length (12 months input window)
- Prediction horizon (3 months rolling forecast)
- Hidden layer dimensions (128 units, 2 layers)
- Feature definitions and scaling parameters
- Training hyperparameters (learning rate, batch size, epochs)

### 4. Code Files

**Core Training & Prediction** (`code/LSTM/`):
- `LSTM_Model.py`: Three model architectures (LSTMCostForecast, LSTMMultiOutput, LSTMSeq2SeqMass)
- `Train.py`: Single model training with early stopping and learning rate scheduling
- `Train_CV.py`: 5-fold cross-validation with multi-seed training (5 seeds per fold = 25 models)
- `Prediction.py`: Single model inference with anchoring and industry index integration
- `Prediction_Ensemble.py`: Multi-seed ensemble evaluation (median, mean, weighted strategies)

**Case Study Pipeline** (`code/Case_Study/`):
- `Prediction_CS.py`: Multi-model prediction on real project data
- `Combine_Ensemble_CS.py`: Ensemble aggregation with P10/P50/P90 quantiles
- `Compare_Prediction.py`: Three-way comparison (LSTM vs. DT vs. Baseline)
- `Governance_Triggers.py`: Automated risk detection with soft/hard thresholds
- `Run_Case_Study.py`: End-to-end automation script
- `Sensitivity_Analysis.py`: Robustness testing across parameter variations

**Experiments** (`code/experiments/`):
- `Baseline_Models.py`: Traditional forecasting methods (EVM/CPI, ARIMA, ETS, Prophet, VAR)
- `Run_Baseline_Comparison.py`: Experiment 2 - Comparative evaluation of forecasting methods
- `Run_Ablation_Study.py`: Experiment 1 - Governance layer incremental value analysis

**Data Generation** (`code/generator/`):
- `ModelGenerator.py`: Synthetic project data generator with configurable parameters
- `SanityCheck.py`: Data quality validation and integrity checks

### 5. Pre-trained Models (Optional)

**Single model** (`models/`):
- `model_config.json`: Architecture and training metadata

---

## Key Features

### 1. LSTM-Based Forecasting
- **Architecture**: Seq2Seq with attention mechanism
- **Input window**: 12 months of historical data
- **Prediction horizon**: Rolling 3-month forecast
- **Features**: Multi-variate (cost components, indices, time features, progress)
- **Uncertainty quantification**: Multi-seed ensemble with probabilistic outputs

### 2. Digital Twin Integration (Optional Component)

**Prerequisites for BIM integration**:
- Autodesk Construction Cloud (ACC) subscription
- Forge API credentials with OAuth 3-legged authentication
- Access to Revit models (24 monthly models, M01-M24)
- Network access to Autodesk API endpoints

**Note**: The core LSTM forecasting functionality works independently without BIM integration. BIM verification provides additional validation but is not required for basic model training and prediction.

**BIM capabilities** (when prerequisites are met):
- **BIM verification**: Revit model component analysis (24 monthly models, M01-M24)
- **Progress fusion**: Weighted combination of BIM counts and site verification
- **API integration**: Autodesk Construction Cloud (ACC) and Forge APIs
- **Automation**: OAuth 3-legged authentication, scheduled model updates

### 3. Governance Triggers
- **Soft threshold**: 5% cumulative deviation warning
- **Hard threshold**: 10% cumulative deviation intervention
- **Duration filter**: Minimum 2 consecutive months to avoid false alarms
- **Risk heatmap**: Visual identification of critical periods
- **Automated alerts**: Project management notifications

### 4. Comprehensive Experiments

**Experiment 1: Ablation Study** (Governance Layer Value)
- Tests 6 drift scenarios
- Compares 4 system configurations: Forecast-only, DT-only, Fusion, Fusion+Governance
- Metrics: False alarm rate, average trigger duration, detection precision
- Result: Governance reduces false alarms from 100% to 0%

**Experiment 2: Baseline Comparison** (LSTM Superiority)
- Baselines: Naive forecast, EVM/CPI, ARIMA, ETS, Prophet, VAR
- Ensures fairness: All methods use identical multivariate features
- Evaluation: MAE, RMSE, R², WAPE, Pinball loss, Interval score

**Experiment 3: Trigger Sensitivity Analysis** (Governance Robustness)
- Grid search: Tolerance (3-12%) × Duration (1-5 months)
- Validates governance rules across parameter ranges
- Identifies optimal threshold-duration combinations

---

## Technical Requirements

### Software Dependencies
- **Python**: 3.8+ (3.10+ recommended)
- **PyTorch**: 1.12+ for deep learning models
- **scikit-learn**: 1.0+ for preprocessing and metrics
- **pandas**: 1.3+ for data manipulation
- **numpy**: 1.21+ for numerical operations
- **matplotlib/seaborn**: Visualization
- **statsmodels**: For baseline time series models (ARIMA, ETS, VAR)
- **prophet**: For Facebook Prophet baseline

### Hardware Requirements
- **Minimum**: CPU-only, 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support, 16GB+ RAM
- **Training time**: 
  - Single model: ~10-20 minutes on CPU, ~2-5 minutes on GPU
  - 5-fold CV (25 models): ~4-8 hours on CPU, ~30-60 minutes on GPU

### BIM Integration (Optional)
**Note**: BIM integration is optional. Core forecasting functionality works without these components.

**Requirements** (if using BIM features):
- **Autodesk Account**: Required for ACC/Forge API access
- **OAuth Setup**: 3-legged authentication configuration
- **Revit Models**: 24 monthly models (M01-M24) for progress tracking
- **Network Access**: API endpoints must be accessible

---

## Quick Start Guide

### 1. Train Models from Scratch

```bash
# Single model training
python code/LSTM/Train.py

# Cross-validation training (25 models)
python code/LSTM/Train_CV.py --n_folds 5 --n_seeds 5
```

### 2. Run Case Study

```bash
# Complete pipeline (prediction → ensemble → comparison → governance)
python code/Case_Study/Run_Case_Study.py \
    --cv_model_dir models/cv_models \
    --input_csv data/real_project/Preview_case_input_for_LSTM.csv \
    --actual_csv data/real_project/Preview_progress_fusion.csv \
    --total_budget 40000000 \
    --n_folds 5 \
    --n_seeds 5
```

### 3. Reproduce Experiments

```bash
# Experiment 1: Ablation study
python code/experiments/Run_Ablation_Study.py

# Experiment 2: Baseline comparison
python code/experiments/Run_Baseline_Comparison.py \
    --case_dir outputs/case_study_latest
```

### 4. Generate New Synthetic Data

```bash
# Generate 50 new projects
python code/generator/ModelGenerator.py
# This creates: data/generated/synthetic_CN_projects.csv

# Validate data quality
python code/generator/SanityCheck.py
```

---

## Data Formats

### CSV Column Specifications

**synthetic_CN_projects.csv** (Training data):
- `project_id`: Unique identifier (P001-P030)
- `project_type`: Category (residential/commercial/municipal/industrial/infrastructure)
- `project_type_name`: Human-readable type name
- `total_duration_months`: Project duration (18-36 months)
- `month`: Current month (1 to total_duration_months)
- `progress_pct`: Cumulative completion percentage (0-100%)
- `is_completion_phase`: Boolean flag for final months (TRUE/FALSE)
- `mat_index`, `lab_index`, `cpi_index`: Economic indices
- `material_cost`, `labour_cost`, `equip_cost`, `admin_cost`: Cost components (CNY)
- `total_cost`: Sum of all cost components (CNY)

**Preview_case_input_for_LSTM.csv** (Case study input):
- `project_id`: Project identifier (CN001)
- `month_index`: Sequential month number (1-12)
- `month_name`: Calendar month (Sep-25, Oct-25, ...)
- `labour_ratio`, `material_ratio`, `equipment_ratio`, `admin_ratio`: Cost proportions
- `*_share_pct`: Monthly and cumulative cost shares (percentage)
- `cumulative_cost_pct`: Total completion percentage

**Preview_progress_fusion.csv** (DT verification):
- `Month`: Month identifier (M01, M02, ...)
- `APS_CumWeighted6`: DT hybrid cumulative progress (percentage string, e.g., "45.2%")
- Additional columns: Component counts, weighted scores

**Chengbei_24m_work.csv** (Contract baseline):
- `project_id`, `month_index`, `month_name`: Identifiers
- `*_share_pct`: Phase-specific progress (foundation, superstructure, equipment, interior, outdoor, handover)
- `monthly_total_share_pct`: Planned monthly progress increment
- `cumulative_share_pct`: Planned cumulative progress
- `work_*`: Textual descriptions of planned activities
