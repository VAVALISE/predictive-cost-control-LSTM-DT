# Predictive Cost Control: LSTM + Digital Twin

Code, data, and configuration files for the paper:

> **"A predictive–comparative framework for construction cost control using long short-term memory and digital twin technologies"**  
> *Automation in Construction* (under review)

---

## Overview

This repository implements an LSTM-based construction cost forecasting framework integrated with Digital Twin (DT) progress verification via Autodesk Construction Cloud (ACC) and Autodesk Platform Services (APS). The framework aligns contractual baselines, probabilistic LSTM cost forecasts, and DT-verified earned-value proxies on a unified monthly cumulative cost-share scale, and applies a two-tier tolerance–duration governance mechanism for automated cost risk detection.

**Three main components:**
- **LSTM forecasting**: Seq2Seq encoder–decoder with multi-seed ensemble (5-fold CV × 5 seeds = 25 models)
- **DT progress verification**: BIM-based cost-weighted earned-value proxy via APS Viewer
- **Governance layer**: Two-tier soft/hard threshold with duration persistence filter

---

## Repository Structure

```
├── data/
│   ├── generated/
│   │   └── synthetic_CN_projects.csv        # 120 synthetic projects, 3,183 records
│   └── input_csv_data/
│       ├── model_project/                   # Case study LSTM input (12-month anchor)
│       └── real_project/                    # DT progress data and contract baseline
├── code/
│   ├── LSTM/
│   │   ├── LSTM_Model.py                    # Seq2Seq LSTM architecture
│   │   ├── Train.py                         # Single model training
│   │   ├── Train_CV.py                      # 5-fold CV multi-seed training
│   │   ├── Prediction.py                    # Single model inference
│   │   ├── Prediction_Ensemble.py           # Ensemble aggregation
│   │   └── Case_Study/
│   │       ├── Run_Case_Study.py            # End-to-end pipeline
│   │       ├── Prediction_CS.py
│   │       ├── Combine_Ensemble_CS.py
│   │       ├── Compare_Prediction.py
│   │       ├── Governance_Triggers.py
│   │       ├── Sensitivity_Analysis.py
│   │       ├── Progress_Data_Received/      # BIM/APS integration scripts
│   │       └── Supplementary_Experiments/   # Baselines, ablation, learning curve
│   ├── models/
│   │   └── cv5_seeds5_stratified/           # CV outputs and config files
│   ├── ModelGenerator.py                    # Synthetic data generator
│   ├── SanityCheck.py                       # Data quality validation
│   └── generate_requirements.py             # Auto-generates requirements.txt
├── industry_config.json                     # Economic index parameters
├── requirements.txt
├── Dataset_Description.md
└── LICENSE
```

---

## Requirements

Tested environment: **Python 3.10, Windows 11**

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.8.0, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.5, statsmodels 0.14.6, prophet 1.2.1. See `requirements.txt` for all exact pinned versions.

---

## Quick Start

**Cross-validation training (25 models):**
```bash
python code/LSTM/Train_CV.py --n_folds 5 --n_seeds 5
```

**End-to-end case study pipeline:**
```bash
python code/LSTM/Case_Study/Run_Case_Study.py \
    --cv_model_dir code/models/cv5_seeds5_stratified \
    --input_csv data/input_csv_data/model_project/Preview_case_input_for_LSTM.csv \
    --actual_csv data/input_csv_data/real_project/Preview_progress_fusion.csv \
    --total_budget 40000000
```

**Reproduce experiments:**
```bash
# Ablation study (Experiment 1)
python code/LSTM/Case_Study/Supplementary_Experiments/Run_Ablation_Study.py

# Baseline comparison (Experiment 2)
python code/LSTM/Case_Study/Supplementary_Experiments/Run_Baseline_Comparison.py \
    --case_dir outputs/case_study_latest

# Descriptive statistics (Table 1)
python code/LSTM/Case_Study/Supplementary_Experiments/Export_Descriptive_Stats.py \
    --data_csv data/generated/synthetic_CN_projects.csv
```

---

## Data and Reproducibility

See [`Dataset_Description.md`](Dataset_Description.md) for full details.

- **Synthetic dataset**: 120 projects (P001–P120), 3,183 monthly records, 5 project types
- **Test set**: 24 held-out projects — see `code/models/cv5_seeds5_stratified/test_projects.json`
- **CV summary**: Per-run training results for all 25 models — see `code/models/cv5_seeds5_stratified/cv_summary.json`
- **Model config**: Representative training configuration — see `code/models/cv5_seeds5_stratified/model_config_example.json`

---

## BIM Integration (Optional)

The core LSTM forecasting and governance functionality operates independently without BIM integration. APS/ACC-based DT verification requires an Autodesk Construction Cloud subscription and API credentials. See [`Dataset_Description.md`](Dataset_Description.md) for setup details.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Citation

*To be updated upon acceptance.*
