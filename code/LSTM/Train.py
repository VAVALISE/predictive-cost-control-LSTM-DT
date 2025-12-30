import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os

# Import model
from LSTM_Model import LSTMCostForecast, get_model_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "generated" / "synthetic_CN_projects.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# Enhanced Feature Engineering
# ===========================
def add_enhanced_features(df):
    """
    Add time-aware and completion-phase features

    Args:
        df: DataFrame with project data

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Calculate per-project total duration and add features
    if 'is_completion_phase' not in df.columns:
        df['is_completion_phase'] = 0.0
    else:
        df['is_completion_phase'] = df['is_completion_phase'].astype(float)

    df['normalized_time'] = 0.0
    df['remaining_months'] = 0.0

    # --- Write in separate items, using index alignment ---
    for pid, grp in df.groupby('project_id'):
        idx = grp.index
        total = int(grp['month'].max())

        df.loc[idx, 'normalized_time'] = grp['month'].to_numpy() / max(total, 1)
        df.loc[idx, 'remaining_months'] = total - grp['month'].to_numpy()

        is_comp = ((total - grp['month'].to_numpy()) <= 2) | (grp['progress_pct'].to_numpy() > 90)
        df.loc[idx, 'is_completion_phase'] = is_comp.astype(float)
    return df


# ===========================
# Mixed Loss Function
# ===========================
class MixedLoss(nn.Module):
    """
    Combined MSE + MAE loss for more robust training
    """

    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        super(MixedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        return self.mse_weight * mse_loss + self.mae_weight * mae_loss


# ===========================
# Seq2Seq Dataset
# ===========================
class Seq2SeqDataset(Dataset):
"""
Construct sliding window samples for each project:
    x_hist: (L, in_hist) Features from the past L months
    x_fut: (H, in_fut) Covariates for the next H months (known/predictable)
    y_seq: (H,) Goals for the next H months (here using total_cost, standardized after log1p)
"""
    def __init__(self, data, sequence_length=12, prediction_horizon=12,
                 hist_feature_cols=None, fut_feature_cols=None,
                 target_col='total_cost'):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_col = target_col

        # Historical features column (consistent with your training).
        self.hist_feature_cols = hist_feature_cols or [
            'progress_pct', 'mat_index', 'lab_index', 'cpi_index',
            'material_cost', 'labour_cost', 'equip_cost', 'admin_cost',
            'total_cost', 'month', 'normalized_time',
            'remaining_months', 'is_completion_phase'
        ]

        # Future covariates (excluding target; emphasizing "time + macroeconomic index + stage marker")
        self.fut_feature_cols = fut_feature_cols or [
            'month', 'normalized_time', 'remaining_months',
            'is_completion_phase', 'cpi_index', 'mat_index', 'lab_index'
        ]

        self.samples = []
        for pid, proj in data.groupby('project_id'):
            proj = proj.sort_values('month')
            n = len(proj)
            L, H = sequence_length, prediction_horizon
            if n < L + H:
                continue

            X_hist_all = proj[self.hist_feature_cols].to_numpy()
            X_fut_all  = proj[self.fut_feature_cols].to_numpy()
            y_all      = proj[self.target_col].to_numpy()

            # Sliding window: History [i, i+L), Future [i+L, i+L+H)
            for i in range(0, n - (L + H) + 1):
                x_hist = X_hist_all[i:i+L, :]
                x_fut  = X_fut_all[i+L:i+L+H, :]
                y_seq  = y_all[i+L:i+L+H]

                self.samples.append((x_hist, x_fut, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xh, xf, y = self.samples[idx]
        return (
            torch.FloatTensor(xh),           # (L, in_hist)
            torch.FloatTensor(xf),           # (H, in_fut)
            torch.FloatTensor(y)             # (H,)
        )

ProjectCostDataset = Seq2SeqDataset


# ===========================
# Modified prepare_data with enhanced features and log transform
# ===========================
def prepare_data_enhanced(data_path, sequence_length=12, train_ratio=0.8,
                          batch_size=32, use_log_transform=True,
                          prediction_horizon=12):
    """
    Enhanced data preparation:
        • Add `normalized_time`, `remaining_months`, and `is_completion_phase`.
        • Standardize `total_cost` by first logging it to log1p, then normalize it as part of the feature/objective.
        • Use a single `StandardScaler` to fit the joint column of historical columns ∪ future columns (to maintain consistency).
        • Construct a `Seq2SeqDataset`: Historical L -> Future H (default H=12).
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION (Seq2Seq, ENHANCED)")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {df['project_id'].nunique()} projects")


    df = add_enhanced_features(df)
    print("Added: normalized_time, remaining_months, is_completion_phase")

    if use_log_transform:
        df['total_cost'] = np.log1p(df['total_cost'])
        print("Applied log1p to total_cost")

    all_projects = df['project_id'].unique()
    n_train = int(len(all_projects) * train_ratio)
    train_projects = all_projects[:n_train]
    val_projects   = all_projects[n_train:]

    train_df = df[df['project_id'].isin(train_projects)].copy()
    val_df   = df[df['project_id'].isin(val_projects)].copy()

    # Historical/Future Columns (to maintain consistency with the dataset)
    hist_feature_cols = [
        'progress_pct', 'mat_index', 'lab_index', 'cpi_index',
        'material_cost', 'labour_cost', 'equip_cost', 'admin_cost',
        'total_cost', 'month', 'normalized_time',
        'remaining_months', 'is_completion_phase'
    ]
    fut_feature_cols = [
        'month', 'normalized_time', 'remaining_months',
        'is_completion_phase', 'cpi_index', 'mat_index', 'lab_index'
    ]

    # Uniform fitting scaler (joint columns)
    union_cols = sorted(set(hist_feature_cols) | set(fut_feature_cols))
    scaler = StandardScaler()

    for c in union_cols:
        train_df[c] = train_df[c].astype(float)
        val_df[c]   = val_df[c].astype(float)

    train_df[union_cols] = scaler.fit_transform(train_df[union_cols])
    val_df[union_cols]   = scaler.transform(val_df[union_cols])

    # Construct Seq2SeqDataset
    train_dataset = Seq2SeqDataset(
        train_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col='total_cost'
    )
    val_dataset = Seq2SeqDataset(
        val_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        hist_feature_cols=hist_feature_cols,
        fut_feature_cols=fut_feature_cols,
        target_col='total_cost'
    )

    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return (train_loader, val_loader, scaler,
            hist_feature_cols, fut_feature_cols, use_log_transform, prediction_horizon)


# ===========================
# Modified train_model with mixed loss
# ===========================
def train_model_enhanced(model, train_loader, val_loader, num_epochs=100,
                         learning_rate=0.001, device='cpu',
                         save_path='models/lstm_cost_forecast.pth',
                         use_mixed_loss=True):

    print("\n" + "=" * 60)
    print("MODEL TRAINING (Seq2Seq, ENHANCED)")
    print("=" * 60)

    model = model.to(device)
    criterion = MixedLoss(0.7, 0.3) if use_mixed_loss else nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    best_val = float('inf')
    patience, max_patience = 0, 20

    print("\n" + "-"*60)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val':<15}")
    print("-"*60)

    for epoch in range(num_epochs):
        model.train()
        tr_losses = []
        for x_hist, x_fut, y_seq in train_loader:
            x_hist = x_hist.to(device)     # (B,L, in_hist)
            x_fut  = x_fut.to(device)      # (B,H, in_fut)
            y_seq  = y_seq.to(device)      # (B,H)

            optimizer.zero_grad()
            out = model(x_hist, x_fut)     # dict: {'p50': (B,H), 'p90': (B,H) ...}
            pred = out['p50']

            loss = criterion(pred, y_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_losses.append(loss.item())

        # ---- val ----
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x_hist, x_fut, y_seq in val_loader:
                x_hist = x_hist.to(device)
                x_fut  = x_fut.to(device)
                y_seq  = y_seq.to(device)
                pred   = model(x_hist, x_fut)['p50']
                loss   = criterion(pred, y_seq)
                va_losses.append(loss.item())

        tr, va = np.mean(tr_losses), np.mean(va_losses)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(va)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            print(f"LR: {prev_lr:.6g} → {cur_lr:.6g}")

        history['train_loss'].append(tr)
        history['val_loss'].append(va)
        history['learning_rate'].append(cur_lr)

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {tr:<15.6f} {va:<15.6f} {best_val:<15.6f}")

        if va < best_val:
            best_val = va
            patience = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': tr,
                'val_loss': va,
            }, save_path)
        else:
            patience += 1

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("-"*60)
    print(f"Training completed! Best val loss: {best_val:.6f}")
    return history



# ===========================
# Plot Training History
# ===========================
def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training and validation loss curves

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1]
    ax.plot(history['learning_rate'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


# ===========================
# Main Training Pipeline
# ===========================
def main():
    config = {
        'data_path': str(DATA_PATH),
        'model_save_path': str(MODELS_DIR / 'lstm_cost_forecast.pth'),
        'history_save_path': str(MODELS_DIR / 'training_history.png'),
        'sequence_length': 12,
        'prediction_horizon': 12,
        'train_ratio': 0.8,
        'batch_size': 32,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'use_log_transform': True,
        'use_mixed_loss': True,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    (train_loader, val_loader, scaler,
     hist_cols, fut_cols, use_log, horizon) = prepare_data_enhanced(
        data_path=config['data_path'],
        sequence_length=config['sequence_length'],
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size'],
        use_log_transform=config['use_log_transform'],
        prediction_horizon=config['prediction_horizon']
    )

    from LSTM_Model import LSTMSeq2SeqMass, get_model_summary
    model = LSTMSeq2SeqMass(
        in_hist=len(hist_cols),
        in_fut=len(fut_cols),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        horizon=horizon
    )

    try:
        get_model_summary(
            model,
            input_size_hist=(config['sequence_length'], len(hist_cols)),
            input_size_fut=(config['prediction_horizon'], len(fut_cols))
        )
    except:
        pass

    history = train_model_enhanced(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device,
        save_path=config['model_save_path'],
        use_mixed_loss=config['use_mixed_loss']
    )

    plot_training_history(history, save_path=config['history_save_path'])

    import pickle, json
    scaler_path = MODELS_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    cfg = dict(config)
    cfg.update({
        'use_log_transform': use_log,
        'hist_feature_cols': hist_cols,
        'fut_feature_cols': fut_cols
    })
    with open(MODELS_DIR / 'model_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Configuration saved to: {MODELS_DIR / 'model_config.json'}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)



if __name__ == "__main__":
    main()