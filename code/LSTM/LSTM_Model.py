import torch
import torch.nn as nn
import numpy as np

# ===============================
# Preserve existing LSTMCostForecast / LSTMMultiOutput
# (If using "full replacement" approach, copy the original two classes here or keep the preserved version below)
# ===============================


# ========= Optional: LSTM forget gate small initialization for stable convergence =========
def _init_lstm_forget_bias(lstm_module: nn.LSTM, bias_value: float = 1.0):
    """
     Initialize LSTM forget gate bias to a positive value to help stabilize training.
    """
    for names in lstm_module._all_weights:
        for name in filter(lambda x: 'bias' in x, names):
            bias = getattr(lstm_module, name)
            n = bias.size(0)
            start, end = n // 4, n // 2  # i, f, g, o -> 选择 forget gate 段
            with torch.no_grad():
                bias[start:end].fill_(bias_value)


# ===============================
# New: Seq2Seq + Residual + Tail Mass Conservation Monthly Cost Model
# ===============================
class LSTMSeq2SeqMass(nn.Module):
    """
    Seq2Seq model: Historical covariates -> Future covariates driven 12-month monthly cost prediction
    - Encoder: Historical L months (in_hist)
    - Decoder: Future H months covariates (in_fut) + encoder final state
    - Residual A1 (historical cost residual): Recent 3 months historical features via MLP regression to monetary baseline
    - Residual A2 (planned share residual): Direct learning of "planned baseline" from future covariate features
    - Output: dict containing p50 and components, training and evaluation use p50
    """

    def __init__(
        self,
        in_hist: int,
        in_fut: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 12,
        use_hist_residual: bool = True,   # A1
        use_plan_residual: bool = True    # A2
    ):
        super().__init__()
        self.horizon = horizon
        self.use_hist_residual = use_hist_residual
        self.use_plan_residual = use_plan_residual

        # Encoder: Historical L×in_hist -> H
        self.enc = nn.LSTM(
            input_size=in_hist,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        _init_lstm_forget_bias(self.enc)

        # Future covariate mapping to hidden space
        self.fut_proj = nn.Linear(in_fut, hidden_size)

        # Decoder: Month-by-month feeding [fut_h, h_last]
        self.dec = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        _init_lstm_forget_bias(self.dec)

        self.drop = nn.Dropout(dropout)

        # Main output head (incremental term)
        self.head_p50_core = nn.Linear(hidden_size, 1)

        # A1: Historical cost residual (recent 3 months historical features -> monetary baseline)
        if use_hist_residual:
            self.res_hist = nn.Sequential(
                nn.Linear(in_hist, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

        # A2: Planned share residual (planned baseline from future features)
        if use_plan_residual:
            self.head_plan = nn.Linear(hidden_size, 1)

    def forward(self, x_hist: torch.Tensor, x_fut: torch.Tensor):
        """
        Args:
            x_hist: (B, L, in_hist)  Historical covariates (actual known), e.g., CPI/material/labour index, time position, etc.
            x_fut : (B, H, in_fut)   Future covariates (planned/a priori), e.g., month sequence, remaining_months, planned share, index forecast, etc.
        Returns:
            dict: {
                'p50': (B,H),
                'p50_core': (B,H),
                'plan_baseline': (B,H) or None,
                'hist_resid': (B,H) or None
            }
        """
        B, L, Fh = x_hist.shape
        B2, H, Ff = x_fut.shape
        assert B == B2, "x_hist/x_fut batch 不一致"

        # 1) Encode history
        enc_out, _ = self.enc(x_hist)         # (B,L,Hs)
        h_last = enc_out[:, -1, :]            # (B,Hs)

        # 2) Future features -> hidden space
        fut_h = self.fut_proj(x_fut)          # (B,H,Hs)

        # 3) Decoder input: concatenate [fut_h, h_last]
        h_rep = h_last.unsqueeze(1).expand(-1, H, -1)   # (B,H,Hs)
        dec_in = torch.cat([fut_h, h_rep], dim=-1)      # (B,H,2*Hs)
        dec_out, _ = self.dec(dec_in)                   # (B,H,Hs)
        dec_out = self.drop(dec_out)

        # 4) Main output (incremental term)
        p50_core = self.head_p50_core(dec_out).squeeze(-1)  # (B,H)

        # 5) A2 planned baseline (optional)
        plan_baseline = None
        if self.use_plan_residual:
            plan_baseline = self.head_plan(fut_h).squeeze(-1)  # (B,H)

        # 6) A1 historical residual (optional): recent 3 months historical features average
        hist_resid = None
        if self.use_hist_residual:
            hist_pool = x_hist[:, -min(3, L):, :].mean(dim=1)  # (B,Fh)
            hist_resid = self.res_hist(hist_pool)              # (B,1)
            hist_resid = hist_resid.expand(-1, H)              # (B,H)

        # 7) Final output: core + (optional) two residuals
        p50 = p50_core
        if plan_baseline is not None:
            p50 = p50 + plan_baseline
        if hist_resid is not None:
            p50 = p50 + hist_resid

        return {
            'p50': p50,
            'p50_core': p50_core,
            'plan_baseline': plan_baseline,
            'hist_resid': hist_resid
        }


# ===============================
#  New: Tail "mass conservation" loss
# ===============================
def mass_conservation_loss(y_hat: torch.Tensor, y_true: torch.Tensor, reduction: str = "mean"):
    """
    Constraint: future H months predicted total ≈ actual total
    Args:
        y_hat: (B,H)  Predicted monthly cost (monetary domain; if trained in log domain, expm1 first before feeding to this loss)
        y_true: (B,H) Actual monthly cost (monetary domain)
    """
    # Mass conservation over future window within batch
    diff = (y_hat.sum(dim=1) - y_true.sum(dim=1))  # (B,)
    loss = diff.pow(2)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# ===============================
# (Preserved) Single-output LSTM (unchanged)
# ===============================
class LSTMCostForecast(nn.Module):
    """
    LSTM-based model for construction project cost forecasting
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMCostForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def predict_sequence(self, x, future_steps=3):
        self.eval()
        predictions = []
        with torch.no_grad():
            current_input = x.clone()
            for _ in range(future_steps):
                pred = self.forward(current_input)
                predictions.append(pred)
                new_step = torch.zeros(current_input.size(0), 1, current_input.size(2))
                new_step[:, 0, 0] = pred.squeeze()
                current_input = torch.cat([current_input[:, 1:, :], new_step], dim=1)
        return predictions


# ===============================
# (Preserved) Multi-output LSTM (unchanged)
# ===============================
class LSTMMultiOutput(nn.Module):
    """
    LSTM model for predicting multiple cost components simultaneously
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_components=4):
        super(LSTMMultiOutput, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

        self.material_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.labour_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.equipment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.admin_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)

        material_pred = self.material_head(out)
        labour_pred = self.labour_head(out)
        equipment_pred = self.equipment_head(out)
        admin_pred = self.admin_head(out)
        return {
            'material': material_pred,
            'labour': labour_pred,
            'equipment': equipment_pred,
            'admin': admin_pred,
            'total': material_pred + labour_pred + equipment_pred + admin_pred
        }

def get_model_summary(model,
                      input_size_hist=None,   # (L, in_hist)
                      input_size_fut=None,    # (H, in_fut)
                      device='cpu'):
    """
     Model summary printing tool compatible with Seq2Seq.
    Usage (Seq2Seq): get_model_summary(model,
                         input_size_hist=(L, in_hist),
                         input_size_fut=(H, in_fut))
    If only single input passed (old version habit), only print parameter count.
    """
    import torch

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"Total params:     {n_params:,}")
    print(f"Trainable params: {n_trainable:,}")

    # If dual input sizes given, do a dry run and print output shape
    if input_size_hist is not None and input_size_fut is not None:
        L, in_hist = input_size_hist
        H, in_fut  = input_size_fut
        model.eval()
        with torch.no_grad():
            x_hist = torch.randn(1, L, in_hist, device=device)
            x_fut  = torch.randn(1, H, in_fut,  device=device)
            out = model(x_hist, x_fut)
        if isinstance(out, dict) and 'p50' in out:
            print(f"Forward OK → p50 shape: {tuple(out['p50'].shape)}")
        else:
            print("Forward OK (no 'p50' in output dict).")
    else:
        print("Skip dry-run (no input sizes provided).")
    print("=" * 60 + "\n")


# ===============================
# (Optional) Quick self-test
# ===============================
if __name__ == "__main__":
    B, L, H = 2, 12, 12
    in_hist, in_fut = 13, 10
    model = LSTMSeq2SeqMass(in_hist=in_hist, in_fut=in_fut, hidden_size=128, horizon=H)
    x_hist = torch.randn(B, L, in_hist)
    x_fut = torch.randn(B, H, in_fut)
    out = model(x_hist, x_fut)
    print("p50 shape:", out['p50'].shape)
