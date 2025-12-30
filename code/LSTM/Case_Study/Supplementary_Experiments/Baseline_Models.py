"""
Baseline_Models.py - Experiment 2: Stronger Baseline Comparison
=========================================
New Baselines:
A. EVM/CPI-based forecast (Engineering domain standard)- explicit formula
B. ETS / ARIMA / Prophet (Time series standard)

Metrics:
- Point forecast:MAE, RMSE, R²
- Probabilistic forecast:pinball loss, interval score, coverage, mean interval width
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================================
# PART 1: Evaluation Metrics (including probabilistic forecast metrics)
# ===========================================================================================

def calculate_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Point forecast metrics

    - MAE: Mean Absolute Error (percentage points)
    - RMSE: Root Mean Square Error (percentage points)
    - R²: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true > 0) else np.nan

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
        Pinball Loss (Quantile Loss)

        Formula:L_τ(y, q) = τ(y - q) if y >= q else (1-τ)(q - y)

        Args:
            y_true: actual values
            y_pred: quantile predictions
            quantile: quantile (e.g., 0.1 for P10, 0.9 for P90)
        """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_true - y_pred
    loss = np.where(errors >= 0, quantile * errors, (1 - quantile) * (-errors))

    return np.mean(loss)


def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float = 0.2) -> float:
    """
    Interval Score (Winkler Score)
    Evaluate prediction interval quality. For(1-α)confidence interval[L, U]：
    Formula:IS = (U - L) + (2/α)(L - y)·I(y < L) + (2/α)(y - U)·I(y > U)
    Lower is better: narrow interval that covers actual values

    Args:
        y_true: actual values
        lower: interval lower bound (e.g., P10)
        upper: interval upper bound (e.g., P90)
        alpha: significance level (default 0.2 for 80% interval)
    """
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)

    width = upper - lower
    penalty_lower = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_upper = (2 / alpha) * np.maximum(y_true - upper, 0)

    score = width + penalty_lower + penalty_upper
    return np.mean(score)


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Empirical Coverage
    Proportion of actuals falling within prediction interval
    Goal:for80%interval，Expected coverage approaches0.8
    """
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)

    covered = np.sum((y_true >= lower) & (y_true <= upper))
    return covered / len(y_true)


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width (Narrower is better, given adequate coverage)"""
    return np.mean(upper - lower)


def calculate_probabilistic_metrics(y_true: np.ndarray, p10: np.ndarray,
                                     p50: np.ndarray, p90: np.ndarray) -> dict:
    """
    calculate all probabilisticprediction metrics

    Args:
        y_true: actual values
        p10: P10 prediction (10% quantile)
        p50: P50 prediction (median/point prediction)
        p90: P90 prediction (90% quantile)
    """
    # Point forecast metrics (based on P50)
    point = calculate_point_metrics(y_true, p50)

    # Pinball loss
    pl_10 = pinball_loss(y_true, p10, 0.1)
    pl_50 = pinball_loss(y_true, p50, 0.5)
    pl_90 = pinball_loss(y_true, p90, 0.9)

    # 80%interval [P10, P90]
    is_80 = interval_score(y_true, p10, p90, alpha=0.2)
    cov_80 = empirical_coverage(y_true, p10, p90)
    miw_80 = mean_interval_width(p10, p90)

    return {
        **point,
        'Pinball_P10': pl_10,
        'Pinball_P50': pl_50,
        'Pinball_P90': pl_90,
        'Pinball_Avg': (pl_10 + pl_50 + pl_90) / 3,
        'IntervalScore_80': is_80,
        'Coverage_80': cov_80,
        'MeanWidth_80': miw_80
    }


# ===========================================================================================
# PART 2: Baseline models - Naive
# ===========================================================================================

class NaiveBaseline:
    """Naive baseline"""

    def __init__(self, method='last', window=3):
        self.method = method
        self.window = window
        self.name = f"Naive_{method}" if method == 'last' else f"Naive_MA{window}"

    def predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        train = np.array(train)
        if self.method == 'last':
            return np.full(horizon, train[-1])
        else:  # moving_average
            w = min(self.window, len(train))
            return np.full(horizon, np.mean(train[-w:]))

    def predict_quantiles(self, train: np.ndarray, horizon: int,
                          std_factor: float = 0.1) -> tuple:
        """Naive does not provide true interval prediction, use fixed std to approximate"""
        p50 = self.predict(train, horizon)
        std = np.std(train) * std_factor
        p10 = p50 - 1.28 * std
        p90 = p50 + 1.28 * std
        return p10, p50, p90

class Naive_Delta:
    def __init__(self):
        self.name = "Naive_Delta"

    def predict(self, train, horizon):
        train = np.array(train)
        last_delta = train[-1] - train[-2] if len(train) >= 2 else 0.0
        pred = train[-1] + np.arange(1, horizon+1) * last_delta
        return np.maximum.accumulate(np.clip(pred, 0, 100))

    def predict_quantiles(self, train, horizon):
        p50 = self.predict(train, horizon)
        std = np.std(np.diff(train)) if len(train) >= 3 else np.std(train)
        p10 = p50 - 1.28 * std
        p90 = p50 + 1.28 * std
        return np.clip(p10, 0, p50), p50, np.clip(p90, p50, 150)

# ===========================================================================================
# PART 3: Baseline models - ARIMA
# ===========================================================================================

class ARIMAModel:
    """ARIMATime series model"""

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.name = f"ARIMA{order}"

    def predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train, order=self.order)
            fitted = model.fit()
            pred = fitted.forecast(steps=horizon)
            return np.array(pred)
        except Exception as e:
            print(f"  ARIMA failed: {e}")
            return np.full(horizon, train[-1])

    def predict_quantiles(self, train: np.ndarray, horizon: int) -> tuple:
        """ARIMA interval prediction"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train, order=self.order)
            fitted = model.fit()
            forecast = fitted.get_forecast(steps=horizon)
            p50 = forecast.predicted_mean.values
            conf = forecast.conf_int(alpha=0.2)  # 80% CI
            p10 = conf.iloc[:, 0].values
            p90 = conf.iloc[:, 1].values
            return p10, p50, p90
        except:
            p50 = self.predict(train, horizon)
            std = np.std(train) * 0.1
            return p50 - 1.28*std, p50, p50 + 1.28*std


# ===========================================================================================
# PART 4: Baseline models - EVM/CPI (Engineering domain standard)
# ===========================================================================================

class EVMForecast:
    """
    Earned Value Management / CPI-based Forecast
    Engineering domain standard cost predictio nmethod.

    coreFormula:
    1. CPI (Cost Performance Index) = EV / AC
       - EV (Earned Value): alreadyPlanned Value of completed work (≈ DT verified progress × BAC)
       - AC (Actual Cost): Actual cumulative cost
    2. EAC (Estimate at Completion) = BAC / CPI
       - by whenbefore CPI trendpredictionfinal cost
    3. ETC (Estimate to Complete) = EAC - AC
       - remaining costprediction
    4. Convert to cumulative percentage trajectory：
       predicted_share(t) = AC(t) + (progress_remaining(t) / CPI_avg) × rate_factor

    inputdepend on：historical actualcost + baselineplanned + DT verified progress
    """

    def __init__(self, method='cpi_trend', smoothing=3):
        """
        Args:
            method: 'cpi_simple' (whenbeforeCPI) or 'cpi_trend' (CPI trend)
            smoothing: smoothwindow
        """
        self.method = method
        self.smoothing = smoothing
        self.name = f"EVM_{method}"

    def calculate_cpi_series(self, actual_cost: np.ndarray,
                              planned_value: np.ndarray,
                              earned_value: np.ndarray = None) -> np.ndarray:
        """
        calculateCPItime series
        CPI = EV / AC
        if EV not yet provide，assumeEV ≈ PV (planned=actual progress)
        """
        if earned_value is None:
            earned_value = planned_value.copy()

        cpi = np.zeros(len(actual_cost))
        for i in range(len(actual_cost)):
            if actual_cost[i] > 0:
                cpi[i] = earned_value[i] / actual_cost[i]
            else:
                cpi[i] = 1.0

        return cpi

    def predict(self, train_cost: np.ndarray, baseline: np.ndarray,
                horizon: int, bac: float = 100.0) -> np.ndarray:
        """
        EVM prediction

        Args:
            train_cost: historicalActual cumulative cost (%)
            baseline: planned value (cumulative %)
            horizon: prediction steps long
            bac: Budget at Completion (default100%)

        Returns:
            predictionCumulative cost trajectory (%)
        """
        n_train = len(train_cost)

        # Step 1: calculateCPIsequence
        # assumeEV ≈ baseline (planned progress = earnedvalue)
        cpi_series = self.calculate_cpi_series(train_cost, baseline[:n_train])

        # Step 2: determine prediction useCPI
        if self.method == 'cpi_simple':

            cpi_forecast = cpi_series[-1] if cpi_series[-1] > 0 else 1.0
        else:

            w = min(self.smoothing, n_train)
            recent_cpi = cpi_series[-w:]
            weights = np.arange(1, w + 1)
            cpi_forecast = np.average(recent_cpi, weights=weights)
            if cpi_forecast <= 0:
                cpi_forecast = 1.0

        # Step 3: prediction not yetto cost
        predictions = []
        last_cost = train_cost[-1]

        # extrapolation baseline obtain not yet to planned value
        if len(baseline) > n_train:
            future_baseline = baseline[n_train:n_train + horizon]
        else:
            # linear extrapolation
            slope = (baseline[-1] - baseline[-2]) if len(baseline) > 1 else 0
            future_baseline = np.array([baseline[-1] + slope * (i+1) for i in range(horizon)])

        for h in range(horizon):
            # EAC principle：prediction cost = planned cost / CPI
            if h < len(future_baseline):
                planned = future_baseline[h]
            else:
                planned = future_baseline[-1] if len(future_baseline) > 0 else bac

            # ifCPI < 1，cost will overrun；CPI > 1，cost will be saved
            predicted = planned / cpi_forecast

            # ensuremonotonicincreasing
            predicted = max(predicted, last_cost)
            predictions.append(predicted)
            last_cost = predicted

        return np.clip(predictions, 0, 150)

    def predict_quantiles(self, train_cost: np.ndarray, baseline: np.ndarray,
                          horizon: int) -> tuple:
        """EVM interval prediction (based on CPIdoes notcertainty)"""
        p50 = self.predict(train_cost, baseline, horizon)

        # based on historical CPI variability estimate interval
        cpi_series = self.calculate_cpi_series(train_cost, baseline[:len(train_cost)])
        cpi_std = np.std(cpi_series) if len(cpi_series) > 1 else 0.1

        # pessimistic CPI (lower)→ higher cost
        cpi_low = max(np.mean(cpi_series) - 1.28 * cpi_std, 0.5)
        # optimistic CPI (higher)→ lower cost
        cpi_high = np.mean(cpi_series) + 1.28 * cpi_std

        p90 = p50 * (np.mean(cpi_series) / cpi_low) if cpi_low > 0 else p50 * 1.2
        p10 = p50 * (np.mean(cpi_series) / cpi_high) if cpi_high > 0 else p50 * 0.8

        return np.array(p10), np.array(p50), np.array(p90)


# ===========================================================================================
# PART 5: Baseline models - ETS (HoltExponential Smoothing)
# ===========================================================================================

class ETSModel:

    def __init__(self, damped=True):
        self.damped = damped
        self.name = f"ETS_Holt{'_damped' if damped else ''}"

    def predict(self, train: np.ndarray, horizon: int) -> np.ndarray:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(train, trend='add', damped_trend=self.damped, seasonal=None)
            fitted = model.fit()
            return fitted.forecast(horizon)
        except Exception as e:
            print(f"  ETS failed: {e}")
            return self._manual_holt(train, horizon)

    def _manual_holt(self, train: np.ndarray, horizon: int,
                     alpha: float = 0.8, beta: float = 0.2) -> np.ndarray:

        level = train[0]
        trend = train[1] - train[0] if len(train) > 1 else 0
        phi = 0.9 if self.damped else 1.0

        for i in range(1, len(train)):
            new_level = alpha * train[i] + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            level, trend = new_level, new_trend

        predictions = []
        for h in range(1, horizon + 1):
            if self.damped:
                pred = level + sum(phi**i for i in range(1, h+1)) * trend
            else:
                pred = level + h * trend
            predictions.append(pred)

        return np.array(predictions)

    def predict_quantiles(self, train: np.ndarray, horizon: int) -> tuple:
        """ETSinterval prediction"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(train, trend='add', damped_trend=self.damped, seasonal=None)
            fitted = model.fit()

            # statsmodels ETS no directinterval prediction，Use residualestimate
            residuals = train - fitted.fittedvalues
            std = np.std(residuals)

            p50 = fitted.forecast(horizon)
            p10 = p50 - 1.28 * std
            p90 = p50 + 1.28 * std
            return p10, p50, p90
        except:
            p50 = self.predict(train, horizon)
            std = np.std(train) * 0.1
            return p50 - 1.28*std, p50, p50 + 1.28*std


# ===========================================================================================
# PART 6: Baseline models - Prophet
# ===========================================================================================

class ProphetModel:
    """Facebook Prophet"""

    def __init__(self):
        self.name = "Prophet"

    def predict(self, train: np.ndarray, horizon: int,
                start_date: str = '2024-01-01') -> np.ndarray:
        try:
            from prophet import Prophet

            dates = pd.date_range(start=start_date, periods=len(train), freq='MS')
            df = pd.DataFrame({'ds': dates, 'y': train})

            model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                           daily_seasonality=False, changepoint_prior_scale=0.05)
            model.fit(df)

            future = model.make_future_dataframe(periods=horizon, freq='MS')
            forecast = model.predict(future)

            return forecast['yhat'].values[-horizon:]
        except ImportError:
            print("  Prophet not installed, using ETS fallback")
            return ETSModel().predict(train, horizon)
        except Exception as e:
            print(f"  Prophet failed: {e}")
            return ETSModel().predict(train, horizon)

    def predict_quantiles(self, train: np.ndarray, horizon: int,
                          start_date: str = '2024-01-01') -> tuple:
        """ProphetBuilt-in interval prediction"""
        try:
            from prophet import Prophet

            dates = pd.date_range(start=start_date, periods=len(train), freq='MS')
            df = pd.DataFrame({'ds': dates, 'y': train})

            model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                           daily_seasonality=False, interval_width=0.8)
            model.fit(df)

            future = model.make_future_dataframe(periods=horizon, freq='MS')
            forecast = model.predict(future)

            p50 = forecast['yhat'].values[-horizon:]
            p10 = forecast['yhat_lower'].values[-horizon:]
            p90 = forecast['yhat_upper'].values[-horizon:]
            return p10, p50, p90
        except:
            p50 = self.predict(train, horizon)
            std = np.std(train) * 0.1
            return p50 - 1.28*std, p50, p50 + 1.28*std


# ===========================================================================================
# PART 7: Helper function
# ===========================================================================================

def get_all_baselines():
    """obtaintakeallBaseline models"""
    return {
        'Naive_Last': NaiveBaseline('last'),
        'Naive_MA3': NaiveBaseline('moving_average', 3),
        'Naive_Delta': Naive_Delta(),
        'ARIMA(2,1,2)': ARIMAModel((2, 1, 2)),
        'EVM_CPI_Simple': EVMForecast('cpi_simple'),
        'EVM_CPI_Trend': EVMForecast('cpi_trend'),
        'ETS_Holt_Damped': ETSModel(damped=True),
        'Prophet': ProphetModel()
    }


def run_baseline_comparison(train: np.ndarray, test: np.ndarray,
                             baseline: np.ndarray = None,
                             lstm_p10: np.ndarray = None,
                             lstm_p50: np.ndarray = None,
                             lstm_p90: np.ndarray = None) -> pd.DataFrame:
    """
    runallbaselinecomparison

    Returns:
        DataFrame with all metrics for each method
    """
    horizon = len(test)
    models = get_all_baselines()
    results = []

    for name, model in models.items():
        print(f"  Running {name}...")
        try:
            # 预测
            if 'EVM' in name and baseline is not None:
                pred = model.predict(train, baseline, horizon)
                p10, p50, p90 = model.predict_quantiles(train, baseline, horizon)
            else:
                pred = model.predict(train, horizon)
                if hasattr(model, 'predict_quantiles'):
                    p10, p50, p90 = model.predict_quantiles(train, horizon)
                else:
                    p10, p50, p90 = pred * 0.9, pred, pred * 1.1

            # ensuremonotonicincreasing
            pred = np.maximum.accumulate(pred)
            pred = np.clip(pred, 0, 100)

            p50 = np.maximum.accumulate(np.clip(p50, 0, 100))
            p10 = np.clip(p10, 0, p50)
            p90 = np.clip(p90, p50, 150)

            # truncatetakelength
            min_len = min(len(pred), len(test))

            # calculatemetrics
            metrics = calculate_probabilistic_metrics(
                test[:min_len], p10[:min_len], p50[:min_len], p90[:min_len]
            )
            metrics['Method'] = name
            metrics['Provides_Interval'] = 'Approx' if name.startswith('Naive_') else 'Yes'
            results.append(metrics)

        except Exception as e:
            print(f"    {name} failed: {e}")

    # addLSTMresults (ifprovide)
    if lstm_p50 is not None:
        min_len = min(len(lstm_p50), len(test))
        if lstm_p10 is None:
            lstm_p10 = lstm_p50 * 0.9
        if lstm_p90 is None:
            lstm_p90 = lstm_p50 * 1.1

        metrics = calculate_probabilistic_metrics(
            test[:min_len], lstm_p10[:min_len], lstm_p50[:min_len], lstm_p90[:min_len]
        )
        metrics['Method'] = 'LSTM (Ours)'
        metrics['Provides_Interval'] = 'Yes'
        results.append(metrics)

    df = pd.DataFrame(results)

    # sortsequence
    cols = ['Method', 'MAE', 'RMSE', 'R2', 'MAPE',
            'Pinball_P10', 'Pinball_P50', 'Pinball_P90', 'Pinball_Avg',
            'IntervalScore_80', 'Coverage_80', 'MeanWidth_80', 'Provides_Interval']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('MAE')

    return df


# ===========================================================================================
# PART 8: test
# ===========================================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # generatetestdata
    n = 24
    t = np.arange(1, n + 1)
    true_curve = 100 / (1 + np.exp(-0.3 * (t - 12)))
    true_curve = true_curve / true_curve[-1] * 100

    noisy = true_curve + np.random.normal(0, 2, n)
    noisy = np.maximum.accumulate(np.clip(noisy, 0, 100))

    train = noisy[:12]
    test = noisy[12:]
    baseline = true_curve[:12]

    print("=" * 70)
    print("BASELINE COMPARISON TEST")
    print("=" * 70)

    df = run_baseline_comparison(train, test, baseline)
    print("\n" + df.to_string(index=False))