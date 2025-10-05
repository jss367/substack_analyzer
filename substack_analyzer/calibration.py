from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PiecewiseLogisticFit:
    carrying_capacity: float
    segment_growth_rates: list[float]
    breakpoints: list[int]
    gamma_pulse: float
    gamma_step: float
    fitted_series: pd.Series
    residuals: pd.Series
    sse: float
    r2_on_deltas: float
    gamma_exog: Optional[float] = None


def _ensure_month_end_index(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    # Normalize to month end to align with app import convention
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s.sort_index()
    return s


def _segments_from_breakpoints(n: int, breakpoints: Sequence[int]) -> list[tuple[int, int]]:
    if not breakpoints:
        return [(0, n - 1)]
    segments: list[tuple[int, int]] = []
    start = 0
    for bp in breakpoints:
        end = max(min(bp - 1, n - 2), start)  # end applies to delta index; safe bound
        if end >= start:
            segments.append((start, end))
        start = bp
    if start <= n - 2:
        segments.append((start, n - 2))
    return segments


def _event_regressors(index: pd.DatetimeIndex, events_df: Optional[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    if events_df is None or events_df.empty:
        return np.zeros(len(index)), np.zeros(len(index))
    df = events_df.dropna(subset=["date"]).copy()
    if df.empty:
        return np.zeros(len(index)), np.zeros(len(index))
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    pulse = np.zeros(len(index), dtype=float)
    step = np.zeros(len(index), dtype=float)
    for _, row in df.iterrows():
        when: pd.Timestamp = row["date"]
        kind = str(row.get("type", ""))
        # Optional: use cost as weight for pulse; default 1.0
        weight = float(row.get("cost", 1.0) or 1.0)
        persistence = str(row.get("persistence", "")).strip().lower()
        if when in index:
            i = int(index.get_loc(when))
            if persistence == "no effect":
                continue
            if persistence == "persistent":
                step[index >= when] += 1.0
            elif persistence == "transient":
                # Pulse at the event month
                if kind.lower().startswith("ad ") or kind.lower() == "ad spend":
                    pulse[i] += weight
                else:
                    pulse[i] += 1.0
            else:
                # Backward-compatibility: both
                if kind.lower().startswith("ad ") or kind.lower() == "ad spend":
                    pulse[i] += weight
                else:
                    pulse[i] += 1.0
                step[index >= when] += 1.0
    return pulse, step


def fit_piecewise_logistic(
    total_series: pd.Series,
    breakpoints: Optional[list[int]] = None,
    events_df: Optional[pd.DataFrame] = None,
    k_grid: Optional[Sequence[float]] = None,
    extra_exog: Optional[pd.Series] = None,
) -> PiecewiseLogisticFit:
    """Fit a piecewise-logistic model on monthly totals via grid-search over K and OLS.

    Discrete dynamic on month t (t>=1):
        ΔS_t = r_seg(t) * S_{t-1} * (1 - S_{t-1}/K) + γ_pulse * pulse_t + γ_step * step_t + γ_exog * exog_t + ε_t

    - r_seg(t) is piecewise-constant across user-provided breakpoints (indices in [0, n)).
    - K (carrying capacity) is shared across segments and chosen by grid-search.
    - γ_pulse, γ_step, γ_exog are global coefficients.
    - Parameters are estimated by OLS on ΔS_t for each candidate K; pick K with lowest SSE.
    """
    s = _ensure_month_end_index(total_series)
    if s.size < 4:
        raise ValueError("Need at least 4 months of data to fit the model")
    # Construct deltas and base regressor X_t(K)
    y = s.diff().dropna()
    s_lag = s.shift(1).reindex(y.index)
    n = y.size

    # Default breakpoints: none (single segment)
    bp = list(breakpoints or [])
    # Build segments on the index of y (which starts at original index[1])
    seg_bounds = _segments_from_breakpoints(n + 1, bp)
    num_segments = len(seg_bounds)

    # Events
    pulse, step = _event_regressors(y.index, events_df)

    # Optional exogenous aligned to y index
    exog = None
    if extra_exog is not None:
        try:
            exog = extra_exog.reindex(y.index).astype(float).to_numpy()
            # Replace nans with zeros
            exog = np.where(np.isfinite(exog), exog, 0.0)
        except Exception:
            exog = None

    # K grid to do grid search for carrying capacity
    max_s = float(s.max())
    if k_grid is None:
        k_grid = np.linspace(max_s * 1.1, max_s * 5.0, num=25)

    best: Optional[PiecewiseLogisticFit] = None
    best_sse = np.inf

    for K in k_grid:
        X_base = (s_lag * (1.0 - s_lag / K)).to_numpy().astype(float)
        # Design matrix: one column per segment (X_base masked), plus pulse, step, optional exog
        X_cols: list[np.ndarray] = []
        for start, end in seg_bounds:
            mask = np.zeros(n, dtype=float)
            mask[start : end + 1] = 1.0
            X_cols.append(X_base * mask)
        X_cols.append(pulse.astype(float))
        X_cols.append(step.astype(float))
        if exog is not None:
            X_cols.append(exog.astype(float))
        X = np.column_stack(X_cols)
        y_vec = y.to_numpy().astype(float)

        # OLS via least squares
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Unpack parameters
        r_segments = [float(b) for b in beta[:num_segments]]
        gamma_pulse = float(beta[num_segments])
        gamma_step = float(beta[num_segments + 1])
        beta_offset = num_segments + 2
        gamma_exog = float(beta[beta_offset]) if exog is not None and len(beta) > beta_offset else None

        # Reconstruct fitted series recursively
        s_hat = [float(s.iloc[0])]
        for t in range(1, s.size):
            x_t = s_hat[-1] * (1.0 - s_hat[-1] / K)
            # Determine which segment t-1 (for ΔS at month t) belongs to
            seg_idx = 0
            for j, (a, b) in enumerate(seg_bounds):
                if (t - 1) >= a and (t - 1) <= b:
                    seg_idx = j
                    break
            delta = r_segments[seg_idx] * x_t
            # Add events
            if t - 1 < len(pulse):
                delta += gamma_pulse * float(pulse[t - 1])
                delta += gamma_step * float(step[t - 1])
            # Add optional exogenous
            if exog is not None and (t - 1) < len(exog) and gamma_exog is not None:
                delta += gamma_exog * float(exog[t - 1])
            s_hat.append(max(s_hat[-1] + delta, 0.0))

        fitted = pd.Series(s_hat, index=s.index)
        # Residuals on deltas
        y_hat = fitted.diff().reindex(y.index)
        resid = y - y_hat
        sse = float(np.square(resid.to_numpy()).sum())
        tss = float(np.square(y.to_numpy() - float(y.mean())).sum())
        r2 = 1.0 - (sse / tss if tss > 0 else np.nan)

        fit = PiecewiseLogisticFit(
            carrying_capacity=float(K),
            segment_growth_rates=r_segments,
            breakpoints=bp,
            gamma_pulse=gamma_pulse,
            gamma_step=gamma_step,
            fitted_series=fitted,
            residuals=resid,
            sse=sse,
            r2_on_deltas=float(r2),
            gamma_exog=gamma_exog,
        )

        if sse < best_sse:
            best_sse = sse
            best = fit

    if best is None:
        raise RuntimeError("Could not fit piecewise logistic model")
    return best


def forecast_piecewise_logistic(
    last_value: float,
    months_ahead: int,
    carrying_capacity: float,
    segment_growth_rate: float,
    gamma_pulse: float = 0.0,
    gamma_step_level: float = 0.0,
) -> np.ndarray:
    """Simple forward simulation with constant parameters for months_ahead.

    Uses the final segment's growth rate and optional constant step level.
    """
    values = [float(last_value)]
    for _ in range(months_ahead):
        x_t = values[-1] * (1.0 - values[-1] / carrying_capacity)
        delta = segment_growth_rate * x_t + gamma_step_level
        values.append(max(values[-1] + delta + gamma_pulse, 0.0))
        # Only apply pulse in first step
        gamma_pulse = 0.0
    return np.array(values[1:], dtype=float)


def fitted_series_from_params(
    total_series: pd.Series,
    breakpoints: Optional[list[int]],
    carrying_capacity: float,
    segment_growth_rates: Sequence[float],
    events_df: Optional[pd.DataFrame] = None,
    extra_exog: Optional[pd.Series] = None,
    gamma_pulse: float = 0.0,
    gamma_step: float = 0.0,
    gamma_exog: Optional[float] = None,
) -> pd.Series:
    """
    This takes the parameters and uses uses them to predict the future.

    Applies the same discrete dynamic used in fitting, aligned to month-end index.
    """
    s = _ensure_month_end_index(total_series)
    if s.size == 0:
        return s

    # Align helper arrays to deltas index
    y_index = s.index[1:]
    pulse, step = _event_regressors(y_index, events_df)

    exog = None
    if extra_exog is not None:
        try:
            exog = extra_exog.reindex(y_index).astype(float).to_numpy()
            exog = np.where(np.isfinite(exog), exog, 0.0)
        except Exception:
            exog = None

    # Segment bounds on the original series index
    seg_bounds = _segments_from_breakpoints(len(s), list(breakpoints or []))
    r_list = list(segment_growth_rates)
    if len(r_list) < len(seg_bounds):
        # Pad with last known rate
        r_list = r_list + [r_list[-1] if r_list else 0.0] * (len(seg_bounds) - len(r_list))

    s_hat = [float(s.iloc[0])]
    for t in range(1, s.size):
        x_t = s_hat[-1] * (1.0 - s_hat[-1] / float(carrying_capacity))
        # Determine segment for delta at t
        seg_idx = 0
        for j, (a, b) in enumerate(seg_bounds):
            if (t - 1) >= a and (t - 1) <= b:
                seg_idx = j
                break
        delta = float(r_list[seg_idx]) * x_t
        if t - 1 < len(pulse):
            delta += float(gamma_pulse) * float(pulse[t - 1])
            delta += float(gamma_step) * float(step[t - 1])
        if exog is not None and (t - 1) < len(exog) and gamma_exog is not None:
            delta += float(gamma_exog) * float(exog[t - 1])
        s_hat.append(max(s_hat[-1] + delta, 0.0))

    return pd.Series(s_hat, index=s.index)
