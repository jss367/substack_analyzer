from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PiecewiseLogisticFit:
    carrying_capacity: float
    segment_growth_rates: List[float]
    breakpoints: List[int]
    gamma_pulse: float
    gamma_step: float
    fitted_series: pd.Series
    residuals: pd.Series
    sse: float
    r2_on_deltas: float


def _ensure_month_end_index(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    # Normalize to month end to align with app import convention
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s.sort_index()
    return s


def _segments_from_breakpoints(n: int, breakpoints: Sequence[int]) -> List[Tuple[int, int]]:
    if not breakpoints:
        return [(0, n - 1)]
    segments: List[Tuple[int, int]] = []
    start = 0
    for bp in breakpoints:
        end = max(min(bp - 1, n - 2), start)  # end applies to delta index; safe bound
        if end >= start:
            segments.append((start, end))
        start = bp
    if start <= n - 2:
        segments.append((start, n - 2))
    return segments


def _event_regressors(index: pd.DatetimeIndex, events_df: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
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
        # Pulse at the event month
        if when in index:
            i = int(index.get_loc(when))
            if kind.lower().startswith("ad ") or kind.lower() == "ad spend":
                pulse[i] += weight
            else:
                pulse[i] += 1.0
        # Step from event month onward
        step[index >= when] += 1.0
    return pulse, step


def fit_piecewise_logistic(
    total_series: pd.Series,
    breakpoints: Optional[List[int]] = None,
    events_df: Optional[pd.DataFrame] = None,
    k_grid: Optional[Sequence[float]] = None,
) -> PiecewiseLogisticFit:
    """Fit a piecewise-logistic model on monthly totals via grid-search over K and OLS.

    Discrete dynamic on month t (t>=1):
        ΔS_t = r_seg(t) * S_{t-1} * (1 - S_{t-1}/K) + γ_pulse * pulse_t + γ_step * step_t + ε_t

    - r_seg(t) is piecewise-constant across user-provided breakpoints (indices in [0, n)).
    - K (carrying capacity) is shared across segments and chosen by grid-search.
    - γ_pulse, γ_step are global coefficients.
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

    # K grid
    max_s = float(s.max())
    if k_grid is None:
        k_grid = np.linspace(max_s * 1.1, max_s * 5.0, num=25)

    best: Optional[PiecewiseLogisticFit] = None
    best_sse = np.inf

    for K in k_grid:
        X_base = (s_lag * (1.0 - s_lag / K)).to_numpy().astype(float)
        # Design matrix: one column per segment (X_base masked), plus pulse, step
        X_cols: List[np.ndarray] = []
        for start, end in seg_bounds:
            mask = np.zeros(n, dtype=float)
            mask[start : end + 1] = 1.0
            X_cols.append(X_base * mask)
        X_cols.append(pulse.astype(float))
        X_cols.append(step.astype(float))
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

        # Reconstruct fitted series recursively
        s_hat = [float(s.iloc[0])]
        for t in range(1, s.size):
            x_t = s_hat[-1] * (1.0 - s_hat[-1] / K)
            # Determine which segment t-1 (for ΔS at month t) belongs to
            # Map t-1 in [0, n-1] to segment
            seg_idx = 0
            for j, (a, b) in enumerate(seg_bounds):
                if (t - 1) >= a and (t - 1) <= b:
                    seg_idx = j
                    break
            delta = r_segments[seg_idx] * x_t
            # Add events
            # Align pulse/step arrays with y index (starts at s.index[1])
            if t - 1 < len(pulse):
                delta += gamma_pulse * float(pulse[t - 1])
                delta += gamma_step * float(step[t - 1])
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
