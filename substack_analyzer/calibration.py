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
    gamma_intercept: float = 0.0


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
        # Pulse/step weighting: use cost as weight for any pulse occurrence when provided
        weight = float(row.get("cost", 1.0) or 1.0)
        persistence = str(row.get("persistence", "")).strip().lower()
        if when in index:
            i = int(index.get_loc(when))
            if persistence == "no effect":
                continue
            if persistence == "persistent":
                step[index >= when] += weight
            elif persistence == "transient":
                # Pulse at the event month (always weight by cost if available)
                pulse[i] += weight
            else:
                # Backward-compatibility: both (weight pulse by cost if provided)
                pulse[i] += weight
                step[index >= when] += weight
    return pulse, step


def fit_piecewise_logistic(
    total_series: pd.Series,
    breakpoints: list[int],
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
    input_series = _ensure_month_end_index(total_series)
    if input_series.size < 4:
        raise ValueError("Need at least 4 months of data to fit the model")
    # Construct deltas and base regressor X_t(K)
    y = input_series.diff().dropna()
    s_lag = input_series.shift(1).reindex(y.index).astype(float)
    n = y.size

    # Build segments on the index of y (which starts at original index[1])
    # Sanitize breakpoints relative to original series length (len(input_series))
    n_series = len(input_series)
    bps = sorted(set(int(b) for b in breakpoints if 1 <= int(b) <= n_series - 2)) if breakpoints else []
    seg_bounds = _segments_from_breakpoints(n_series, bps)
    num_segments = len(seg_bounds)

    # Events
    pulse, step = _event_regressors(y.index, events_df)
    pulse = np.asarray(pd.Series(pulse, index=y.index), dtype=float)
    step = np.asarray(pd.Series(step, index=y.index), dtype=float)

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
    max_s = float(input_series.max())
    if k_grid is None:
        # Ensure K is strictly greater than max_s with a relative epsilon to avoid degeneracy
        baseline = max_s if max_s > 0.0 else 1.0
        eps = max(baseline * 1e-3, 1e-6)
        start = baseline + eps
        k_grid = np.concatenate(
            [
                np.linspace(start, baseline * 1.5 + eps, 8),
                np.linspace(baseline * 1.5 + eps, baseline * 5.0 + eps, 25),
                np.linspace(baseline * 6.0 + eps, baseline * 10.0 + eps, 10),
            ]
        )

    best: Optional[PiecewiseLogisticFit] = None
    best_sse = np.inf

    # Precompute Δ-space segment masks for efficiency and stability
    masks: list[np.ndarray] = []
    for start, end in seg_bounds:
        mask = np.zeros(n, dtype=float)
        lo = max(0, start)
        hi = min(n, end + 1)
        if lo < hi:
            mask[lo:hi] = 1.0
        masks.append(mask)

    for K in k_grid:
        X_base = (s_lag * (1.0 - s_lag / K)).to_numpy().astype(float)
        # Build design matrix from precomputed masks
        X_cols: list[np.ndarray] = [(X_base * m) for m in masks]
        X_cols.append(np.ones(n, dtype=float))  # intercept
        X_cols.append(pulse)
        X_cols.append(step)
        if exog is not None:
            X_cols.append(exog)
        X = np.column_stack(X_cols)
        y_vec = y.to_numpy().astype(float)

        # OLS with a tiny ridge for stability (helps when columns are nearly collinear)
        lam = 1e-6
        XtX = X.T @ X
        Xty = X.T @ y_vec
        try:
            beta = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), Xty)
        except np.linalg.LinAlgError:
            # very rare; fall back to lstsq
            beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)

        # Unpack parameters
        r_segments = [float(b) for b in beta[:num_segments]]
        gamma_intercept = float(beta[num_segments])
        gamma_pulse = float(beta[num_segments + 1])
        gamma_step = float(beta[num_segments + 2])
        beta_offset = num_segments + 3
        gamma_exog = float(beta[beta_offset]) if exog is not None and len(beta) > beta_offset else None

        # Reconstruct fitted series recursively
        s_hat = [float(input_series.iloc[0])]
        for t in range(1, input_series.size):
            x_t = s_hat[-1] * (1.0 - s_hat[-1] / K)
            # Determine which segment t-1 (for ΔS at month t) belongs to
            seg_idx = 0
            for j, (a, b) in enumerate(seg_bounds):
                if (t - 1) >= a and (t - 1) <= b:
                    seg_idx = j
                    break
            delta = gamma_intercept + r_segments[seg_idx] * x_t
            # Add events
            if t - 1 < len(pulse):
                delta += gamma_pulse * float(pulse[t - 1])
                delta += gamma_step * float(step[t - 1])
            # Add optional exogenous
            if exog is not None and (t - 1) < len(exog) and gamma_exog is not None:
                delta += gamma_exog * float(exog[t - 1])
            s_hat.append(max(s_hat[-1] + delta, 0.0))

        fitted = pd.Series(s_hat, index=input_series.index)
        # Residuals on deltas
        y_hat = fitted.diff().reindex(y.index)
        resid = y - y_hat
        sse = float(np.square(resid.to_numpy()).sum())
        tss = float(np.square(y.to_numpy() - float(y.mean())).sum())
        r2 = 1.0 - (sse / tss if tss > 0 else np.nan)

        fit = PiecewiseLogisticFit(
            carrying_capacity=float(K),
            segment_growth_rates=r_segments,
            breakpoints=bps,
            gamma_pulse=gamma_pulse,
            gamma_step=gamma_step,
            fitted_series=fitted,
            residuals=resid,
            sse=sse,
            r2_on_deltas=float(r2),
            gamma_exog=gamma_exog,
            gamma_intercept=gamma_intercept,
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
    breakpoints: list[int],
    carrying_capacity: float,
    segment_growth_rates: Sequence[float],
    events_df: Optional[pd.DataFrame] = None,
    extra_exog: Optional[pd.Series] = None,
    gamma_pulse: float = 0.0,
    gamma_step: float = 0.0,
    gamma_exog: Optional[float] = None,
    gamma_intercept: Optional[float] = None,
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

    # Sanitise breakpoints in the same manner as the fitter
    n_series = len(s)
    if breakpoints:
        bps = sorted({int(b) for b in breakpoints if 1 <= int(b) <= max(n_series - 2, 1)})
    else:
        bps = []

    # Segment bounds on the original series index
    seg_bounds = _segments_from_breakpoints(n_series, bps)
    r_list = list(segment_growth_rates)
    if len(r_list) < len(seg_bounds):
        # Pad with last known rate
        r_list = r_list + [r_list[-1] if r_list else 0.0] * (len(seg_bounds) - len(r_list))

    # If intercept not supplied, infer it from observed deltas with provided params
    if gamma_intercept is None:
        y = s.diff().dropna()
        s_lag = s.shift(1).reindex(y.index).astype(float)
        # Map each delta row to its segment index
        seg_idx_per_row: list[int] = []
        for t in range(y.size):
            seg_idx = 0
            for j, (a, b) in enumerate(seg_bounds):
                if a <= t <= b:
                    seg_idx = j
                    break
            seg_idx_per_row.append(seg_idx)

        x_base = s_lag.to_numpy() * (1.0 - s_lag.to_numpy() / float(carrying_capacity))
        contrib = np.zeros_like(x_base, dtype=float)
        for t, seg_idx in enumerate(seg_idx_per_row):
            contrib[t] = float(r_list[seg_idx]) * x_base[t]

        contrib += float(gamma_pulse) * np.asarray(pulse, dtype=float)
        contrib += float(gamma_step) * np.asarray(step, dtype=float)
        if exog is not None and gamma_exog is not None:
            contrib += float(gamma_exog) * exog

        residual = y.to_numpy(dtype=float) - contrib
        gamma_intercept = float(np.nanmean(residual)) if residual.size else 0.0
    else:
        gamma_intercept = float(gamma_intercept)

    s_hat = [float(s.iloc[0])]
    for t in range(1, s.size):
        x_t = s_hat[-1] * (1.0 - s_hat[-1] / float(carrying_capacity))
        # Determine segment for delta at t
        seg_idx = 0
        for j, (a, b) in enumerate(seg_bounds):
            if (t - 1) >= a and (t - 1) <= b:
                seg_idx = j
                break
        delta = float(gamma_intercept) + float(r_list[seg_idx]) * x_t
        if t - 1 < len(pulse):
            delta += float(gamma_pulse) * float(pulse[t - 1])
            delta += float(gamma_step) * float(step[t - 1])
        if exog is not None and (t - 1) < len(exog) and gamma_exog is not None:
            delta += float(gamma_exog) * float(exog[t - 1])
        s_hat.append(max(s_hat[-1] + delta, 0.0))

    # Return float-valued series to exactly match the fitted recursion
    return pd.Series(np.asarray(s_hat, dtype=float), index=s.index)
