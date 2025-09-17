"""Advanced subscriber growth modelling utilities.

This module prepares daily subscriber history from Substack exports and fits a
piecewise logistic growth model with optional event pulses provided by the
user. The implementation aims to give analysts an interpretable baseline that
can be upgraded later to full Bayesian estimation. It exposes two main entry
points used by the app:

``prepare_daily_history``
    Takes the raw Substack "All" and "Paid" exports (as pandas Series) and
    constructs a consistent daily history with derived flow metrics.

``fit_piecewise_logistic``
    Fits a logistic growth model with change points and optional exogenous
    pulses representing shout-outs or paid acquisition bursts. The
    implementation uses deterministic optimisation so it can run quickly inside
    an interactive app, while still capturing saturation dynamics and phase
    shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class EventPulse:
    """Represents an exogenous pulse (e.g., shout-out or ad burst).

    Attributes
    ----------
    date: ``pd.Timestamp``
        Calendar date of the event. The pulse will start on this date.
    lift: ``float``
        Total incremental subscribers attributed to the event. The pulse
        distributes this lift across subsequent days using an exponential
        decay kernel so the sum matches the specified lift.
    decay_days: ``float``
        Controls the half-life (in days) of the pulse. Larger values spread the
        lift over more days; values close to zero behave like a one-day spike.
    """

    date: pd.Timestamp
    lift: float
    decay_days: float = 3.0


@dataclass(frozen=True)
class PreparedHistory:
    """Daily aligned subscriber history with derived flow metrics."""

    timeline: pd.DatetimeIndex
    total: pd.Series
    paid: pd.Series
    free: pd.Series
    total_daily_adds: pd.Series
    paid_daily_adds: pd.Series
    free_daily_adds: pd.Series


@dataclass(frozen=True)
class SegmentParameters:
    """Parameters for a single logistic segment."""

    start_index: int
    end_index: int
    growth_rate: float
    carrying_capacity: float


@dataclass(frozen=True)
class LogisticFitResult:
    """Return object from ``fit_piecewise_logistic``."""

    fitted: pd.Series
    event_contribution: pd.Series
    segments: List[SegmentParameters]
    diagnostics: dict[str, float]


def prepare_daily_history(
    total_series: pd.Series | None,
    paid_series: pd.Series | None,
) -> PreparedHistory:
    """Align the Substack exports to a shared daily timeline and derive flows.

    Parameters
    ----------
    total_series:
        Total active subscribers over time (free + paid).
    paid_series:
        Paid subscribers over time.

    Returns
    -------
    ``PreparedHistory``
        Daily-aligned cumulative series and the associated first differences.
    """

    if total_series is None and paid_series is None:
        raise ValueError("At least one of total_series or paid_series must be provided.")

    series_list: list[pd.Series] = []
    if total_series is not None:
        series_list.append(total_series.rename("total"))
    if paid_series is not None:
        series_list.append(paid_series.rename("paid"))

    # Construct the union index and reindex each series to daily frequency.
    combined_index = series_list[0].index
    for s in series_list[1:]:
        combined_index = combined_index.union(s.index)
    combined_index = pd.DatetimeIndex(sorted(combined_index.unique()))

    if not isinstance(combined_index, pd.DatetimeIndex):
        combined_index = pd.to_datetime(combined_index)

    daily_index = pd.date_range(combined_index.min(), combined_index.max(), freq="D")

    aligned = {}
    for s in series_list:
        aligned_series = (
            s.astype(float)
            .reindex(combined_index)
            .interpolate(method="time")
            .reindex(daily_index)
            .ffill()
        )
        aligned[aligned_series.name] = aligned_series

    total = aligned.get("total")
    paid = aligned.get("paid")

    if total is None and paid is not None:
        total = paid.copy()
    if paid is None:
        paid = pd.Series(np.zeros_like(total, dtype=float), index=total.index, name="paid")

    free = (total - paid).clip(lower=0.0)
    free.name = "free"

    total_adds = total.diff().fillna(0.0)
    paid_adds = paid.diff().fillna(0.0)
    free_adds = free.diff().fillna(0.0)

    return PreparedHistory(
        timeline=daily_index,
        total=total.reindex(daily_index).ffill(),
        paid=paid.reindex(daily_index).ffill(),
        free=free.reindex(daily_index).ffill(),
        total_daily_adds=total_adds.reindex(daily_index).fillna(0.0),
        paid_daily_adds=paid_adds.reindex(daily_index).fillna(0.0),
        free_daily_adds=free_adds.reindex(daily_index).fillna(0.0),
    )


def _compute_event_impulse(index: pd.DatetimeIndex, events: Sequence[EventPulse]) -> np.ndarray:
    """Return the incremental subscribers contributed by events for each day."""

    impulse = np.zeros(len(index), dtype=float)
    if not events:
        return impulse

    for event in events:
        if np.isnan(event.lift) or event.lift == 0:
            continue
        try:
            event_date = pd.to_datetime(event.date).normalize()
        except Exception:
            continue
        if event_date < index[0] or event_date > index[-1]:
            continue
        start_idx = int(index.get_indexer([event_date], method="nearest")[0])
        decay = max(float(event.decay_days), 0.0)
        if decay <= 1e-6:
            impulse[start_idx] += float(event.lift)
            continue
        max_days = int(min(len(index) - start_idx, max(1, round(decay * 5))))
        weights = np.exp(-np.arange(max_days) / decay)
        weights /= weights.sum()
        impulse[start_idx : start_idx + max_days] += float(event.lift) * weights

    return impulse


def _fit_segment_params(values: np.ndarray, impulses: np.ndarray) -> tuple[float, float]:
    """Estimate logistic parameters for a single segment.

    ``values`` contains the observed subscriber counts for the segment, and
    ``impulses`` contains the incremental subscribers attributed to events for
    the same interval. Both arrays are contiguous slices of the overall series.
    """

    if values.size < 3:
        current = float(values[-1])
        return (0.0, max(current * 1.1, 1.0))

    deltas = np.diff(values)
    residual = deltas - impulses[1:]
    prev = values[:-1]

    # If residuals are nearly zero, return near-zero growth.
    if np.allclose(residual, 0.0):
        return (0.0, max(float(values.max()), 1.0))

    max_prev = max(float(prev.max()), 1.0)
    mean_resid = float(np.mean(residual))
    initial_r = max(mean_resid / max_prev, 1e-4)
    initial_log_k = np.log(max_prev * 1.5)

    def _residual(theta: np.ndarray) -> np.ndarray:
        r = np.exp(theta[0])
        k = max_prev + np.exp(theta[1])
        pred = r * prev * (1.0 - prev / k)
        return pred - residual

    result = optimize.least_squares(_residual, x0=np.array([np.log(initial_r), initial_log_k]))
    r = float(np.exp(result.x[0]))
    k = float(max_prev + np.exp(result.x[1]))
    return (r, k)


def fit_piecewise_logistic(
    series: pd.Series,
    change_points: Iterable[pd.Timestamp] | Iterable[int],
    events: Sequence[EventPulse] | None = None,
) -> LogisticFitResult:
    """Fit a piecewise logistic model with optional event pulses.

    Parameters
    ----------
    series:
        Daily subscriber counts (free or total) as a ``pd.Series``.
    change_points:
        Iterable of change point indicators. Values can be integer indices into
        ``series`` or timestamps aligned with the series index. Change points
        split the series into segments with their own growth rates and
        capacities.
    events:
        Optional sequence of ``EventPulse`` objects representing external lifts.

    Returns
    -------
    ``LogisticFitResult``
        Contains the fitted trajectory, event contributions, segment parameters
        and summary diagnostics.
    """

    if series.empty:
        raise ValueError("Series must contain at least one observation.")

    y = series.astype(float).sort_index()
    idx = y.index
    impulses = _compute_event_impulse(idx, events or [])

    # Normalize change points to integer indices relative to the sorted series.
    cp_indices: list[int] = []
    for cp in change_points:
        if isinstance(cp, (int, np.integer)):
            cp_idx = int(cp)
        else:
            cp_ts = pd.to_datetime(cp)
            if cp_ts < idx[0] or cp_ts > idx[-1]:
                continue
            cp_idx = int(idx.get_loc(cp_ts))
        if 0 < cp_idx < len(y) - 1:
            cp_indices.append(cp_idx)
    cp_indices = sorted(set(cp_indices))

    boundaries = [0] + cp_indices + [len(y) - 1]

    segments: list[SegmentParameters] = []
    fitted = np.zeros_like(y.to_numpy())
    fitted[0] = float(y.iloc[0])

    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        seg_values = y.iloc[seg_start : seg_end + 1].to_numpy()
        seg_impulses = impulses[seg_start : seg_end + 1]
        r, k = _fit_segment_params(seg_values, seg_impulses)
        segments.append(
            SegmentParameters(
                start_index=seg_start,
                end_index=seg_end,
                growth_rate=r,
                carrying_capacity=k,
            )
        )
        # Simulate within the segment using the estimated parameters.
        for t in range(seg_start + 1, seg_end + 1):
            prev = fitted[t - 1]
            logistic_inc = r * prev * (1.0 - prev / k)
            fitted[t] = prev + logistic_inc + impulses[t]

        # Reset fitted value at boundary if the actual series has a jump that the
        # logistic dynamics cannot follow (e.g., due to unmodelled events). This
        # prevents runaway error in subsequent segments.
        if seg_end < len(y) - 1:
            fitted[seg_end] = float(y.iloc[seg_end])

    fitted_series = pd.Series(fitted, index=idx, name="fitted")
    event_series = pd.Series(impulses, index=idx, name="event_contribution")

    residual = (y - fitted_series).abs()
    diagnostics = {
        "mae": float(residual.mean()),
        "rmse": float(np.sqrt(np.mean((y - fitted_series) ** 2))),
        "final_residual": float(abs(y.iloc[-1] - fitted_series.iloc[-1])),
    }

    return LogisticFitResult(
        fitted=fitted_series,
        event_contribution=event_series,
        segments=segments,
        diagnostics=diagnostics,
    )


__all__ = [
    "EventPulse",
    "PreparedHistory",
    "SegmentParameters",
    "LogisticFitResult",
    "prepare_daily_history",
    "fit_piecewise_logistic",
]
