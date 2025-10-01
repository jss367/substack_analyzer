from typing import Optional

import numpy as np
import pandas as pd

from substack_analyzer.types import SegmentSlope


def _fit_slope_per_month(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = pd.RangeIndex(len(values)).to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    denom = float(((x - x.mean()) ** 2).sum())
    if denom == 0.0:
        return 0.0
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
    return slope


def compute_segment_slopes(series: pd.Series, breakpoints: list[int]) -> list[SegmentSlope]:
    s = series.dropna()
    if not breakpoints:
        breakpoints = [s.shape[0]]
    segments: list[SegmentSlope] = []
    start = 0
    for bp in breakpoints:
        seg_vals = s.iloc[start:bp]
        slope = _fit_slope_per_month(seg_vals)
        segments.append(
            SegmentSlope(
                start_index=start,
                end_index=bp - 1,
                start_date=s.index[start],
                end_date=s.index[bp - 1],
                slope_per_month=float(slope),
            )
        )
        start = bp
    return segments


def slope_around(series: pd.Series, event_date: pd.Timestamp, window: int = 6) -> tuple[float, float]:
    """Return (pre_slope, post_slope) using +/- window months around event_date."""
    s = series.dropna()
    if s.empty:
        return (0.0, 0.0)
    # Find closest index at or before event_date
    idx = s.index.searchsorted(event_date, side="right") - 1
    idx = max(min(idx, len(s) - 2), 1)
    start_pre = max(0, idx - window + 1)
    end_pre = idx + 1
    start_post = idx + 1
    end_post = min(len(s), idx + 1 + window)
    pre = _fit_slope_per_month(s.iloc[start_pre:end_pre])
    post = _fit_slope_per_month(s.iloc[start_post:end_post])
    return (float(pre), float(post))


def detect_change_points(
    series: pd.Series,
    paid: Optional[pd.Series] = None,  # placeholder for future multi-series extension
    max_changes: int = 4,
    min_seg_len: int = 2,
    penalty_scale: float = 4.0,
    return_timestamps: bool = False,
) -> list[int] | list[pd.Timestamp]:
    """
    Detect change points emphasizing slope breaks (mean shifts in first differences).
    Uses binary segmentation on delta = diff(series) with a BIC-like penalty.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series (dates as index).
    paid : Optional[pd.Series]
        Currently unused; kept for future multi-series extension.
    max_changes : int
        Maximum number of change points to return.
    min_seg_len : int
        Minimum segment length in delta-space (requires this many deltas per segment).
    penalty_scale : float
        Multiplier on sigma^2 * log(n_delta) as the acceptance penalty for adding a split.
        Larger -> fewer breaks.
    return_timestamps : bool
        If True, return change points as pd.Timestamp values instead of integer indices.

    Returns
    -------
    list[int] | list[pd.Timestamp]
        Breakpoint positions (indices into the original series or corresponding timestamps).
    """
    s = series.dropna().sort_index()
    n = len(s)
    if n < (2 * min_seg_len + 2) or max_changes <= 0:
        return []

    # First difference: slope change = mean shift in delta
    x = s.diff().dropna().to_numpy()  # length n-1
    m = len(x)
    if m < (2 * min_seg_len + 1):
        return []

    # Estimate noise variance for penalty
    mad = np.median(np.abs(x - np.median(x))) if m > 0 else 0.0
    sigma2 = (1.4826 * mad) ** 2 if mad > 0 else (np.var(x, ddof=1) if m > 1 else 0.0)
    if not np.isfinite(sigma2) or sigma2 == 0.0:
        sigma2 = 1e-12

    penalty = penalty_scale * sigma2 * np.log(max(m, 2))

    # Prefix sums for SSE
    S1 = np.zeros(m + 1)  # sum x
    S2 = np.zeros(m + 1)  # sum x^2
    S1[1:] = np.cumsum(x)
    S2[1:] = np.cumsum(x * x)

    def seg_sse(a: int, b: int) -> float:
        n_ = b - a
        if n_ <= 0:
            return 0.0
        s1 = S1[b] - S1[a]
        s2 = S2[b] - S2[a]
        mu = s1 / n_
        return s2 - n_ * mu * mu

    def best_split(a: int, b: int) -> tuple[Optional[int], float]:
        """Return (k, gain) for the best split k in (a+min_seg_len ... b-min_seg_len)."""
        L = b - a
        if L < 2 * min_seg_len + 1:
            return None, 0.0
        total = seg_sse(a, b)
        best_k, best_gain = None, 0.0
        for k in range(a + min_seg_len, b - min_seg_len):
            gain = total - (seg_sse(a, k) + seg_sse(k, b))
            if gain > best_gain:
                best_gain, best_k = gain, k
        return best_k, best_gain

    breaks: list[int] = []
    segments: list[tuple[int, int]] = [(0, m)]

    while segments and len(breaks) < max_changes:
        candidate = []
        for a, b in segments:
            k, gain = best_split(a, b)
            if k is not None:
                candidate.append((gain, a, k, b))
        if not candidate:
            break
        gain, a, k, b = max(candidate, key=lambda t: t[0])
        if gain <= penalty:
            break
        breaks.append(k)
        segments.remove((a, b))
        segments.extend([(a, k), (k, b)])

    # Map delta index k to original series index
    cp_indices = sorted([k + 1 for k in breaks if 0 <= k + 1 < n])

    if return_timestamps:
        return [s.index[i] for i in cp_indices]
    return cp_indices
