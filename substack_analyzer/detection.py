from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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
    n = s.shape[0]
    if n == 0:
        return []

    # Normalise breakpoints so that we do not attempt to index out of range when
    # computing start/end dates. Users of the API can provide arbitrary
    # breakpoints (including unsorted or out-of-bounds values), so we defensively
    # clamp them to the valid range and ensure they are strictly increasing.
    cleaned_breaks: list[int] = []
    for bp in sorted(set(int(b) for b in breakpoints or [])):
        if bp <= 0 or bp > n:
            continue
        if cleaned_breaks and bp <= cleaned_breaks[-1]:
            continue
        cleaned_breaks.append(bp)

    if not cleaned_breaks or cleaned_breaks[-1] != n:
        cleaned_breaks.append(n)

    segments: list[SegmentSlope] = []
    start = 0
    for bp in cleaned_breaks:
        if start >= n:
            break
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


@dataclass
class ChangePoint:
    index: int
    timestamp: pd.Timestamp
    kind: Literal["level", "slope", "accel"]
    order: int  # 0 (level), 1 (slope), 2 (accel)
    gain: float  # raw SSE reduction from the split
    penalty: float  # penalty applied for adding this split
    score: float  # gain - penalty (used for ranking)


def detect_change_points(
    series: pd.Series,
    *,
    max_changes: int = 4,
    min_seg_len: int = 2,
    penalty_scale: float = 4.0,
    kinds: Sequence[Literal["level", "slope", "accel"]] = ("level", "slope", "accel"),
    merge_min_gap: int | None = None,
    return_mode: Literal["rich", "indices", "timestamps"] = "rich",
) -> list[ChangePoint] | list[int] | list[pd.Timestamp]:
    """
    Multi-order change-point detector for LEVEL (order=0), SLOPE (order=1), and ACCELERATION (order=2).

    Strategy:
      • Run the same binary segmentation on Δ^d(series) for d in {0,1,2} corresponding to
        level/slope/accel mean-shift detection.
      • Each pass produces candidate splits with (gain, penalty, score).
      • Merge candidates across orders via non-maximum suppression (by index radius).
      • Return up to `max_changes` merged change points.

    Parameters
    ----------
    series : pd.Series
        Time-indexed numeric series (index must be sortable).
    max_changes : int
        Max number of merged change points to return.
    min_seg_len : int
        Minimum segment length measured *in the differenced space* (Δ^d).
    penalty_scale : float
        Multiplier on sigma^2 * log(m) as acceptance penalty (larger → fewer breaks).
    kinds : Sequence[str]
        Subset of {"level","slope","accel"} to enable.
    merge_min_gap : int | None
        Minimum separation (in original index units) between accepted change points.
        If None, defaults to max(2, min_seg_len + 1).
    return_mode : {"rich","indices","timestamps"}
        Output format.

    Returns
    -------
    list of ChangePoint | list[int] | list[pd.Timestamp]
    """
    if max_changes <= 0:
        return [] if return_mode != "rich" else []

    # Basic prep
    s = series.dropna().sort_index()
    n = len(s)
    if n < max(6, 2 * min_seg_len + 2):
        return [] if return_mode != "rich" else []

    kind_to_order = {"level": 0, "slope": 1, "accel": 2}
    orders = sorted({kind_to_order[k] for k in kinds})
    if not orders:
        return [] if return_mode != "rich" else []

    # Default merge radius
    if merge_min_gap is None:
        merge_min_gap = max(2, min_seg_len + 1)

    # --- Core helpers ---

    def _robust_sigma2(x: np.ndarray) -> float:
        m = len(x)
        if m <= 1:
            return 1e-12
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad > 0:
            return float((1.4826 * mad) ** 2)
        # fallback to sample variance
        v = float(np.var(x, ddof=1)) if m > 1 else 0.0
        return v if np.isfinite(v) and v > 0 else 1e-12

    def _prefix_sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S1 = np.zeros(len(x) + 1)
        S2 = np.zeros(len(x) + 1)
        if len(x):
            S1[1:] = np.cumsum(x)
            S2[1:] = np.cumsum(x * x)
        return S1, S2

    def _seg_sse(S1: np.ndarray, S2: np.ndarray, a: int, b: int) -> float:
        n_ = b - a
        if n_ <= 0:
            return 0.0
        s1 = S1[b] - S1[a]
        s2 = S2[b] - S2[a]
        mu = s1 / n_
        return float(s2 - n_ * mu * mu)

    def _binary_segmentation_on_diff(d: int) -> list[ChangePoint]:
        """
        Run binary segmentation on x = Δ^d(s), detecting mean shifts in x.
        Map a split at position k in x back to original series index i = k + d.
        """
        # Build differenced data
        if d == 0:
            x_ser = s
        else:
            x_ser = s.diff(d).dropna()
        x = x_ser.to_numpy()
        m = len(x)
        # Need enough points to split: at least (2*min_seg_len + 1)
        if m < (2 * min_seg_len + 1):
            return []

        sigma2 = _robust_sigma2(x)
        penalty = penalty_scale * sigma2 * np.log(max(m, 2))

        S1, S2 = _prefix_sums(x)

        def best_split(a: int, b: int) -> tuple[int | None, float]:
            L = b - a
            if L < 2 * min_seg_len + 1:
                return None, 0.0
            total = _seg_sse(S1, S2, a, b)
            best_k, best_gain = None, 0.0
            # search interior split points honoring min_seg_len
            for k in range(a + min_seg_len, b - min_seg_len):
                gain = total - (_seg_sse(S1, S2, a, k) + _seg_sse(S1, S2, k, b))
                if gain > best_gain:
                    best_gain, best_k = gain, k
            return best_k, best_gain

        # Greedy binary segmentation with penalty threshold
        segs: list[tuple[int, int]] = [(0, m)]
        ks: list[int] = []
        raw_gains: list[float] = []

        while segs and len(ks) < max_changes:
            candidates: list[tuple[float, int, int, int]] = []
            for a, b in segs:
                k, gain = best_split(a, b)
                if k is not None:
                    candidates.append((gain, a, k, b))
            if not candidates:
                break
            gain, a, k, b = max(candidates, key=lambda t: t[0])
            if gain <= penalty:
                break
            ks.append(k)
            raw_gains.append(gain)
            segs.remove((a, b))
            segs.extend([(a, k), (k, b)])

        # Map to original index space and build ChangePoint list
        kind_name = {0: "level", 1: "slope", 2: "accel"}[d]
        cps: list[ChangePoint] = []
        for k, g in zip(ks, raw_gains):
            i = k + d
            if 0 <= i < n:
                cps.append(
                    ChangePoint(
                        index=i,
                        timestamp=s.index[i],
                        kind=kind_name,
                        order=d,
                        gain=g,
                        penalty=float(penalty),
                        score=float(g - penalty),
                    )
                )
        return cps

    # --- Generate candidates across requested orders ---
    candidates: list[ChangePoint] = []
    for d in orders:
        candidates.extend(_binary_segmentation_on_diff(d))

    if not candidates:
        return [] if return_mode != "rich" else []

    # --- Merge via non-maximum suppression by index radius ---
    # Sort candidates by descending score, then prefer higher order when scores tie
    candidates.sort(key=lambda c: (c.score, c.order), reverse=True)

    accepted: list[ChangePoint] = []
    for cp in candidates:
        if len(accepted) >= max_changes:
            break
        too_close = any(abs(cp.index - a.index) < merge_min_gap for a in accepted)
        if not too_close:
            accepted.append(cp)

    # Final sort by index (temporal order)
    accepted.sort(key=lambda c: c.index)

    # --- Return in requested mode ---
    if return_mode == "rich":
        return accepted
    elif return_mode == "indices":
        return [cp.index for cp in accepted]
    elif return_mode == "timestamps":
        return [cp.timestamp for cp in accepted]
    else:
        raise ValueError(f"Unknown return_mode={return_mode!r}")
