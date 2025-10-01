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


def _robust_z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return pd.Series(dtype=float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad <= 0:
        # fall back to std if MAD is zero (flat series)
        std = x.std(ddof=1)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std
    # 0.6745 makes MAD comparable to std for normal data
    return 0.6745 * (x - med) / mad


def _local_peaks(y: pd.Series) -> pd.Index:
    """Simple 1D peak detector: strictly >= neighbors; ignores endpoints."""
    if y.empty or len(y) < 3:
        return pd.Index([])
    vals = y.values
    idx = []
    for i in range(1, len(vals) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            idx.append(y.index[i])
    return pd.Index(idx)


def detect_change_points(
    total: pd.Series,
    paid: Optional[pd.Series] = None,
    max_changes: Optional[int] = 4,
    consider_ratio: bool = True,
    min_separation: str = "60D",  # time-based separation
    smooth_window: int = 3,  # small rolling median smoothing of scores
    return_scores: bool = True,
) -> tuple[list[pd.Timestamp], Optional[pd.DataFrame]]:
    """
    Detect visually obvious 'something happened' points in one or two time series
    (e.g., Substack total and paid subscribers).

    Features scored (robust z):
      - Level jumps: |Δ series|
      - Slope changes: |Δ² series| (acceleration)
      - If paid provided and consider_ratio=True: same features on ratio = paid/total

    Steps:
      1) Align to common DatetimeIndex, drop NaNs, sort by time.
      2) Build features (abs diff and abs accel) for available series (+ ratio).
      3) Robust z-score each feature; smooth each with rolling median(window).
      4) Combined score = max across features at each timestamp.
      5) Pick local peaks in combined score; enforce time-based min_separation.
      6) If max_changes is set, keep the top-k by combined score.

    Returns:
      - List of pd.Timestamp change points (sorted).
      - Optional DataFrame of per-feature and combined scores (if return_scores).
    """
    if total is None or total.empty:
        return [], None

    # 1) Align & clean
    frames = {"total": total}
    if paid is not None:
        frames["paid"] = paid
    df = pd.concat(frames, axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all")
    if df.shape[0] < 6:
        return [], None

    # 2) Build features
    feats: dict[str, pd.Series] = {}

    def add_feats(prefix: str, s: pd.Series):
        s = s.dropna()
        if s.shape[0] < 6:
            return
        d1 = s.diff()
        d2 = d1.diff()
        feats[f"{prefix}_abs_delta"] = d1.abs()
        feats[f"{prefix}_abs_accel"] = d2.abs()

    add_feats("total", df["total"])
    if "paid" in df:
        add_feats("paid", df["paid"])
        if consider_ratio:
            # conversion rate; guard division & extreme values
            ratio = (df["paid"] / df["total"]).replace([np.inf, -np.inf], np.nan).clip(0, 1)
            add_feats("ratio", ratio)

    # If we ended up with no features (all too short), bail
    if not feats:
        return [], None

    # 3) Robust z-score and smooth
    zfeats: dict[str, pd.Series] = {}
    for k, s in feats.items():
        # align to shared index by reindex to df.index (NaNs ok)
        s = s.reindex(df.index)
        z = _robust_z(s.dropna())
        z = z.reindex(df.index).fillna(0.0)
        if smooth_window and smooth_window > 1:
            z = z.rolling(smooth_window, center=True, min_periods=1).median()
        zfeats[k] = z

    zdf = pd.DataFrame(zfeats, index=df.index).fillna(0.0)

    # 4) Combined score = max across features at each timestamp
    zdf["combined"] = zdf.max(axis=1)

    # 5) Local peaks on combined
    peaks_idx = _local_peaks(zdf["combined"])
    if len(peaks_idx) == 0:
        # maybe the series is short or very smooth; consider taking top raw scores
        peaks_idx = zdf["combined"].nlargest(min(len(zdf), (max_changes or 3))).index

    # Enforce time-based min separation
    min_sep = pd.Timedelta(min_separation)
    selected = []
    peaks = zdf.loc[peaks_idx, "combined"].sort_values(ascending=False)
    for ts, _score in peaks.items():
        if not selected:
            selected.append(ts)
        else:
            if all(abs(ts - s) >= min_sep for s in selected):
                selected.append(ts)
        if max_changes is not None and len(selected) >= max_changes:
            break

    selected = sorted(selected)

    if return_scores:
        return selected, zdf
    else:
        return selected, None
