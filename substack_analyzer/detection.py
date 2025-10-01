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


def detect_change_points(series: pd.Series, max_changes: int = 4) -> list[int]:
    """Detect change points emphasizing slope changes rather than level shifts.

    Strategy:
    - Work on the first difference (monthly deltas) and then its difference
      (acceleration). Large absolute acceleration indicates a slope change.
    - Pick the top-k local maxima in |acceleration| with a minimum spacing
      between change points to avoid clustered duplicates.
    - Return indices relative to the original monthly series.
    """
    s = series.dropna()
    n = s.shape[0]
    if n < 6:
        return []

    # First and second differences
    delta = s.diff().dropna()
    accel = delta.diff().dropna()
    if accel.empty:
        return []

    # Score by absolute acceleration
    score = accel.abs()

    # Identify candidate peaks (greater than neighbors)
    candidates: list[pd.Timestamp] = []
    values: list[float] = []
    accel_vals = score.to_numpy()
    accel_index = score.index
    for i in range(1, len(accel_vals) - 1):
        if accel_vals[i] >= accel_vals[i - 1] and accel_vals[i] >= accel_vals[i + 1]:
            candidates.append(accel_index[i])
            values.append(float(accel_vals[i]))

    if not candidates:
        return []

    # Sort by magnitude descending and enforce min separation (in months)
    order = sorted(range(len(candidates)), key=lambda k: values[k], reverse=True)
    min_separation = 2  # months
    selected_dates: list[pd.Timestamp] = []
    for idx in order:
        d = candidates[idx]
        # Enforce spacing relative to already selected
        if all(abs(s.index.get_loc(d) - s.index.get_loc(sd)) >= min_separation for sd in selected_dates):
            selected_dates.append(d)
        if len(selected_dates) >= max_changes:
            break

    # Map dates back to series indices
    indices = [int(s.index.get_loc(d)) for d in sorted(selected_dates)]
    # Ensure indices are within bounds [1, n-1]
    indices = [i for i in indices if 0 <= i < n]
    return indices
