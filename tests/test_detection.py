import pandas as pd

from substack_analyzer.detection import compute_segment_slopes, detect_change_points, slope_around


def test_compute_segment_slopes_basic():
    idx = pd.period_range("2024-01", periods=6, freq="M").to_timestamp("M")
    s = pd.Series([100, 105, 111, 118, 126, 135], index=idx)
    segs = compute_segment_slopes(s, breakpoints=[3, 6])
    assert len(segs) == 2
    assert segs[0].start_index == 0 and segs[0].end_index == 2


def test_compute_segment_slopes_ignores_invalid_breakpoints():
    idx = pd.period_range("2024-01", periods=5, freq="M").to_timestamp("M")
    s = pd.Series([5, 7, 9, 12, 15], index=idx)
    # Includes duplicates, values outside the valid range and unsorted input.
    segs = compute_segment_slopes(s, breakpoints=[0, 10, 3, 2, 3])
    assert [(seg.start_index, seg.end_index) for seg in segs] == [(0, 1), (2, 2), (3, 4)]


def test_slope_around():
    idx = pd.period_range("2024-01", periods=12, freq="M").to_timestamp("M")
    s = pd.Series(range(100, 112), index=idx)
    pre, post = slope_around(s, event_date=idx[6], window=3)
    assert isinstance(pre, float) and isinstance(post, float)


def test_detect_change_points_small_series():
    idx = pd.period_range("2024-01", periods=5, freq="M").to_timestamp("M")
    s = pd.Series([1, 2, 3, 4, 5], index=idx)
    # Too short to detect
    assert detect_change_points(s, max_changes=3, return_mode="indices") == []


def test_detect_change_points_single_break():
    # Piecewise linear: slope 5 for 6 months, then slope 1 for 6 months
    vals = []
    v = 100
    for _ in range(6):
        vals.append(v)
        v += 5
    for _ in range(6):
        vals.append(v)
        v += 1
    idx = pd.period_range("2024-01", periods=len(vals), freq="M").to_timestamp("M")
    input_series = pd.Series(vals, index=idx)
    bkps = detect_change_points(input_series, max_changes=3, return_mode="indices")
    assert bkps[0] in {6, 7} or bkps[1] in {6, 7}


def test_detect_change_points_respects_max_changes_and_spacing():
    # Build multiple slope changes
    parts = [(5, 4), (1, 4), (6, 4), (0, 4)]  # (slope, months)
    vals = []
    v = 50
    for slope, months in parts:
        for _ in range(months):
            vals.append(v)
            v += slope
    idx = pd.period_range("2024-01", periods=len(vals), freq="M").to_timestamp("M")
    s = pd.Series(vals, index=idx)
    bkps = detect_change_points(s, max_changes=2, return_mode="indices")
    # At most 2 per max_changes
    assert len(bkps) <= 2
    # Enforce minimum separation of 2 months
    assert all((j - i) >= 2 for i, j in zip(bkps, bkps[1:]))


def test_detect_change_points_constant_slope_returns_empty():
    idx = pd.period_range("2024-01", periods=10, freq="M").to_timestamp("M")
    s = pd.Series(range(10), index=idx)
    assert detect_change_points(s, max_changes=5, return_mode="indices") == []


def test_detect_change_points_accelerated_growth_like_chart():
    # Build a series similar to the provided chart:
    # Sep 2023..Jun 2025: slow/steady growth; Jul..Sep 2025: sharp acceleration
    idx = pd.period_range("2023-09", periods=25, freq="M").to_timestamp("M")
    values = [0]
    # Slow trend for 22 months after the first value -> 23 total so far
    for _ in range(22):
        values.append(values[-1] + 80)
    # Sharp acceleration (Jul, Aug, Sep 2025)
    values.append(values[-1] + 1600)
    values.append(values[-1] + 1600)
    # Keep exactly 25 values to match index length
    # The last append would make it 26, so skip it.

    s = pd.Series(values[:25], index=idx)

    # Expect a change-point near Jul 2025 (allow slight off-by-one tolerance)
    bkps_ts = detect_change_points(s, max_changes=3, return_mode="timestamps")
    assert any(pd.Timestamp("2025-06-30") <= ts <= pd.Timestamp("2025-08-31") for ts in bkps_ts)


def test_detect_change_points_with_cy_series():
    """
    Slow growth then a faster growth rate
    """
    # Build monthly index
    idx = pd.period_range("2023-09", periods=26, freq="M").to_timestamp("M")

    vals: list[float] = []
    v = 0.0
    jump_month = pd.Timestamp("2025-01-31")
    jump_index = list(idx).index(jump_month)
    for i, ts in enumerate(idx):
        if ts < jump_month:
            v += 80.0 + 1.2 * i
        else:
            v += 80.0 + 1.2 * i + 12 * (i - jump_index)
        vals.append(float(round(v)))

    input_series = pd.Series(vals, index=idx)
    bkps = detect_change_points(input_series, max_changes=3, return_mode="indices")
    # make sure the breakpoint is within 2 of jump_index
    assert abs(bkps[0] - jump_index) <= 2
