import pandas as pd

from substack_analyzer.detection import compute_segment_slopes, detect_change_points, slope_around


def test_compute_segment_slopes_basic():
    idx = pd.period_range("2024-01", periods=6, freq="M").to_timestamp("M")
    s = pd.Series([100, 105, 111, 118, 126, 135], index=idx)
    segs = compute_segment_slopes(s, breakpoints=[3, 6])
    assert len(segs) == 2
    assert segs[0].start_index == 0 and segs[0].end_index == 2


def test_slope_around():
    idx = pd.period_range("2024-01", periods=12, freq="M").to_timestamp("M")
    s = pd.Series(range(100, 112), index=idx)
    pre, post = slope_around(s, event_date=idx[6], window=3)
    assert isinstance(pre, float) and isinstance(post, float)


def test_detect_change_points_small_series():
    idx = pd.period_range("2024-01", periods=5, freq="M").to_timestamp("M")
    s = pd.Series([1, 2, 3, 4, 5], index=idx)
    # Too short to detect
    assert detect_change_points(s, max_changes=3) == []
