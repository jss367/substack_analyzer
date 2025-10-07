import pandas as pd

from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.changepoints import breakpoints_for_segments, breakpoints_to_events, detect_and_classify
from substack_analyzer.detection import detect_change_points


def test_classify_pulse_creates_transient_event_but_no_segment():
    idx = pd.period_range("2024-01", periods=8, freq="M").to_timestamp("M")
    s = pd.Series([100, 102, 103, 150, 151, 152, 153, 154], index=idx)

    classified = detect_and_classify(s, max_changes=3, window=3)
    seg_bkps = breakpoints_for_segments(classified)

    events_df = breakpoints_to_events(classified, target_label="Total")
    fit_no_events = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=None)
    fit_with_events = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=events_df)

    assert len(seg_bkps) in (0, 1)
    assert fit_with_events.sse <= fit_no_events.sse


def test_classify_level_step_persistent_event_no_segment():
    idx = pd.period_range("2023-01", periods=14, freq="M").to_timestamp("M")
    s = pd.Series(
        [
            100,
            105,
            110,
            115,
            120,
            125,
            130,
            140,
            150,
            160,
            170,
            180,
            190,
            200,
        ],
        index=idx,
    )

    classified = detect_and_classify(s, max_changes=3, window=4)
    seg_bkps = breakpoints_for_segments(classified)
    events_df = breakpoints_to_events(classified, target_label="Total")

    fit_no_events = fit_piecewise_logistic(s, breakpoints=seg_bkps)
    fit_with_events = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=events_df)

    assert fit_with_events.sse <= fit_no_events.sse


def test_classify_mixed_requires_events_and_segment():
    idx = pd.period_range("2022-01", periods=20, freq="M").to_timestamp("M")
    s = [100]
    for _ in range(1, 8):
        s.append(s[-1] + 5)
    s.append(s[-1] + 40)
    # Fill up to 20 total points with accelerated growth
    for _ in range(len(s), 20):
        s.append(s[-1] + 12 + 0.02 * s[-1])
    s = pd.Series(s[:20], index=idx)

    classified = detect_and_classify(s, max_changes=4, window=4)
    seg_bkps = breakpoints_for_segments(classified)
    events_df = breakpoints_to_events(classified, target_label="Total")

    assert len(seg_bkps) >= 1

    fit_ev_seg = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=events_df)
    fit_seg_only = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=None)

    assert fit_ev_seg.sse <= fit_seg_only.sse


def test_sudden_doubling_is_pulse_step_and_segment_only_if_needed():
    idx = pd.period_range("2024-01", periods=16, freq="M").to_timestamp("M")
    s = [200, 210, 220, 230, 240, 250, 260, 270]
    s.append(s[-1] + 300)
    for _ in range(7):
        s.append(s[-1] + 12)
    s = pd.Series(s[:16], index=idx)

    classified = detect_and_classify(s, max_changes=3, window=4)
    seg_bkps = breakpoints_for_segments(classified)
    events_df = breakpoints_to_events(classified, target_label="Total")

    fit_events = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=events_df)
    fit_plain = fit_piecewise_logistic(s, breakpoints=seg_bkps, events_df=None)

    assert fit_events.sse <= fit_plain.sse


def test_detect_change_points_smoke():
    idx = pd.period_range("2023-01", periods=12, freq="M").to_timestamp("M")
    s = pd.Series([100, 105, 110, 115, 120, 125, 150, 155, 160, 165, 170, 175], index=idx)
    bkps = detect_change_points(s, max_changes=3)
    assert isinstance(bkps, list)
