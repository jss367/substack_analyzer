import numpy as np
import pandas as pd
import pytest

from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params, forecast_piecewise_logistic
from substack_analyzer.changepoints import breakpoints_for_segments, detect_and_classify


def test_fit_piecewise_logistic_minimal():
    idx = pd.period_range("2023-01", periods=8, freq="M").to_timestamp("M")
    # Simple increasing series
    s = pd.Series(np.linspace(100, 200, num=8), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=[4])
    assert fit.carrying_capacity > 0
    assert len(fit.segment_growth_rates) == 2
    assert len(fit.fitted_series) == len(s)


@pytest.mark.parametrize(
    "case",
    [
        {
            "description": "top-tier newsletter saturating after sustained marketing",
            "start": 15_000.0,
            "months": 48,
            "breakpoints": [12, 24, 36],
            "carrying_capacity": 500_000.0,
            "segment_rates": [0.45, 0.30, 0.18, 0.12],
            "gamma_pulse": 3_500.0,
            "gamma_step": 1_200.0,
            "events": lambda idx: pd.DataFrame(
                {
                    "date": [idx[14], idx[26]],
                    "type": ["Brand Campaign", "Referral push"],
                    "persistence": ["persistent", "transient"],
                    "cost": [1.0, 1.0],
                }
            ),
            "sse_threshold": 1e-8,
        },
        {
            "description": "small newsletter that eventually breaks out",
            "start": 300.0,
            "months": 42,
            "breakpoints": [18, 30],
            "carrying_capacity": 90_000.0,
            "segment_rates": [0.05, 0.10, 0.26],
            "gamma_pulse": 0.0,
            "gamma_step": 80.0,
            "events": lambda idx: pd.DataFrame(
                {
                    "date": [idx[24]],
                    "type": ["Product Launch"],
                    "persistence": ["persistent"],
                    "cost": [1.0],
                }
            ),
            "sse_threshold": 1e-10,
        },
        {
            "description": "niche newsletter that stays relatively small but steady",
            "start": 800.0,
            "months": 36,
            "breakpoints": [],
            "carrying_capacity": 4_000.0,
            "segment_rates": [0.04],
            "gamma_pulse": 0.0,
            "gamma_step": 0.0,
            "events": lambda idx: None,
            "sse_threshold": 1e-10,
        },
        {
            "description": "mid-sized publication with seasonal campaigns and a big conference push",
            "start": 2_500.0,
            "months": 30,
            "breakpoints": [10, 20],
            "carrying_capacity": 75_000.0,
            "segment_rates": [0.07, 0.09, 0.12],
            "gamma_pulse": 420.0,
            "gamma_step": 260.0,
            "events": lambda idx: pd.DataFrame(
                {
                    "date": [idx[5], idx[11], idx[17], idx[23]],
                    "type": ["Guest Post", "Holiday Promo", "Guest Post", "Conference"],
                    "persistence": ["transient", "transient", "transient", "persistent"],
                    "cost": [1.0, 1.5, 1.0, 1.0],
                }
            ),
            "sse_threshold": 1e-8,
        },
    ],
    ids=lambda case: case["description"],
)
def test_fit_piecewise_logistic_handles_realistic_growth_profiles(case):
    idx = pd.period_range("2020-01", periods=case["months"], freq="M").to_timestamp("M")
    base_series = pd.Series([case["start"]] * len(idx), index=idx)

    events_df = case["events"](idx)

    total_series = fitted_series_from_params(
        total_series=base_series,
        breakpoints=case["breakpoints"],
        carrying_capacity=case["carrying_capacity"],
        segment_growth_rates=case["segment_rates"],
        events_df=events_df,
        gamma_pulse=case["gamma_pulse"],
        gamma_step=case["gamma_step"],
    )

    fit = fit_piecewise_logistic(
        total_series,
        breakpoints=case["breakpoints"],
        events_df=events_df,
        k_grid=[case["carrying_capacity"]],
    )

    assert fit.sse < case["sse_threshold"], f"{case['description']} SSE too high: {fit.sse}"
    assert np.isclose(
        fit.carrying_capacity,
        case["carrying_capacity"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        fit.segment_growth_rates,
        case["segment_rates"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isclose(fit.gamma_pulse, case["gamma_pulse"], atol=1e-6)
    assert np.isclose(fit.gamma_step, case["gamma_step"], atol=1e-6)
    assert fit.r2_on_deltas > 0.999999


def test_forecast_piecewise_logistic_shapes():
    out = forecast_piecewise_logistic(
        last_value=150.0,
        months_ahead=6,
        carrying_capacity=1000.0,
        segment_growth_rate=0.05,
    )
    assert out.shape == (6,)
    assert (out >= 0).all()


def test_fit_piecewise_logistic_requires_minimum_length():
    idx = pd.period_range("2024-01", periods=3, freq="M").to_timestamp("M")
    s = pd.Series([100, 101, 102], index=idx)
    try:
        fit_piecewise_logistic(s, breakpoints=[])
        assert False, "Expected ValueError for series shorter than 4 months"
    except ValueError:
        pass


def test_fit_piecewise_logistic_single_segment_when_no_breakpoints():
    idx = pd.period_range("2023-01", periods=10, freq="M").to_timestamp("M")
    s = pd.Series(np.linspace(50, 150, num=10), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=[])
    assert len(fit.segment_growth_rates) == 1
    assert fit.carrying_capacity > s.max()


def test_fit_piecewise_logistic_two_segment_rates_ordered():
    # Build a series with faster growth early, slower later
    vals = []
    v = 100
    for _ in range(5):
        vals.append(v)
        v += 20
    for _ in range(5):
        vals.append(v)
        v += 5
    idx = pd.period_range("2023-01", periods=len(vals), freq="M").to_timestamp("M")
    s = pd.Series(vals, index=idx)

    fit = fit_piecewise_logistic(s, breakpoints=[5])
    assert len(fit.segment_growth_rates) == 2
    assert fit.segment_growth_rates[0] > fit.segment_growth_rates[1]


def test_fit_piecewise_logistic_events_reduce_sse():
    # Series with a one-time spike in delta that events can explain
    idx = pd.period_range("2024-01", periods=7, freq="M").to_timestamp("M")
    s = pd.Series([100, 100, 120, 120, 120, 120, 120], index=idx)

    # No events
    fit_no_events = fit_piecewise_logistic(s, breakpoints=[])

    # Transient event at the month of the spike (case-insensitive accepted)
    events_df = pd.DataFrame({"date": [idx[2]], "type": ["promo"], "persistence": ["Transient"]})
    fit_with_events = fit_piecewise_logistic(s, breakpoints=[], events_df=events_df)

    assert fit_with_events.sse < fit_no_events.sse


def test_fit_piecewise_logistic_exogenous_included_and_handles_nans():
    # Construct deltas driven by an exogenous signal
    idx = pd.period_range("2024-01", periods=8, freq="M").to_timestamp("M")
    exog_deltas = [0, 2, 0, 2, 0, 2, 0]
    s_vals = [100]
    for d in exog_deltas:
        s_vals.append(s_vals[-1] + d)
    s = pd.Series(s_vals, index=idx)

    # exog aligned to y index (length len(s)-1), include NaNs which should be treated as zeros
    exog_series = pd.Series([0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0], index=idx[1:])

    fit_without_exog = fit_piecewise_logistic(s, breakpoints=[])
    fit_with_exog = fit_piecewise_logistic(s, breakpoints=[], extra_exog=exog_series)

    assert fit_with_exog.gamma_exog is not None
    assert fit_with_exog.sse < fit_without_exog.sse


def test_fit_piecewise_logistic_three_breaks_mixed_persistence_events():
    """
    Build a synthetic series with three breakpoints and mixed persistence events.
    """
    idx = pd.period_range("2022-01", periods=42, freq="M").to_timestamp("M")

    # Use the simulator to generate a ground-truth series under the same dynamic
    base_series = pd.Series([30.0] * len(idx), index=idx)
    breakpoints = [24, 32, 36]  # three breaks → four segments
    segment_growth_rates = [0.010, 0.017, 0.04, 0.02]

    events_df = pd.DataFrame(
        {
            "date": [idx[20], idx[26], idx[34]],
            "type": ["campaign A", "promo", "campaign B"],
            "persistence": ["persistent", "transient", "persistent"],
        }
    )

    input_series = fitted_series_from_params(
        total_series=base_series,
        breakpoints=breakpoints,
        carrying_capacity=150.0,
        segment_growth_rates=segment_growth_rates,
        events_df=events_df,
        gamma_pulse=3.0,
        gamma_step=0.6,
    )

    # Fit with correct mixed persistence
    fit_mixed = fit_piecewise_logistic(input_series, breakpoints=breakpoints, events_df=events_df)

    # Fit with a mis-specified events table (all transient)
    events_all_transient = events_df.copy()
    events_all_transient["persistence"] = "transient"
    fit_all_transient = fit_piecewise_logistic(input_series, breakpoints=breakpoints, events_df=events_all_transient)

    # Expectations: correct persistence should explain deltas better (lower SSE),
    # both gamma coefficients should be utilized, and four segment rates returned.
    assert len(fit_mixed.segment_growth_rates) == 4
    assert abs(fit_mixed.gamma_step) > 0 or abs(fit_mixed.gamma_pulse) > 0
    assert fit_mixed.sse <= fit_all_transient.sse


def test_fit_piecewise_logistic_on_gm_series():

    vals = [
        0,
        0,
        0,
        2,
        3,
        4,
        4,
        4,
        4,
        5,
        7,
        30,
        31,
        31,
        32,
        33,
        33,
        33,
        33,
        35,
        36,
        36,
        35,
        36,
        39,
        42,
        42,
        44,
        45,
        45,
        47,
        50,
        56,
        60,
        82,
        93,
        104,
        109,
        108,
        116,
        121,
        124,
        128,
        128,
        131,
        134,
        133,
        134,
    ]
    idx = pd.period_range("2021-10", periods=len(vals), freq="M").to_timestamp("M")
    input_series = pd.Series(vals, index=idx)
    # now plot the series
    # import matplotlib.pyplot as plt

    # plt.plot(input_series)
    # plt.show()

    # Detect and classify, then use only segment-worthy breakpoints
    classified = detect_and_classify(input_series, max_changes=4, window=6)
    bkps = breakpoints_for_segments(classified)
    # Optionally could pass events from classification; not required for this test
    fit = fit_piecewise_logistic(input_series, breakpoints=bkps)

    # Basic shape and plausibility checks
    assert len(fit.fitted_series) == len(input_series)
    assert fit.carrying_capacity > input_series.max()
    assert len(fit.segment_growth_rates) == (len(bkps) + 1 if bkps else 1)
    assert fit.sse >= 0.0


def test_fit_piecewise_logistic_with_cy_series():
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

    bkps = [16]
    fit = fit_piecewise_logistic(input_series, breakpoints=bkps)

    # from substack_analyzer.plot_utils import plot_fit_vs_actual

    # plot_fit_vs_actual(input_series, fit)

    assert len(fit.fitted_series) == len(input_series)
    assert fit.carrying_capacity > input_series.max()
    assert len(fit.segment_growth_rates) == (len(bkps) + 1 if bkps else 1)
    if len(fit.segment_growth_rates) >= 2:
        # Later growth should be faster than the early growth in this scenario
        assert fit.segment_growth_rates[-1] > fit.segment_growth_rates[0]
    assert fit.sse >= 0.0
