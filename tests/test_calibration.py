import numpy as np
import pandas as pd

from substack_analyzer.calibration import fit_piecewise_logistic, forecast_piecewise_logistic


def test_fit_piecewise_logistic_minimal():
    idx = pd.period_range("2023-01", periods=8, freq="ME").to_timestamp("ME")
    # Simple increasing series
    s = pd.Series(np.linspace(100, 200, num=8), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=[4])
    assert fit.carrying_capacity > 0
    assert len(fit.segment_growth_rates) == 2
    assert len(fit.fitted_series) == len(s)


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
    idx = pd.period_range("2024-01", periods=3, freq="ME").to_timestamp("ME")
    s = pd.Series([100, 101, 102], index=idx)
    try:
        fit_piecewise_logistic(s)
        assert False, "Expected ValueError for series shorter than 4 months"
    except ValueError:
        pass


def test_fit_piecewise_logistic_single_segment_when_no_breakpoints():
    idx = pd.period_range("2023-01", periods=10, freq="ME").to_timestamp("ME")
    s = pd.Series(np.linspace(50, 150, num=10), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=None)
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
    idx = pd.period_range("2023-01", periods=len(vals), freq="ME").to_timestamp("ME")
    s = pd.Series(vals, index=idx)

    fit = fit_piecewise_logistic(s, breakpoints=[5])
    assert len(fit.segment_growth_rates) == 2
    assert fit.segment_growth_rates[0] > fit.segment_growth_rates[1]


def test_fit_piecewise_logistic_events_reduce_sse():
    # Series with a one-time spike in delta that events can explain
    idx = pd.period_range("2024-01", periods=7, freq="ME").to_timestamp("ME")
    s = pd.Series([100, 100, 120, 120, 120, 120, 120], index=idx)

    # No events
    fit_no_events = fit_piecewise_logistic(s, breakpoints=None)

    # Transient event at the month of the spike (aligned to y index)
    events_df = pd.DataFrame(
        {
            "date": [idx[2]],
            "type": ["promo"],
            "persistence": ["transient"],
        }
    )
    fit_with_events = fit_piecewise_logistic(s, breakpoints=None, events_df=events_df)

    assert fit_with_events.sse < fit_no_events.sse


def test_fit_piecewise_logistic_exogenous_included_and_handles_nans():
    # Construct deltas driven by an exogenous signal
    idx = pd.period_range("2024-01", periods=8, freq="ME").to_timestamp("ME")
    exog_deltas = [0, 2, 0, 2, 0, 2, 0]
    s_vals = [100]
    for d in exog_deltas:
        s_vals.append(s_vals[-1] + d)
    s = pd.Series(s_vals, index=idx)

    # exog aligned to y index (length len(s)-1), include NaNs which should be treated as zeros
    exog_series = pd.Series([0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0], index=idx[1:])

    fit_without_exog = fit_piecewise_logistic(s, breakpoints=None)
    fit_with_exog = fit_piecewise_logistic(s, breakpoints=None, extra_exog=exog_series)

    assert fit_with_exog.gamma_exog is not None
    assert fit_with_exog.sse < fit_without_exog.sse
