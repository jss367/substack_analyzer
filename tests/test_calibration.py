import numpy as np
import pandas as pd

from substack_analyzer.calibration import fit_piecewise_logistic, forecast_piecewise_logistic


def test_fit_piecewise_logistic_minimal():
    idx = pd.period_range("2023-01", periods=8, freq="M").to_timestamp("M")
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
