"""
Headless E2E that exercises the real code paths without Streamlit rendering.

Run with:
    pytest -q
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

from substack_analyzer.analysis import build_events_features, compute_estimates
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params
from substack_analyzer.detection import compute_segment_slopes, detect_change_points
from substack_analyzer.model import simulate_growth
from substack_analyzer.persistence import apply_session_bundle, collect_session_bundle
from substack_analyzer.types import AdSpendSchedule, SimulationInputs


def _make_synthetic_series(n_months: int = 48, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2022-01", periods=n_months, freq="M").to_timestamp("M")
    values: list[float] = []
    s = 200.0
    for t in range(n_months):
        r = 0.12 if t < (n_months // 2) else 0.05
        K = 15000.0
        delta = r * s * (1.0 - s / K) + rng.normal(0.0, 15.0)
        s = max(s + delta, 0.0)
        values.append(s)
    return pd.Series(values, index=idx, name="Total")


def _make_paid_from_total(total: pd.Series, frac: float = 0.08, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    paid = (total * frac).astype(float)
    noise = rng.normal(0.0, 5.0, len(paid))
    paid = (paid + noise).clip(lower=0.0)
    paid.name = "Paid"
    return paid


def test_e2e_walkthrough_headless():
    # Synthetic data
    total = _make_synthetic_series()
    paid = _make_paid_from_total(total)

    # Seed session state needed by feature builders
    st.session_state.clear()
    st.session_state["start_premium"] = 80
    st.session_state["events_df"] = pd.DataFrame(
        {
            "date": [total.index[14], total.index[28]],
            "type": ["Ad spend", "Launch"],
            "persistence": ["Transient", "Persistent"],
            "cost": [2500.0, 0.0],
        }
    )
    st.session_state["import_total"] = total
    st.session_state["import_paid"] = paid

    # Change-points
    bkps = detect_change_points(total, max_changes=4, min_seg_len=3)
    assert isinstance(bkps, list)

    # Features (with exogenous)
    plot_df = pd.DataFrame({"Total": total}).set_index(total.index)
    covariates_df, features_df = build_events_features(plot_df, lam=0.5, theta=500.0, ad_file=None)
    assert {"pulse", "step", "adstock", "ad_effect_log"}.issubset(features_df.columns)
    exog = features_df["ad_effect_log"]

    # Estimates (All+Paid branch)
    est = compute_estimates(all_series=total, paid_series=paid, window_months=6)
    assert "start_free" in est and "start_premium" in est and "organic_growth" in est

    # Fit piecewise logistic with events + exog
    fit = fit_piecewise_logistic(
        total_series=total,
        breakpoints=bkps,
        events_df=st.session_state.get("events_df"),
        extra_exog=exog,
    )
    sse_naive = float((total.diff().dropna() ** 2).sum())
    assert fit.sse < sse_naive
    assert np.isfinite(fit.r2_on_deltas)

    # Recompute fitted path; must match
    s_hat = fitted_series_from_params(
        total_series=total,
        breakpoints=fit.breakpoints,
        carrying_capacity=fit.carrying_capacity,
        segment_growth_rates=fit.segment_growth_rates,
        events_df=st.session_state.get("events_df"),
        extra_exog=exog,
        gamma_pulse=fit.gamma_pulse,
        gamma_step=fit.gamma_step,
        gamma_exog=fit.gamma_exog,
    )
    assert s_hat.index.equals(total.index)
    assert np.allclose(s_hat.values, fit.fitted_series.values, atol=1e-6)

    # Segment slopes sanity: at least one segment
    segs = compute_segment_slopes(total, breakpoints=bkps)
    assert len(segs) >= 1

    # Simulation sanity
    sim = simulate_growth(
        SimulationInputs(
            starting_free_subscribers=int(est.get("start_free", 1500)),
            starting_premium_subscribers=int(est.get("start_premium", 80)),
            horizon_months=24,
            organic_monthly_growth_rate=float(est.get("organic_growth", 0.01)),
            ongoing_premium_conv_rate=0.0003,
            new_subscriber_premium_conv_rate=0.02,
            monthly_churn_rate_free=float(est.get("churn_free", 0.01)),
            monthly_churn_rate_premium=float(est.get("churn_prem", 0.01)),
            ad_spend_schedule=AdSpendSchedule.constant(2000.0),
        )
    )
    m = sim.monthly
    assert (m["free_subscribers"] >= 0).all()
    assert (m["premium_subscribers"] >= 0).all()
    assert m["cumulative_ad_spend"].is_monotonic_increasing

    # Persistence roundtrip
    bundle = collect_session_bundle(include_fit=True, include_sim=False)
    apply_session_bundle(io.BytesIO(bundle))
    it = st.session_state["import_total"]
    assert it.index.equals(total.index)
    assert np.allclose(it.values, total.values)
