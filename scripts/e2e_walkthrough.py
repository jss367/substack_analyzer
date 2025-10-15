"""
End-to-end walkthrough (visual).

Run with:
    streamlit run scripts/e2e_walkthrough.py

Tip: set breakpoints anywhere in substack_analyzer/*
"""

import io
import os
from contextlib import suppress

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from substack_analyzer.analysis import build_events_features, compute_estimates
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params
from substack_analyzer.detection import compute_segment_slopes, detect_change_points
from substack_analyzer.model import simulate_growth
from substack_analyzer.persistence import apply_session_bundle, collect_session_bundle
from substack_analyzer.types import AdSpendSchedule, SimulationInputs

VISUALIZE = os.getenv("E2E_VISUALIZE", "1") != "0"


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


def _overlay_chart(actual: pd.Series, fitted: pd.Series, title: str) -> None:
    if not VISUALIZE:
        return
    df = pd.DataFrame({"Actual": actual, "Fitted": fitted}).reset_index(names="date")
    base = alt.Chart(df).encode(x=alt.X("date:T", axis=alt.Axis(format="%b %Y"), title=None))
    folded = base.transform_fold(["Actual", "Fitted"], as_=["Series", "Value"])
    chart = folded.mark_line().encode(
        y="Value:Q",
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=["Actual", "Fitted"], range=["#1f77b4", "#ff7f0e"]),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%b %Y"),
            alt.Tooltip("Series:N"),
            alt.Tooltip("Value:Q"),
        ],
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(title)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    st.set_page_config(page_title="E2E Walkthrough", layout="wide")
    st.title("Substack Analyzer: End-to-End Walkthrough")

    # 1) Synthetic data
    st.subheader("1) Generate synthetic monthly total series")
    total = _make_synthetic_series()
    if VISUALIZE:
        st.line_chart(total)

    # 2) Seed minimal state (no full clear)
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

    # Also make a Paid series to exercise All+Paid branch in estimators
    paid = _make_paid_from_total(total)
    st.session_state["import_paid"] = paid

    # 3) Change-point detection
    st.subheader("2) Detect change points")
    bkps = detect_change_points(total, max_changes=4, min_seg_len=3, return_mode="indices")
    st.write({"breakpoints": bkps})

    # 4) Feature building from events
    st.subheader("3) Build events features")
    plot_df = pd.DataFrame({"Total": total}).set_index(total.index)
    covariates_df, features_df = build_events_features(plot_df, lam=0.5, theta=500.0, ad_file=None)
    st.session_state["covariates_df"] = covariates_df
    st.session_state["features_df"] = features_df
    if VISUALIZE:
        st.write("Features head:")
        st.dataframe(features_df.head())
    exog = features_df["ad_effect_log"]

    # 5) Estimates for starting values and rates (All+Paid path)
    st.subheader("4) Compute estimates")
    est = compute_estimates(all_series=total, paid_series=paid, window_months=6)
    st.write(est)

    # 6) Fit piecewise logistic (with events + exogenous)
    st.subheader("5) Fit piecewise logistic model")
    fit = fit_piecewise_logistic(
        total_series=total,
        breakpoints=bkps,
        events_df=st.session_state.get("events_df"),
        extra_exog=exog,
    )
    _overlay_chart(total, fit.fitted_series, "Actual vs Fitted")

    # Sanity checks
    sse_naive = float((total.diff().dropna() ** 2).sum())
    _assert(fit.sse < sse_naive, "Fit SSE should beat naive delta=0 baseline")
    _assert(np.isfinite(fit.r2_on_deltas) and fit.r2_on_deltas > 0.1, "R² on deltas unexpectedly low")

    # 7) Recompute fitted series from parameters (sanity)
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
    st.write("Recomputed fitted series aligned:", bool(s_hat.index.equals(total.index)))
    _assert(np.allclose(s_hat.values, fit.fitted_series.values, atol=1e-6), "Param refit mismatch")

    # 8) Segment slopes for reporting
    st.subheader("6) Segment slopes")
    segs = compute_segment_slopes(total, breakpoints=bkps)
    st.write([s.__dict__ for s in segs])

    # 9) Forward simulation
    st.subheader("7) Forward simulation")
    sim_inputs = SimulationInputs(
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
    sim = simulate_growth(sim_inputs)
    if VISUALIZE:
        st.dataframe(sim.monthly.head())

    # 10) Persistence roundtrip
    st.subheader("8) Persistence roundtrip")
    bundle = collect_session_bundle(include_fit=True, include_sim=False)
    st.write({"bundle_bytes": len(bundle)})
    if st.button("Apply bundle to session"):
        with suppress(Exception):
            bio = io.BytesIO(bundle)
            apply_session_bundle(bio)
    st.write({"has_import_total_after_apply": ("import_total" in st.session_state)})

    # Extra invariants
    m = sim.monthly
    _assert((m["free_subscribers"] >= 0).all() and (m["premium_subscribers"] >= 0).all(), "Negative counts")
    _assert(m["cumulative_ad_spend"].is_monotonic_increasing, "Cumulative spend must be monotone")


if __name__ == "__main__":
    main()
