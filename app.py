from __future__ import annotations

import math
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from substack_analyzer.analysis import (
    build_events_features,
    compute_estimates,
    derive_adds_churn,
    plot_series,
    read_series,
)
from substack_analyzer.calibration import fit_piecewise_logistic, forecast_piecewise_logistic
from substack_analyzer.detection import compute_segment_slopes, detect_change_points, slope_around
from substack_analyzer.model import simulate_growth
from substack_analyzer.persistence import apply_session_bundle, collect_session_bundle
from substack_analyzer.types import AdSpendSchedule, SimulationInputs
from substack_analyzer.ui import format_currency as ui_format_currency
from substack_analyzer.ui import format_date_badges as ui_format_date_badges
from substack_analyzer.ui import inject_brand_styles as ui_inject_brand_styles
from substack_analyzer.ui import render_brand_header as ui_render_brand_header

# Asset paths
ASSETS_DIR = Path(__file__).parent / "logos"
LOGO_ICON = ASSETS_DIR / "ROPI_IconDark Green_RGB.png"
LOGO_FULL = ASSETS_DIR / "RPI_Full logo_Dark Green_RGB.png"

st.set_page_config(
    page_title="Substack Ads ROI Simulator",
    layout="wide",
    page_icon=str(LOGO_ICON) if LOGO_ICON.exists() else (str(LOGO_FULL) if LOGO_FULL.exists() else None),
)


def _inject_brand_styles() -> None:
    ui_inject_brand_styles()


def render_brand_header() -> None:
    ui_render_brand_header(LOGO_FULL, LOGO_ICON)


def format_currency(value: float) -> str:
    return ui_format_currency(value)


def _get_state(key: str, default):
    return st.session_state.get(key, default)


def _apply_pending_state_updates() -> None:
    """Apply any deferred session state updates before widgets render."""

    pending = st.session_state.pop("_pending_state_update", None)
    if isinstance(pending, dict):
        for k, v in pending.items():
            st.session_state[k] = v


# Apply brand styles and sidebar logo once
_inject_brand_styles()
_apply_pending_state_updates()
if LOGO_FULL.exists() or LOGO_ICON.exists():
    st.sidebar.image(str(LOGO_FULL if LOGO_FULL.exists() else LOGO_ICON), use_container_width=True)


def _format_date_badges(dates: list[pd.Timestamp | str]) -> str:
    # Backward-compatible wrapper delegating to ui helper
    return ui_format_date_badges(dates)


def read_head_preview(fh, has_header: bool, nrows: int = 5) -> pd.DataFrame:
    """Read a small preview from an uploaded CSV/XLSX without consuming the file pointer."""
    try:
        if fh.name.lower().endswith((".xlsx", ".xls")):
            tmp = pd.read_excel(fh, header=0 if has_header else None, nrows=nrows)
        else:
            tmp = pd.read_csv(fh, header=0 if has_header else None, nrows=nrows)
    finally:
        with suppress(Exception):
            fh.seek(0)
    return tmp


def upload_panel(
    title: str,
    help_hint: str,
    key_prefix: str,
    default_header: bool = False,
) -> tuple[Optional[Any], bool, Optional[int], Optional[int]]:
    """Shared UI for file upload + optional header + column choices."""
    file_obj = st.file_uploader(title, type=["csv", "xlsx", "xls"], key=f"{key_prefix}_file", help=help_hint)
    has_header = st.checkbox(
        f"{key_prefix.capitalize()} file has header row", value=default_header, key=f"{key_prefix}_has_header"
    )
    date_sel: Optional[int] = 0
    count_sel: Optional[int] = 1
    if file_obj is not None:
        try:
            head = read_head_preview(file_obj, has_header, nrows=5)
            ncols = head.shape[1]
            date_sel = st.selectbox(
                f"{key_prefix.capitalize()}: date column (index)",
                list(range(ncols)),
                index=0,
                key=f"{key_prefix}_date_sel",
            )
            count_sel = st.selectbox(
                f"{key_prefix.capitalize()}: count column (index)",
                list(range(ncols)),
                index=min(1, max(ncols - 1, 0)),
                key=f"{key_prefix}_count_sel",
            )
        except Exception as e:
            st.error(f"Could not read {key_prefix.capitalize()} file: {e}")
            file_obj = None
    return file_obj, has_header, date_sel, count_sel


def emit_observations(plot_df: pd.DataFrame) -> None:
    """Stage 1 output: observations_df (current granularity)."""
    idx = plot_df.index
    total = plot_df.get("Total")
    paid = plot_df.get("Paid")
    if total is not None and paid is not None:
        free = (total.astype(float) - paid.astype(float)).clip(lower=0)
    elif total is not None:
        free = total.astype(float) - float(_get_state("start_premium", 0))
    else:
        free = pd.Series(index=idx, dtype=float)

    obs = pd.DataFrame(
        {
            "active_total": (total.astype(float) if total is not None else pd.Series(index=idx, dtype=float)),
            "active_paid": (paid.astype(float) if paid is not None else pd.Series(index=idx, dtype=float)),
            "active_free": free.astype(float),
            "is_imputed": False,
        },
        index=idx,
    )
    obs.index.name = "date"
    st.session_state["observations_df"] = obs
    with st.expander("Stage 1 output: observations_df", expanded=False):
        st.dataframe(obs.reset_index(), use_container_width=True)
        st.download_button(
            "Download observations.csv",
            data=obs.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="observations.csv",
            mime="text/csv",
        )


def trend_detection_ui(plot_df: pd.DataFrame, target_col: Optional[str]) -> list[int]:
    st.caption("Detected trend changes help annotate what happened (e.g., shout-outs, ad spend).")
    if target_col is None:
        return []
    max_bkps = st.slider("Max changes to detect", 0, 8, 3, 1, key="max_changes_detect")
    try:
        bkps = detect_change_points(plot_df[target_col], max_changes=max_bkps)
    except Exception:
        bkps = []

    if bkps:
        s_idx = plot_df[target_col].dropna().index
        dates = [pd.to_datetime(s_idx[i]) for i in bkps if i < len(s_idx)]
        st.markdown(f"**Detected change dates (on {target_col}):**")
        st.markdown(_format_date_badges(dates), unsafe_allow_html=True)
        st.caption("Tip: To adjust these, use the draggable timeline below (purple bars).")
        # Persist detected dates for the Events editor button
        try:
            change_dates_for_events = [s_idx[i - 1] if i > 0 else s_idx[i] for i in bkps if i < len(s_idx)]
            st.session_state["detected_change_dates"] = [pd.to_datetime(d) for d in change_dates_for_events]
            st.session_state["detected_target_col"] = target_col
        except Exception:
            st.session_state.pop("detected_change_dates", None)
            st.session_state.pop("detected_target_col", None)
    return bkps or []


def events_editor() -> None:
    st.subheader("Stage 2: Events & annotations")
    st.caption("Track shout-outs, ad campaigns, launches, etc. Dates must match the series timeline.")
    # Offer to add detected change dates directly here
    with st.container():
        add_col1, _ = st.columns([1, 3])
        with add_col1:
            if st.button("Add detected change dates to Events"):
                try:
                    change_dates = [pd.to_datetime(d).date() for d in st.session_state.get("detected_change_dates", [])]
                    if not change_dates:
                        st.info("No detected change dates available. Run detection below.")
                    else:
                        target_col = st.session_state.get("detected_target_col", "series")
                        seeded = pd.DataFrame(
                            {
                                "date": change_dates,
                                "type": ["Change"] * len(change_dates),
                                "notes": [f"Detected change in {target_col}"] * len(change_dates),
                                "cost": [0.0] * len(change_dates),
                            }
                        )
                        existing = st.session_state.get("events_df")
                        merged = (
                            pd.concat([existing, seeded], ignore_index=True)
                            if (existing is not None and not existing.empty)
                            else seeded
                        )
                        st.session_state["events_df"] = merged
                        st.success("Added detected change dates to Events.")
                        st.rerun()
                except Exception:
                    st.info("Could not add detected dates. Try again after loading data.")
    default_events = pd.DataFrame(columns=["date", "type", "persistence", "notes", "cost"])
    events_df = st.session_state.get("events_df", default_events)

    edited = st.data_editor(
        events_df,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "type": st.column_config.SelectboxColumn(
                "Type",
                options=[
                    "Ad spend",
                    "Shout-out",
                    "Viral post",
                    "Launch",
                    "Paywall change",
                    "Change",
                    "Other",
                ],
                width="medium",
            ),
            "persistence": st.column_config.SelectboxColumn(
                "Persistence", options=["Transient", "Persistent"], width="medium"
            ),
            "notes": st.column_config.TextColumn("Notes", width="large"),
            "cost": st.column_config.NumberColumn(
                "Cost ($)", step=10.0, min_value=0.0, format="%.2f", help="For Ad spend ROI calc"
            ),
        },
        use_container_width=True,
        key="events_editor",
    )
    with suppress(Exception):
        # Auto-fill persistence from type when not provided
        if "persistence" not in edited.columns:
            edited["persistence"] = None
        type_to_persistence = {
            "ad spend": "Transient",
            "ad": "Transient",
            "shout-out": "Transient",
            "viral post": "Transient",
            "launch": "Persistent",
            "paywall change": "Persistent",
            "change": "Transient",
        }
        with suppress(Exception):
            typed_lower = edited.get("type").astype(str).str.lower()
            need_fill = edited.get("persistence").isna() | (edited.get("persistence").astype(str).str.len() == 0)
            edited.loc[need_fill, "persistence"] = typed_lower.map(type_to_persistence)
        if "cost" in edited.columns:
            edited["cost"] = pd.to_numeric(edited["cost"], errors="coerce")
        if "date" in edited.columns:
            edited["date"] = (
                pd.to_datetime(edited["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M").dt.date
            )
    st.session_state["events_df"] = edited


def events_features_ui(plot_df: pd.DataFrame) -> None:
    with st.expander("Stage 2: Events & Features (monthly)", expanded=False):
        st.caption("Encodes pulse/step features from Events and optional ad spend adstock + log transform.")
        cov_col1, cov_col2 = st.columns(2)
        with cov_col1:
            ad_file = st.file_uploader(
                "Optional: Ad spend CSV (date, spend)", type=["csv", "xlsx", "xls"], key="ad_csv"
            )
        with cov_col2:
            lam = st.slider("Adstock lambda (carryover)", 0.0, 0.99, 0.5, 0.01)
            theta = st.number_input("Log transform theta", min_value=1.0, value=500.0, step=50.0)

        covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
        st.session_state["covariates_df"] = covariates_df
        st.session_state["features_df"] = features_df
        st.markdown("**Outputs**: `events_df` (above), `covariates_df`, `features_df`.")
        st.dataframe(features_df.reset_index(), use_container_width=True)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "Download covariates.csv",
                data=covariates_df.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="covariates.csv",
                mime="text/csv",
            )
        with dcol2:
            st.download_button(
                "Download features.csv",
                data=features_df.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="features.csv",
                mime="text/csv",
            )


def adds_and_churn_ui(plot_df: pd.DataFrame) -> None:
    with st.expander("Stage 3: Adds & Churn (monthly)", expanded=False):
        st.caption("Totals-only path: derive gross adds from net deltas using churn rate heuristics.")

        c1, c2 = st.columns(2)
        with c1:
            churn_free_est = st.number_input(
                "Monthly churn rate (free)",
                min_value=0.0,
                max_value=1.0,
                value=float(_get_state("churn_free", 0.0)),
                step=0.001,
                format="%0.3f",
            )
        with c2:
            churn_paid_est = st.number_input(
                "Monthly churn rate (paid)",
                min_value=0.0,
                max_value=1.0,
                value=float(_get_state("churn_prem", 0.0)),
                step=0.001,
                format="%0.3f",
            )

        adds_df, churn_df = derive_adds_churn(plot_df, churn_free_est=churn_free_est, churn_paid_est=churn_paid_est)
        if not adds_df.empty or not churn_df.empty:
            st.session_state["adds_df"] = adds_df
            st.session_state["churn_df"] = churn_df
            st.markdown("**Outputs**: `adds_df`, `churn_df` (monthly, heuristics).")
            st.dataframe(adds_df.reset_index(), use_container_width=True)
            st.dataframe(churn_df.reset_index(), use_container_width=True)
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Download adds.csv",
                    data=adds_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="adds.csv",
                    mime="text/csv",
                )
            with b2:
                st.download_button(
                    "Download churn.csv",
                    data=churn_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="churn.csv",
                    mime="text/csv",
                )


def quick_fit_ui(plot_df: pd.DataFrame, breakpoints: list[int]) -> None:
    st.subheader("Stage 4: Model fitting (Quick Fit)")
    st.caption("Fits on Total (preferred) or Free if Total is unavailable. Uses detected change points as segments.")
    horizon_ahead = st.slider("Forecast months ahead", 0, 36, 12, 1)
    if st.button("Fit model and overlay"):
        try:
            fit_series_source = plot_df.get("Total") if "Total" in plot_df.columns else plot_df.get("Free")
            if fit_series_source is None or fit_series_source.empty:
                st.info("Need Total or Free series to fit.")
                return
            fit = fit_piecewise_logistic(
                total_series=fit_series_source, breakpoints=breakpoints, events_df=st.session_state.get("events_df")
            )
            st.session_state["pwlog_fit"] = fit

            overlay_df = pd.DataFrame(
                {"Actual": fit_series_source, "Fitted": fit.fitted_series.reindex(fit_series_source.index)}
            )
            base_overlay = alt.Chart(overlay_df.reset_index().rename(columns={"index": "date"})).encode(
                x=alt.X("date:T", title="Date")
            )
            actual_line = (
                base_overlay.transform_fold(["Actual"], as_=["Series", "Value"])
                .mark_line()
                .encode(y="Value:Q", color=alt.Color("Series:N", scale=alt.Scale(range=["#1f77b4"])))
            )
            fitted_line = (
                base_overlay.transform_fold(["Fitted"], as_=["Series", "Value"])
                .mark_line(strokeDash=[5, 3])
                .encode(y="Value:Q", color=alt.Color("Series:N", scale=alt.Scale(range=["#ff7f0e"])))
            )
            st.altair_chart(alt.layer(actual_line, fitted_line).properties(height=240), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("K (capacity)", f"{int(fit.carrying_capacity):,}")
            c2.metric("Segments (r)", ", ".join(f"{r:0.3f}" for r in fit.segment_growth_rates))
            c3.metric("R² on ΔS", f"{fit.r2_on_deltas:0.3f}")
            if getattr(fit, "gamma_exog", None) is not None:
                st.caption(f"Exogenous effect (log ad): γ_exog={fit.gamma_exog:0.4f}")

            with st.expander("Model equation and parameters", expanded=False):
                eq = r"\Delta S_t = r_{seg(t)}\, S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) + \gamma_{pulse}\,pulse_t + \gamma_{step}\,step_t"
                if getattr(fit, "gamma_exog", None) is not None:
                    eq += r" + \gamma_{exog}\,x_t"
                st.latex(eq)
                st.markdown("**Fitted parameters**")
                st.markdown(f"- **K (capacity)**: {fit.carrying_capacity:,.0f}")
                st.markdown(
                    "- **Segment growth rates r_j**: "
                    + ", ".join(f"r{j+1}={r:0.3f}" for j, r in enumerate(fit.segment_growth_rates))
                )
                st.markdown(f"- **γ_pulse**: {fit.gamma_pulse:0.4f}")
                st.markdown(f"- **γ_step**: {fit.gamma_step:0.4f}")
                if getattr(fit, "gamma_exog", None) is not None:
                    st.markdown(f"- **γ_exog**: {fit.gamma_exog:0.4f}")

            if horizon_ahead > 0:
                last_val = float(fit.fitted_series.iloc[-1])
                last_r = float(fit.segment_growth_rates[-1]) if fit.segment_growth_rates else 0.0
                fc = forecast_piecewise_logistic(
                    last_value=last_val,
                    months_ahead=horizon_ahead,
                    carrying_capacity=fit.carrying_capacity,
                    segment_growth_rate=last_r,
                    gamma_step_level=fit.gamma_step,
                )
                fc_index = pd.date_range(
                    fit.fitted_series.index[-1] + pd.offsets.MonthEnd(1), periods=horizon_ahead, freq="M"
                )
                fc_df = pd.DataFrame({"Forecast": fc}, index=fc_index)
                merged = pd.concat([overlay_df, fc_df], axis=0)
                chart_fc = (
                    alt.Chart(merged.reset_index().rename(columns={"index": "date"}))
                    .transform_fold(["Actual", "Fitted", "Forecast"], as_=["Series", "Value"])
                    .mark_line()
                    .encode(x=alt.X("date:T"), y="Value:Q", color="Series:N")
                    .properties(height=240)
                )
                st.altair_chart(chart_fc, use_container_width=True)
        except Exception as e:
            st.error(f"Model fit failed: {e}")


def tail_view_ui(
    plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, target_col: Optional[str], breakpoints: list[int]
) -> None:
    st.subheader("Stage 5: Diagnostics (delta view)")
    st.bar_chart(plot_df.diff().fillna(0))

    window_default = int(_get_state("est_window", 6))
    window = st.slider("Estimation window (last N months)", 3, 12, window_default, 1, key="est_window")
    st.caption("This window recomputes trailing medians for the estimates and the tail chart below.")

    st.subheader(f"Last {window} months (tail)")
    tail_df = plot_df.tail(window)

    series_title = "Series (Paid is dashed)" if (use_dual_axis and ("Paid" in tail_df.columns)) else "Series"
    base_chart = plot_series(tail_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title)

    if target_col is not None and breakpoints:
        try:
            full_s = plot_df[target_col].dropna()
            segs = compute_segment_slopes(full_s, breakpoints)
            tail_start, tail_end = tail_df.index[0], tail_df.index[-1]
            segs_t = [seg for seg in segs if (seg.end_date >= tail_start and seg.start_date <= tail_end)]
            fit_rows_t = []
            for seg in segs_t:
                xs = pd.date_range(max(seg.start_date, tail_start), min(seg.end_date, tail_end), freq="M")
                start_val = float(full_s.loc[seg.start_date])
                for i, d in enumerate(xs):
                    fit_rows_t.append({"date": d, "Fit": start_val + seg.slope_per_month * i})
            if fit_rows_t:
                fit_df_t = pd.DataFrame(fit_rows_t)
                fit_t = alt.Chart(fit_df_t).mark_line(color="#7f8c8d").encode(x="date:T", y="Fit:Q")
                base_chart = alt.layer(base_chart, fit_t).resolve_scale(y="independent").properties(height=240)
        except Exception:
            pass

    st.altair_chart(base_chart, use_container_width=True)


def metrics_and_apply_ui(all_series, paid_series, net_only: bool) -> None:
    estimates = _compute_estimates(all_series, paid_series, int(_get_state("est_window", 6)))

    cols = st.columns(3)
    if "start_free" in estimates:
        cols[0].metric("Starting free (latest)", f"{estimates['start_free']:,}")
    if "start_premium" in estimates:
        cols[1].metric("Starting premium (latest)", f"{estimates['start_premium']:,}")
    if "organic_growth" in estimates:
        cols[2].metric("Net free growth (monthly)", f"{estimates['organic_growth']*100:0.2f}%")

    cols2 = st.columns(3)
    if "conv_ongoing" in estimates:
        cols2[0].metric("Ongoing premium conversion (proxy)", f"{estimates['conv_ongoing']*100:0.3f}%")
    if not net_only:
        cols2[1].metric("Free churn", f"{float(estimates.get('churn_free', 0.0))*100:0.2f}%")
        cols2[2].metric("Premium churn", f"{float(estimates.get('churn_prem', 0.0))*100:0.2f}%")
    else:
        cols2[1].metric("Net growth (includes churn)", "—")
        cols2[2].metric(" ", " ")

    st.caption(
        "Notes: From totals alone we can compute net growth and a conversion proxy (when both series present). Churn and CAC need more detail."
    )

    if st.button("Apply estimates to Simulator"):
        for k, v in estimates.items():
            st.session_state[k] = v
        if net_only:
            st.session_state["churn_free"] = 0.0
            st.session_state["churn_prem"] = 0.0
        st.session_state["ad_stage1"] = 0.0
        st.session_state["ad_stage2"] = 0.0
        st.session_state["ad_const"] = 0.0
        st.session_state["spend_mode_index"] = 1
        st.session_state["conv_new"] = 0.0
        st.session_state["horizon_months"] = max(int(_get_state("horizon_months", 60)), 24)
        st.session_state["switch_to_sim"] = True
        st.success("Applied. Switching to Simulator…")
        st.rerun()


def number_input_state(label: str, *, key: str, default_value, **kwargs):
    kwargs["key"] = key
    if key not in st.session_state:
        kwargs["value"] = default_value
    return st.number_input(label, **kwargs)


def slider_state(label: str, *, key: str, default_value, **kwargs):
    kwargs["key"] = key
    if key not in st.session_state:
        kwargs["value"] = default_value
    return st.slider(label, **kwargs)


def sidebar_inputs() -> SimulationInputs:
    st.sidebar.header("Assumptions")

    with st.sidebar.expander("Starting point", expanded=True):
        start_free = number_input_state(
            "Starting free subscribers",
            min_value=0,
            default_value=int(_get_state("start_free", 0)),
            step=10,
            key="start_free",
        )
        start_premium = number_input_state(
            "Starting premium subscribers",
            min_value=0,
            default_value=int(_get_state("start_premium", 0)),
            step=1,
            key="start_premium",
        )

    with st.sidebar.expander("Horizon", expanded=False):
        horizon = slider_state(
            "Months to simulate",
            min_value=12,
            max_value=120,
            default_value=int(_get_state("horizon_months", 60)),
            step=6,
            key="horizon_months",
        )

    with st.sidebar.expander("Growth & churn", expanded=True):
        organic_growth = number_input_state(
            "Organic monthly growth (free)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("organic_growth", 0.01)),
            step=0.001,
            format="%0.3f",
            key="organic_growth",
        )
        churn_free = number_input_state(
            "Monthly churn (free)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("churn_free", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_free",
        )
        churn_prem = number_input_state(
            "Monthly churn (premium)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("churn_prem", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_prem",
        )

    with st.sidebar.expander("Conversions", expanded=True):
        conv_new = number_input_state(
            "New-subscriber premium conversion",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("conv_new", 0.0)),
            step=0.001,
            format="%0.3f",
            key="conv_new",
        )
        conv_ongoing = number_input_state(
            "Ongoing premium conversion of existing free",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("conv_ongoing", 0.0)),
            step=0.0001,
            format="%0.4f",
            key="conv_ongoing",
        )

    with st.sidebar.expander("Acquisition", expanded=True):
        spend_mode = st.selectbox(
            "Ad spend schedule",
            ["Two-stage (Years 1-2 / 3-5)", "Constant"],
            index=int(_get_state("spend_mode_index", 1)),
            key="spend_mode",
        )
        if spend_mode.startswith("Two-stage"):
            stage1 = number_input_state(
                "Monthly ad spend (years 1-2)",
                min_value=0.0,
                default_value=float(_get_state("ad_stage1", 0.0)),
                step=50.0,
                key="ad_stage1",
            )
            stage2 = number_input_state(
                "Monthly ad spend (years 3-5)",
                min_value=0.0,
                default_value=float(_get_state("ad_stage2", 0.0)),
                step=50.0,
                key="ad_stage2",
            )
            ad_schedule = AdSpendSchedule.two_stage(stage1, stage2)
            st.session_state["spend_mode_index"] = 0
        else:
            const_spend = number_input_state(
                "Monthly ad spend (constant)",
                min_value=0.0,
                default_value=float(_get_state("ad_const", 0.0)),
                step=50.0,
                key="ad_const",
            )
            ad_schedule = AdSpendSchedule.constant(const_spend)
            st.session_state["spend_mode_index"] = 1

        cac = number_input_state(
            "Cost per new free subscriber (CAC)",
            min_value=0.01,
            default_value=float(_get_state("cac", 2.0)),
            step=0.1,
            key="cac",
        )
        ad_manager_fee = number_input_state(
            "Ad manager monthly fee",
            min_value=0.0,
            default_value=float(_get_state("ad_manager_fee", 0.0)),
            step=50.0,
            key="ad_manager_fee",
        )

    with st.sidebar.expander("Pricing & fees", expanded=True):
        price_monthly = number_input_state(
            "Premium monthly price (gross)",
            min_value=0.0,
            default_value=float(_get_state("price_monthly", 10.0)),
            step=1.0,
            key="price_monthly",
        )
        price_annual = number_input_state(
            "Premium annual price (gross)",
            min_value=0.0,
            default_value=float(_get_state("price_annual", 70.0)),
            step=5.0,
            key="price_annual",
        )
        substack_pct = number_input_state(
            "Substack fee %",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("substack_pct", 0.10)),
            step=0.01,
            format="%0.2f",
            key="substack_pct",
        )
        stripe_pct = number_input_state(
            "Stripe % (billing + card)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("stripe_pct", 0.036)),
            step=0.001,
            format="%0.3f",
            key="stripe_pct",
        )
        stripe_flat = number_input_state(
            "Stripe flat per transaction",
            min_value=0.0,
            default_value=float(_get_state("stripe_flat", 0.30)),
            step=0.05,
            key="stripe_flat",
        )
        annual_share = slider_state(
            "Share of premium on annual plans",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("annual_share", 0.0)),
            step=0.05,
            key="annual_share",
        )

    return SimulationInputs(
        starting_free_subscribers=int(start_free),
        starting_premium_subscribers=int(start_premium),
        horizon_months=int(horizon),
        organic_monthly_growth_rate=float(organic_growth),
        monthly_churn_rate_free=float(churn_free),
        monthly_churn_rate_premium=float(churn_prem),
        new_subscriber_premium_conv_rate=float(conv_new),
        ongoing_premium_conv_rate=float(conv_ongoing),
        cost_per_new_free_subscriber=float(cac),
        ad_spend_schedule=ad_schedule,
        ad_manager_monthly_fee=float(ad_manager_fee),
        premium_monthly_price_gross=float(price_monthly),
        premium_annual_price_gross=float(price_annual),
        substack_fee_pct=float(substack_pct),
        stripe_fee_pct=float(stripe_pct),
        stripe_flat_fee=float(stripe_flat),
        annual_share=float(annual_share),
    )


def render_kpis(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ending free", f"{int(last.free_subscribers):,}")
    col2.metric("Ending premium", f"{int(last.premium_subscribers):,}")
    col3.metric("Net MRR", format_currency(last.mrr_net))
    col4.metric("Cumulative profit", format_currency(last.cumulative_net_profit))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Cumulative ad spend", format_currency(last.cumulative_ad_spend))
    roas = 0.0 if last.cumulative_ad_spend == 0 else (df.net_revenue.sum() / df.ad_spend.sum())
    col6.metric("ROAS (net revenue / ad spend)", f"{roas:0.2f}x")
    avg_cac = float("nan") if df.new_free_paid.sum() == 0 else df.ad_spend.sum() / df.new_free_paid.sum()
    col7.metric("Blended CAC (paid only)", format_currency(avg_cac))
    payback_month = next((i + 1 for i, c in enumerate(df.cumulative_net_profit) if c > 0), math.nan)
    col8.metric("Payback month (cumulative)", "—" if math.isnan(payback_month) else str(int(payback_month)))


def render_charts(df: pd.DataFrame) -> None:
    st.subheader("Subscribers over time")
    st.line_chart(
        df[["free_subscribers", "premium_subscribers"]].rename(
            columns={
                "free_subscribers": "Free",
                "premium_subscribers": "Premium",
            }
        )
    )

    st.subheader("Revenue and profit")
    st.area_chart(
        df[["mrr_net", "net_revenue", "profit"]].rename(
            columns={
                "mrr_net": "Net MRR",
                "net_revenue": "Net revenue (monthly)",
                "profit": "Profit (monthly)",
            }
        )
    )

    st.subheader("Spend vs revenue")
    st.line_chart(
        df[["ad_spend", "ad_manager_fee", "net_revenue"]].rename(
            columns={
                "ad_spend": "Ad spend",
                "ad_manager_fee": "Ad manager fee",
                "net_revenue": "Net revenue",
            }
        )
    )


def render_estimators() -> None:
    st.subheader("Quick estimators from your Substack stats")

    with st.expander("Organic growth and churn", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            f_start = st.number_input("Free subs at start", min_value=0, value=1000, step=10)
            f_end = st.number_input("Free subs at end", min_value=0, value=1200, step=10)
            months = st.number_input("Months in period", min_value=1, value=3, step=1)
        with c2:
            paid_new_total = st.number_input("New free from ads in period", min_value=0, value=200, step=10)
            free_churn_total = st.number_input(
                "Free churn (unsubs + cleaning) in period", min_value=0, value=50, step=10
            )

        avg_free = max((f_start + f_end) / 2.0, 1.0)
        organic_new_total_no_churn = (f_end - f_start) - paid_new_total
        organic_rate_no_churn = organic_new_total_no_churn / max(months * avg_free, 1.0)

        organic_new_total_with_churn = (f_end - f_start) - paid_new_total + free_churn_total
        organic_rate_with_churn = organic_new_total_with_churn / max(months * avg_free, 1.0)

        churn_rate_est = free_churn_total / max(months * avg_free, 1.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Organic monthly growth (w/ churn)", f"{organic_rate_with_churn*100:0.2f}%")
        m2.metric("Organic monthly growth (simple)", f"{organic_rate_no_churn*100:0.2f}%")
        m3.metric("Monthly churn (free)", f"{churn_rate_est*100:0.2f}%")

        st.caption(
            "Tip: Use Subscribers over time + exports. Count paid-attributed signups to estimate 'new free from ads'."
        )

    with st.expander("Event evaluation (pre/post trend)", expanded=False):
        st.caption("Add an event date to compare the slope before and after for Total or Free.")
        target = st.selectbox("Series", ["Total", "Free", "Paid"], index=0)
        date_str = st.text_input("Event date (YYYY-MM-DD)", value="")
        if st.session_state.get("sim_df") is not None:
            # Use imported series when available
            series_map = {}
            with suppress(Exception):
                series_map.update({"Total": st.session_state.get("import_total")})
                series_map.update({"Paid": st.session_state.get("import_paid")})
                if series_map.get("Total") is not None and series_map.get("Paid") is not None:
                    series_map["Free"] = series_map["Total"] - series_map["Paid"]
            if (s := series_map.get(target)) is not None and date_str:
                try:
                    dt = pd.to_datetime(date_str)
                    pre, post = slope_around(s, dt, window=6)
                    st.metric("Pre slope (per month)", f"{pre:0.2f}")
                    st.metric("Post slope (per month)", f"{post:0.2f}")
                except Exception:
                    st.info("Provide a valid date inside your imported series range.")

    with st.expander("Event ROI (rough)", expanded=False):
        st.caption("For Ad spend events with a cost, compare pre/post slope and estimate incremental subs.")
        ev = st.session_state.get("events_df")
        total_series = st.session_state.get("import_total")
        if ev is not None and total_series is not None:
            ev2 = ev.dropna(subset=["date"]) if not ev.empty else ev
            if ev2 is not None and not ev2.empty:
                ev2 = ev2.copy()
                ev2["date"] = pd.to_datetime(ev2["date"]).dt.to_period("M").dt.to_timestamp("M")
                rows = []
                for _, r in ev2.iterrows():
                    d = r["date"]
                    pre, post = slope_around(total_series, d, window=6)
                    delta = post - pre
                    cost = float(r.get("cost", 0.0) or 0.0)
                    rows.append({"date": d, "type": r.get("type", ""), "slope_delta": delta, "cost": cost})
                if rows:
                    out = pd.DataFrame(rows)
                    st.dataframe(out, use_container_width=True)

    with st.expander("Acquisition cost (CAC)", expanded=True):
        spend = st.number_input("Ad spend in period", min_value=0.0, value=3000.0, step=50.0)
        paid_new = st.number_input("New free subscribers from ads in period", min_value=0, value=150, step=10)
        cac = float("nan") if paid_new == 0 else spend / paid_new
        st.metric("CAC (cost per new free subscriber)", format_currency(cac if cac == cac else 0.0))
        st.caption("Tip: From ad manager or Substack 'Where subscribers came from' tagged as paid.")

    with st.expander("Premium conversions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            new_prem_from_new = st.number_input(
                "New premium from brand-new free (first month)", min_value=0, value=10, step=1
            )
            new_free_this_month = st.number_input("New free this month (all sources)", min_value=0, value=500, step=10)
            conv_new = 0.0 if new_free_this_month == 0 else new_prem_from_new / new_free_this_month
            st.metric("New-subscriber premium conversion", f"{conv_new*100:0.2f}%")
        with c2:
            upgrades_existing = st.number_input(
                "Premium upgrades from existing free in period", min_value=0, value=15, step=1
            )
            avg_free_base = st.number_input("Average free base in period", min_value=1, value=1200, step=10)
            months2 = st.number_input("Months measured", min_value=1, value=3, step=1, key="months2")
            conv_ongoing = upgrades_existing / max(months2 * avg_free_base, 1.0)
            st.metric("Ongoing premium conversion (monthly)", f"{conv_ongoing*100:0.3f}%")

    with st.expander("Net revenue per premium (sanity check)", expanded=False):
        price_m = st.number_input("Monthly gross price", min_value=0.0, value=10.0, step=1.0)
        sub_pct = st.number_input("Substack %", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%0.2f")
        stripe_pct = st.number_input("Stripe %", min_value=0.0, max_value=1.0, value=0.036, step=0.001, format="%0.3f")
        stripe_flat = st.number_input("Stripe flat", min_value=0.0, value=0.30, step=0.05)
        net = price_m * (1 - sub_pct - stripe_pct) - stripe_flat
        st.metric("Net monthly revenue per premium", format_currency(max(net, 0)))


def render_help() -> None:
    st.subheader("How to map Substack stats to this simulator")
    st.markdown(
        """
1) Organic monthly growth (free)
- Use Subscribers over time (or export). Compute organic new = total new free − new free from ads.
- Monthly organic rate ≈ organic new ÷ (months × average free base).

2) Monthly churn (free and premium)
- Use unsubscribes + list cleaning totals. Monthly churn ≈ churned ÷ (months × average cohort size).

3) CAC (cost per new free)
- CAC = ad spend ÷ new free from ads in the same period. Use your ad manager or Substack source tags.

4) New-subscriber premium conversion
- In a recent month, new premium from brand‑new free signups ÷ number of new free that month.

5) Ongoing premium conversion of existing free
- Premium upgrades not tied to first‑month signups ÷ (months × average free).

6) Pricing and fees
- Defaults: Substack 10%, Stripe 3.6% + $0.30. Adjust to your setup.

7) Ad spend schedule
- Two-stage: months 0–23 vs 24–59. Constant: one number.

Use the Estimators tab to compute these, then plug them into the Simulator sidebar.
        """
    )


def render_save_load() -> None:
    st.subheader("Save / Load session")
    st.caption("Download a portable bundle to save your work, or upload to restore it later.")

    c1, c2 = st.columns(2)
    with c1:
        has_fit = st.session_state.get("pwlog_fit") is not None
        include_fit = st.checkbox("Include model fit", value=bool(has_fit))
        include_sim = st.checkbox("Include simulation results", value=False)
        bundle = collect_session_bundle(include_fit, include_sim)
        st.download_button(
            "Download session bundle (.zip)",
            data=bundle,
            file_name="substack_session.zip",
            mime="application/zip",
        )
    with c2:
        uploaded = st.file_uploader("Upload session bundle (.zip)", type=["zip"], key="session_bundle")
        if uploaded is not None:
            try:
                apply_session_bundle(uploaded)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")


def _compute_estimates(
    all_series: Optional[pd.Series], paid_series: Optional[pd.Series], window_months: int = 6
) -> dict:
    return compute_estimates(all_series, paid_series, window_months)


def render_data_import() -> None:
    """
    Stage 1–5: Import, preview, annotate, feature-build, quick-fit, diagnostics, and handoff.

    Notes:
    - Uses st.session_state keys:
      import_total, import_paid, observations_df, events_df, covariates_df, features_df,
      adds_df, churn_df, pwlog_fit, switch_to_sim, est_window, horizon_months, ...
    """

    # ---------- Small internal utilities ----------

    def _upload_panel(
        title: str,
        help_hint: str,
        key_prefix: str,
        default_header: bool = False,
    ) -> tuple[Optional[Any], bool, Optional[int], Optional[int]]:
        return upload_panel(title, help_hint, key_prefix, default_header)

    def _plot_series(plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, series_title: str) -> alt.Chart:
        return plot_series(plot_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title)

    # ---------- UI: Header + quick save/load ----------
    st.subheader("Stage 1: Import Substack exports (time series)")
    st.caption(
        "Upload two files: All subscribers over time, and Paid subscribers over time."
        " No headers by default: first column is date, second is count."
    )

    with st.expander("Save / Load (quick access)", expanded=False):
        has_fit_i = st.session_state.get("pwlog_fit") is not None
        include_fit_i = st.checkbox("Include model fit", value=bool(has_fit_i), key="import_include_fit")
        include_sim_i = st.checkbox("Include simulation results", value=False, key="import_include_sim")
        bundle_i = collect_session_bundle(include_fit_i, include_sim_i)
        st.download_button(
            "Export my config (.zip)",
            data=bundle_i,
            file_name="substack_session.zip",
            mime="application/zip",
            key="import_export_btn",
        )
        uploaded_i = st.file_uploader("Restore session bundle (.zip)", type=["zip"], key="import_session_bundle")
        if uploaded_i is not None:
            try:
                apply_session_bundle(uploaded_i)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")

    # ---------- Upload inputs (All / Paid) ----------
    c_all, c_paid = st.columns(2)
    with c_all:
        all_file, all_has_header, all_date_sel, all_count_sel = _upload_panel(
            "All subscribers file (CSV/XLSX, often downloaded as `[blogname]_emails_[date].csv`)",
            help_hint="Pick the time series of all subscribers over time.",
            key_prefix="all",
            default_header=False,
        )
    with c_paid:
        paid_file, paid_has_header, paid_date_sel, paid_count_sel = _upload_panel(
            "Paid subscribers file (CSV/XLSX, often downloaded as `[blogname]_subscribers_[date].csv`)",
            help_hint="Pick the time series of paid subscribers over time.",
            key_prefix="paid",
            default_header=False,
        )

    # ---------- Main flow only if at least one file present ----------
    if all_file is not None or paid_file is not None:
        net_only = st.checkbox("Use net-only growth (set churn to 0)", value=True)
        debug_mode = st.checkbox("Enable debug logging", value=True, help="Adds console logs and inline diagnostics.")

        try:
            all_series = (
                read_series(all_file, all_has_header, all_date_sel, all_count_sel) if all_file is not None else None
            )
            paid_series = (
                read_series(paid_file, paid_has_header, paid_date_sel, paid_count_sel)
                if paid_file is not None
                else None
            )

            plot_df = pd.DataFrame()
            if all_series is not None and not all_series.empty:
                plot_df["Total"] = all_series
                st.session_state["import_total"] = all_series
            if paid_series is not None and not paid_series.empty:
                plot_df["Paid"] = paid_series
                st.session_state["import_paid"] = paid_series
            if debug_mode:
                st.caption(
                    f"Debug: all_series={'None' if all_series is None else len(all_series)}; "
                    f"paid_series={'None' if paid_series is None else len(paid_series)}; "
                    f"plot_df_cols={list(plot_df.columns)}"
                )

            if not plot_df.empty:
                if "Total" in plot_df.columns and "Paid" in plot_df.columns:
                    plot_df["Free"] = plot_df["Total"].astype(float) - plot_df["Paid"].astype(float)

                st.subheader("Imported series")
                st.caption("Mode: Paid and unpaid" if "Paid" in plot_df.columns else "Mode: Unpaid only")

                # Stage 1 output: observations_df
                with suppress(Exception):
                    emit_observations(plot_df)

                # Chart controls
                use_dual_axis = st.checkbox(
                    "Use separate right axis for Paid",
                    value=True,
                    help="Plots Total/Free on left axis and Paid on right axis for readability.",
                )
                default_show_total = "Paid" not in plot_df.columns
                show_total = st.checkbox("Show Total line", value=default_show_total)

                series_title = (
                    "Series (Paid is dashed)" if (use_dual_axis and ("Paid" in plot_df.columns)) else "Series"
                )
                chart = _plot_series(
                    plot_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title
                )
                st.altair_chart(chart, use_container_width=True)

                # Stage 2: Events editor (moved below the graph per request)
                events_editor()

                # Trend detection on Total if present else Free
                target_col = "Total" if "Total" in plot_df.columns else ("Free" if "Free" in plot_df.columns else None)
                breakpoints = trend_detection_ui(plot_df, target_col)

                events_features_ui(plot_df)

                # Stage 3: Adds & Churn
                adds_and_churn_ui(plot_df)

                # Stage 4: Quick Fit
                quick_fit_ui(plot_df, breakpoints)

                # Stage 5: Diagnostics + Tail view
                tail_view_ui(
                    plot_df,
                    use_dual_axis=use_dual_axis,
                    show_total=show_total,
                    target_col=target_col,
                    breakpoints=breakpoints,
                )

            # Summary metrics + apply to simulator
            metrics_and_apply_ui(all_series, paid_series, net_only=net_only)

        except Exception as e:
            st.error(f"Estimation failed: {e}")


def render_outputs_formulas() -> None:
    st.subheader("Stage 8: Outputs and formulas")


render_brand_header()

# Tabs
with st.container():
    tab_import, tab_sim, tab_est, tab_out, tab_save, tab_stages, tab_help = st.tabs(
        [
            "Data Import",
            "Simulator",
            "Estimators",
            "Outputs & Formulas",
            "Save / Load",
            "Stages",
            "Help",
        ]
    )

with tab_import:
    render_data_import()

with tab_sim:
    inputs = sidebar_inputs()
    result = simulate_growth(inputs)
    sim_df = result.monthly
    st.session_state["sim_df"] = sim_df
    st.subheader("Stage 7: Cohort & Finance Simulator")
    with st.expander("Save / Load (quick access)", expanded=False):
        has_fit_q = st.session_state.get("pwlog_fit") is not None
        include_fit_q = st.checkbox("Include model fit", value=bool(has_fit_q), key="quick_include_fit")
        include_sim_q = st.checkbox("Include simulation results", value=False, key="quick_include_sim")
        bundle_q = collect_session_bundle(include_fit_q, include_sim_q)
        st.download_button(
            "Export my config (.zip)",
            data=bundle_q,
            file_name="substack_session.zip",
            mime="application/zip",
            key="quick_export_btn",
        )
        uploaded_q = st.file_uploader("Restore session bundle (.zip)", type=["zip"], key="quick_session_bundle")
        if uploaded_q is not None:
            try:
                apply_session_bundle(uploaded_q)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")
    render_kpis(sim_df)
    with st.expander("Monthly details", expanded=False):
        st.dataframe(sim_df, width="stretch")
    render_charts(sim_df)
    st.caption(
        "MVP model: instant conversion of a share of new free subs, small ongoing conversion of existing free base, "
        "and simple net revenue after Substack + Stripe fees."
    )

with tab_est:
    render_estimators()

with tab_out:
    render_outputs_formulas()

with tab_save:
    render_save_load()

with tab_stages:
    st.subheader("Modeling Stages")
    st.markdown(
        """
1. **File Upload & Parsing**: Upload All/Paid exports. Outputs `observations_df` (daily preferred) with imputation flags; quick charts.
2. **Events & Covariates**: Add shoutouts/ads; adstock + diminishing returns; seasonality features. Outputs `events_df`, `covariates_df`, `features_df`.
3. **Adds/Churn Prep**: From logs (preferred) or totals-only via smoothing; tenure-based churn hazard. Outputs `adds_df`, `churn_df`.
4. **Model Fitting (Pro)**: Bayesian state-space with saturation, events, and uncertainty. Outputs `bayes_fit`, fit plots.
5. **Diagnostics**: Rolling-origin CV, PPC, PSIS-LOO; attribution and elasticity checks. Outputs `validation_report`.
6. **Forecasting & Scenarios**: Posterior simulation under levers (events, ads, cadence, market). Outputs `forecast_df`.
7. **Cohort & Finance Simulator**: Monthly KPIs, ROAS, CAC, payback. Outputs `sim_df`.
8. **Outputs & Docs**: Formulas and downloadable artifacts.

Status: Stages 1–3 (import, events, quick estimators) and Quick Fit are implemented. Pro Fit (Bayesian), diagnostics, and scenario planner are in progress.
        """
    )

with tab_help:
    render_help()

# If requested, auto-switch to the Simulator tab by simulating a click
if st.session_state.get("switch_to_sim"):
    components.html(
        """
        <script>
        const tabs = parent.document.querySelectorAll('button[role="tab"]');
        for (const t of tabs) {
            if (t.innerText.trim() === 'Simulator') { t.click(); break; }
        }
        </script>
        """,
        height=0,
    )
    st.session_state["switch_to_sim"] = False
