import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.logger import get_logger

from substack_analyzer.analysis import (
    build_events_features,
    compute_estimates,
    derive_adds_churn,
    plot_series,
    read_series,
)
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params, forecast_piecewise_logistic
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

# MUST be the first Streamlit call:
st.set_page_config(
    page_title="Substack Ads ROI Simulator",
    layout="wide",
    page_icon=str(LOGO_ICON) if LOGO_ICON.exists() else (str(LOGO_FULL) if LOGO_FULL.exists() else None),
)

# Streamlit logger (appears in deployment logs)
logger = get_logger(__name__)
logger.info("App startup: Streamlit logger initialized (hello from logger)")


# --- Events table: single source of truth ---
EVENTS_COLUMNS = ["date", "type", "persistence", "notes", "cost"]
TYPE_TO_PERSISTENCE = {
    "ad spend": "Transient",
    "ad": "Transient",
    "shout-out": "Transient",
    "viral post": "Transient",
    "launch": "Persistent",
    "paywall change": "Persistent",
    "change": "Transient",
    "other": None,
}
EVENT_TYPE_OPTIONS = [
    "Ad spend",
    "Shout-out",
    "Viral post",
    "Launch",
    "Paywall change",
    "Other",
]
if "events_df" not in st.session_state:
    st.session_state["events_df"] = pd.DataFrame(columns=EVENTS_COLUMNS)


def _clean_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types without changing the date *month/day* a user entered."""
    logger.info("_clean_events_df has been called")
    logger.info(f"df: {df}")
    df = df.copy()
    for col in EVENTS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    logger.info(f"df after adding missing columns: {df}")
    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    # Fill persistence only where missing
    typed_lower = df.get("type").astype(str).str.lower()
    need_fill = df.get("persistence").isna() | (df.get("persistence").astype(str).str.len() == 0)
    df.loc[need_fill, "persistence"] = typed_lower.map(TYPE_TO_PERSISTENCE)
    logger.info(f"df at the end of _clean_events_df: {df}")
    return df


def _event_rules_from_events() -> Optional[alt.Chart]:
    logger.info("_event_rules_from_events has been called")
    ev = st.session_state.get("events_df")
    logger.info(f"ev: {ev}")
    if not isinstance(ev, pd.DataFrame) or ev.empty or "date" not in ev.columns:
        return None
    ev2 = ev.copy()
    ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce")
    ev2 = ev2.dropna(subset=["date"])
    # Normalize Effect labels for reliable styling
    eff_map = {"persistent": "Persistent", "transient": "Transient", "no effect": "No effect"}
    with suppress(Exception):
        ev2["effect_norm"] = ev2.get("persistence").astype(str).str.strip().str.lower().map(eff_map).fillna("Transient")

    layers = []
    # Persistent: green solid
    ev_p = ev2[ev2.get("effect_norm") == "Persistent"]
    if not ev_p.empty:
        logger.info("Adding persistent event rules")
        layers.append(
            alt.Chart(ev_p)
            .mark_rule(strokeWidth=2, color="#27ae60")
            .encode(
                x=alt.X("date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )
    # Transient: purple dashed
    ev_t = ev2[ev2.get("effect_norm") == "Transient"]
    if not ev_t.empty:
        logger.info("Adding transient event rules")
        layers.append(
            alt.Chart(ev_t)
            .mark_rule(strokeWidth=2, color="#8e44ad", strokeDash=[6, 4])
            .encode(
                x=alt.X("date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )
    # No effect: grey dotted
    ev_n = ev2[ev2.get("effect_norm") == "No effect"]
    if not ev_n.empty:
        logger.info("Adding no effect event rules")
        layers.append(
            alt.Chart(ev_n)
            .mark_rule(strokeWidth=2, color="#bdc3c7", strokeDash=[2, 4])
            .encode(
                x=alt.X("date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )

    if not layers:
        return None
    return alt.layer(*layers)


def _on_events_editor_change():
    logger.info("_on_events_editor_change has been called")
    grid_dict = st.session_state.get("events_editor") or {}
    logger.info(f"grid_dict: {grid_dict}")
    try:
        rows_or_cols = grid_dict.get("data", grid_dict) if isinstance(grid_dict, dict) else grid_dict
        df = pd.DataFrame(rows_or_cols)
        st.session_state["events_df"] = _clean_events_df(df)
        logger.info(f"st.session_state['events_df']: {st.session_state['events_df']}")
        _set_markers_from_events()
    except Exception:
        pass


def _events_change_dates() -> list[pd.Timestamp]:
    ev = st.session_state.get("events_df")
    if not isinstance(ev, pd.DataFrame) or ev.empty or "date" not in ev.columns:
        return []
    ev2 = ev.copy()
    ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce")
    # Determine which event types count as breakpoints
    mode = str(st.session_state.get("breakpoint_mode", "all")).lower()
    types_series = ev2.get("type").astype(str).str.lower()
    effect = ev2.get("persistence").astype(str).str.lower()
    if mode == "all":
        mask = ev2["date"].notna() & ~effect.eq("no effect")
    elif mode == "selected":
        sel = st.session_state.get("breakpoint_types", [])
        sel_l = {str(t).lower() for t in (sel or [])}
        mask = types_series.isin(sel_l) & ~effect.eq("no effect")
    else:
        # throw error
        raise ValueError("Invalid breakpoint mode")
    return [pd.Timestamp(d) for d in ev2.loc[mask, "date"].dropna().tolist()]


def _set_markers_from_events() -> None:
    """Make the chart markers come from events (and keep them there)."""
    st.session_state["markers_source"] = "events"
    st.session_state["detected_change_dates"] = _events_change_dates()


def _normalize_month_end(dates: list[Any]) -> list[pd.Timestamp]:
    """Coerce arbitrary date-like values to month-end pd.Timestamp and de-duplicate/sort."""
    if dates is None:
        return []
    try:
        s = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        s = s.dropna().dt.to_period("M").dt.to_timestamp("M")
        vals = [pd.Timestamp(d) for d in pd.unique(s.sort_values())]
        return vals
    except Exception:
        return []


def _dates_to_breakpoint_indices(dates: list[pd.Timestamp], index: pd.DatetimeIndex) -> list[int]:
    """Map month-end dates to integer indices into the given monthly index."""
    if dates is None or len(dates) == 0:
        return []
    idxs: list[int] = []
    for d in _normalize_month_end(dates):
        try:
            if d in index:
                idxs.append(int(index.get_loc(d)))
        except Exception:
            continue
    # unique + sorted + valid interior indices only
    return sorted(set(i for i in idxs if i is not None))


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
    st.sidebar.image(str(LOGO_FULL if LOGO_FULL.exists() else LOGO_ICON), width="stretch")


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
    """Shared UI for file upload + optional header + column choices.

    Returns
    -------
    (file_obj, has_header, date_sel, count_sel)
        - file_obj: The uploaded file-like object, or None if no file selected or preview failed.
        - has_header: Whether the uploaded file is expected to contain a header row.
        - date_sel: The zero-based index of the date column derived from a small preview, or None.
        - count_sel: The zero-based index of the count column derived from a small preview, or None.

    Notes
    -----
    - The two selectboxes (date/count column indices) are only shown when a file is present and a
      preview can be read. If reading the preview fails, an error is shown and file_obj is set to None.
    - When no file is provided, the returned indices may be defaults and should be treated as optional.
    """
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
        st.dataframe(obs.reset_index(), width="stretch")
        st.download_button(
            "Download observations.csv",
            data=obs.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="observations.csv",
            mime="text/csv",
        )


def trend_detection_ui(plot_df: pd.DataFrame, target_col: Optional[str]) -> list[int]:
    st.caption("Detection runs only when you click the Events button above.")
    if target_col is None:
        return []
    # Let users choose detection sensitivity ahead of time; used when button is clicked.
    st.slider("Max changes to detect", 0, 8, 3, 1, key="max_changes_detect")

    # Show any previously detected results (if the user clicked the button).
    bkps = list(st.session_state.get("detected_breakpoints", []))
    if bkps:
        s_idx = plot_df[target_col].dropna().index
        dates = [pd.to_datetime(s_idx[i]) for i in bkps if i < len(s_idx)]
        st.markdown(f"**Detected change dates (on {target_col}):**")
        st.markdown(_format_date_badges(dates), unsafe_allow_html=True)
    return bkps


def events_editor(plot_df: pd.DataFrame, target_col: Optional[str]) -> None:
    st.subheader("Stage 2: Events & annotations")
    st.caption("Track shout-outs, ad campaigns, launches, etc. Dates must match the series timeline.")

    # Add detected change dates
    with st.container():
        add_col1, _ = st.columns([1, 3])
        with add_col1:
            if st.button("Detect change dates"):
                if target_col is None:
                    st.info("No target series selected for detection.")
                else:
                    try:
                        max_bkps = int(st.session_state.get("max_changes_detect", 3))
                        bkps = detect_change_points(plot_df[target_col], max_changes=max_bkps)
                    except Exception:
                        bkps = []

                    if not bkps:
                        st.info("No change dates detected with current settings.")
                    else:
                        s_idx = plot_df[target_col].dropna().index
                        change_dates_for_events = [s_idx[i - 1] if i > 0 else s_idx[i] for i in bkps if i < len(s_idx)]
                        st.session_state["detected_breakpoints"] = list(bkps)
                        st.session_state["detected_change_dates"] = [pd.to_datetime(d) for d in change_dates_for_events]
                        st.session_state["detected_target_col"] = target_col

                        seeded = pd.DataFrame(
                            {
                                "date": [pd.to_datetime(d).date() for d in change_dates_for_events],
                                "type": ["Other"] * len(change_dates_for_events),
                                "persistence": ["Persistent"] * len(change_dates_for_events),
                                "notes": [f"Automatically detected change in {target_col}"]
                                * len(change_dates_for_events),
                                "cost": [0.0] * len(change_dates_for_events),
                            }
                        )
                        base = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
                        merged = pd.concat([base, seeded], ignore_index=True) if not base.empty else seeded
                        # De-dupe so multiple clicks don't clobber later edits
                        merged = merged.drop_duplicates(subset=["date", "type", "notes"], keep="first")
                        st.session_state["events_df"] = _clean_events_df(merged)
                        _set_markers_from_events()
                        st.rerun()

    # The editable grid. Do *not* overwrite events_df here unless the grid actually changed.
    st.data_editor(
        st.session_state["events_df"],
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "type": st.column_config.SelectboxColumn("Type", options=EVENT_TYPE_OPTIONS, width="medium"),
            "persistence": st.column_config.SelectboxColumn(
                "Effect", options=["No effect", "Transient", "Persistent"], width="medium"
            ),
            "notes": st.column_config.TextColumn("Notes", width="large"),
            "cost": st.column_config.NumberColumn(
                "Cost ($)", step=10.0, min_value=0.0, format="%.2f", help="For Ad spend ROI calc"
            ),
        },
        width="stretch",
        key="events_editor",
        on_change=_on_events_editor_change,
    )

    # Quick-add (do NOT force month-end here)
    with st.expander("Quick add event", expanded=False):
        with st.form("quick_add_event_form", clear_on_submit=True):
            qa_date = st.date_input("Date (YYYY-MM-DD)")
            qa_type = st.selectbox("Type", EVENT_TYPE_OPTIONS)
            qa_persist = st.selectbox("Effect", ["No effect", "Transient", "Persistent"], index=1)
            qa_cost = st.number_input("Cost ($)", min_value=0.0, step=10.0, value=0.0, format="%.2f")
            qa_notes = st.text_input("Notes", value="")
            submitted = st.form_submit_button("Add to Events")

        if submitted and qa_date is not None:
            new_row = {
                "date": pd.to_datetime(qa_date).date(),  # keep the user's exact day
                "type": qa_type,
                "persistence": qa_persist,
                "notes": qa_notes,
                "cost": float(qa_cost or 0.0),
            }
            base = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
            merged = (
                pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)
                if not base.empty
                else pd.DataFrame([new_row])
            )
            st.session_state["events_df"] = _clean_events_df(merged)
            _set_markers_from_events()
            st.rerun()


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

        # --- Protect the user-edited events from accidental in-place mutation downstream ---
        _ev_backup = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
        st.session_state["events_df"] = _ev_backup.copy(deep=True)
        try:
            covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
        finally:
            # Always restore the user-owned table, even if the builder throws
            st.session_state["events_df"] = _ev_backup
        # ------------------------------------------------------------------------------

        st.session_state["adstock_lambda"] = float(lam)
        st.session_state["ad_log_theta"] = float(theta)
        st.session_state["covariates_df"] = covariates_df
        st.session_state["features_df"] = features_df
        st.markdown("**Outputs**: `events_df` (above), `covariates_df`, `features_df`.")
        st.dataframe(features_df.reset_index(), width="stretch")
        # download buttons unchanged...


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
            st.dataframe(adds_df.reset_index(), width="stretch")
            st.dataframe(churn_df.reset_index(), width="stretch")
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
    """Stage 4: fit piecewise logistic, wire in exogenous feature, and draw overlay/forecast."""
    st.subheader("Stage 4: Fit model")
    st.caption(
        "Fits on Total (preferred) or Free if Total is unavailable. "
        "Uses detected change points as segments. If built, uses the ad-response exogenous feature."
    )

    # Controls
    use_exog = False
    features_df = st.session_state.get("features_df")
    if isinstance(features_df, pd.DataFrame) and "ad_effect_log" in features_df.columns:
        use_exog = st.checkbox("Use ad-response feature in fit (γ_exog·ad_effect_log)", value=True)

    horizon_ahead = st.slider("Forecast months ahead", 0, 36, 12, 1)

    if st.button("Fit model and overlay"):
        try:
            # ----- choose series to fit -----
            fit_series_source = plot_df.get("Total") if "Total" in plot_df.columns else plot_df.get("Free")
            if fit_series_source is None or fit_series_source.empty:
                st.info("Need Total or Free series to fit.")
                return

            # ----- optional exogenous regressor -----
            extra_exog = None
            if use_exog and isinstance(features_df, pd.DataFrame):
                extra_exog = features_df["ad_effect_log"].astype(float)

            # ----- fit -----
            fit = fit_piecewise_logistic(
                total_series=fit_series_source,
                breakpoints=breakpoints,
                events_df=st.session_state.get("events_df"),
                extra_exog=extra_exog,
            )
            st.session_state["pwlog_fit"] = fit

            # ----- initialize sidebar override defaults if absent -----
            if "modelfit_K" not in st.session_state:
                st.session_state["modelfit_K"] = float(getattr(fit, "carrying_capacity", 0.0))
            if "modelfit_gamma_pulse" not in st.session_state:
                st.session_state["modelfit_gamma_pulse"] = float(getattr(fit, "gamma_pulse", 0.0))
            if "modelfit_gamma_step" not in st.session_state:
                st.session_state["modelfit_gamma_step"] = float(getattr(fit, "gamma_step", 0.0))
            if getattr(fit, "gamma_exog", None) is not None and "modelfit_gamma_exog" not in st.session_state:
                st.session_state["modelfit_gamma_exog"] = float(getattr(fit, "gamma_exog", 0.0))
            if "modelfit_r" not in st.session_state:
                st.session_state["modelfit_r"] = list(getattr(fit, "segment_growth_rates", []) or [])

            # ----- read current overrides & recompute fitted line with them -----
            def _current_fit_params():
                fit_obj = st.session_state.get("pwlog_fit")
                k_val = float(st.session_state.get("modelfit_K", getattr(fit_obj, "carrying_capacity", 0.0) or 0.0))
                r_list = list(
                    st.session_state.get("modelfit_r", list(getattr(fit_obj, "segment_growth_rates", []) or []))
                )
                gp_val = float(
                    st.session_state.get("modelfit_gamma_pulse", getattr(fit_obj, "gamma_pulse", 0.0) or 0.0)
                )
                gs_val = float(st.session_state.get("modelfit_gamma_step", getattr(fit_obj, "gamma_step", 0.0) or 0.0))
                gx_val = st.session_state.get("modelfit_gamma_exog", getattr(fit_obj, "gamma_exog", None))
                return k_val, r_list, gp_val, gs_val, gx_val

            K_now, r_list_now, gp_now, gs_now, gx_now = _current_fit_params()

            fitted_from_overrides = fitted_series_from_params(
                total_series=fit_series_source,
                breakpoints=breakpoints,
                carrying_capacity=float(K_now),
                segment_growth_rates=r_list_now,
                events_df=st.session_state.get("events_df"),
                extra_exog=(extra_exog if gx_now is not None else None),
                gamma_pulse=float(gp_now),
                gamma_step=float(gs_now),
                gamma_exog=(float(gx_now) if gx_now is not None else None),
            )

            # ----- overlay chart: Actual vs Fitted (overrides) -----
            overlay_df = pd.DataFrame(
                {"Actual": fit_series_source, "Fitted": fitted_from_overrides.reindex(fit_series_source.index)}
            )
            base_overlay = alt.Chart(overlay_df.reset_index().rename(columns={"index": "date"})).encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y"))
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

            # ----- metrics -----
            c1, c2, c3 = st.columns(3)
            c1.metric("K (capacity)", f"{int(K_now):,}")
            c2.metric("Segments (r)", ", ".join(f"{r:0.3f}" for r in r_list_now))
            c3.metric("R² on ΔS", f"{fit.r2_on_deltas:0.3f}")
            if getattr(fit, "gamma_exog", None) is not None:
                st.caption(f"Exogenous effect: γ_exog={float(gx_now):0.4f}")

            # ----- latex equation & stash for simulator tab -----
            eq = (
                r"\Delta S_t = r_{seg(t)}\, S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) "
                r"+ \gamma_{pulse}\,pulse_t + \gamma_{step}\,step_t"
            )
            if getattr(fit, "gamma_exog", None) is not None:
                eq += r" + \gamma_{exog}\,x_t"
            st.session_state["growth_equation_latex"] = eq
            with st.expander("Model equation and parameters", expanded=False):
                st.latex(eq)
                st.markdown("**Fitted parameters**")
                st.markdown(f"- **K (capacity)**: {float(K_now):,.0f}")
                st.markdown(
                    "- **Segment growth rates r_j**: " + ", ".join(f"r{j+1}={r:0.3f}" for j, r in enumerate(r_list_now))
                )
                st.markdown(f"- **γ_pulse**: {gp_now:0.4f}")
                st.markdown(f"- **γ_step**: {gs_now:0.4f}")
                if gx_now is not None:
                    st.markdown(f"- **γ_exog**: {float(gx_now):0.4f}")

            # ----- optional forecast ahead -----
            if horizon_ahead > 0:
                last_val = float(fitted_from_overrides.iloc[-1])
                last_r = float(r_list_now[-1]) if r_list_now else 0.0
                fc = forecast_piecewise_logistic(
                    last_value=last_val,
                    months_ahead=horizon_ahead,
                    carrying_capacity=float(K_now),
                    segment_growth_rate=float(last_r),
                    gamma_step_level=float(gs_now),
                )
                fc_index = pd.date_range(
                    fitted_from_overrides.index[-1] + pd.offsets.MonthEnd(1),
                    periods=horizon_ahead,
                    freq="ME",
                )
                fc_df = pd.DataFrame({"Forecast": fc}, index=fc_index)
                merged = pd.concat([overlay_df, fc_df], axis=0)
                chart_fc = (
                    alt.Chart(merged.reset_index().rename(columns={"index": "date"}))
                    .transform_fold(["Actual", "Fitted", "Forecast"], as_=["Series", "Value"])
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y")),
                        y="Value:Q",
                        color="Series:N",
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart_fc, use_container_width=True)

        except Exception as e:
            st.error(f"Model fit failed: {e}")


def _current_fit_params():
    """Return current model parameters, preferring sidebar overrides when present.

    Returns (K, r_list, gamma_pulse, gamma_step, gamma_exog).
    """
    fit_obj = st.session_state.get("pwlog_fit")
    k_val = float(st.session_state.get("modelfit_K", getattr(fit_obj, "carrying_capacity", 0.0) or 0.0))
    r_list = list(st.session_state.get("modelfit_r", list(getattr(fit_obj, "segment_growth_rates", []) or [])))
    gp_val = float(st.session_state.get("modelfit_gamma_pulse", getattr(fit_obj, "gamma_pulse", 0.0) or 0.0))
    gs_val = float(st.session_state.get("modelfit_gamma_step", getattr(fit_obj, "gamma_step", 0.0) or 0.0))
    gx_val = st.session_state.get("modelfit_gamma_exog", getattr(fit_obj, "gamma_exog", None))
    return k_val, r_list, gp_val, gs_val, gx_val


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
                xs = pd.date_range(max(seg.start_date, tail_start), min(seg.end_date, tail_end), freq="ME")
                start_val = float(full_s.loc[seg.start_date])
                for i, d in enumerate(xs):
                    fit_rows_t.append({"date": d, "Fit": start_val + seg.slope_per_month * i})
            if fit_rows_t:
                fit_df_t = pd.DataFrame(fit_rows_t)
                fit_t = (
                    alt.Chart(fit_df_t)
                    .mark_line(color="#7f8c8d")
                    .encode(
                        x=alt.X(
                            "date:T",
                            title="Date",
                            axis=alt.Axis(
                                labelExpr="timeFormat(datum.value, '%b %Y')",
                                labelAngle=0,
                                labelPadding=6,
                                titlePadding=10,
                            ),
                        ),
                        y="Fit:Q",
                    )
                )
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
        # Ensure the Simulator shows an equation even if model fit wasn't run
        if "growth_equation_latex" not in st.session_state:
            st.session_state["growth_equation_latex"] = (
                r"F_t = F_{t-1}(1 - c_f) + F_{t-1}\,g + \frac{AdSpend_t}{CAC} - conv_t\\"
                r"P_t = P_{t-1}(1 - c_p) + conv_t\\"
                r"conv_t = (new^{free}_t)\,p_{new} + F_{t-1}\,p_{ongoing},\\"
                r"\quad new^{free}_t = F_{t-1}\,g + \frac{AdSpend_t}{CAC}"
            )
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

    # Brief status: show fitted segment growth rates if available
    fit_side = st.session_state.get("pwlog_fit")
    seg_rates = list(getattr(fit_side, "segment_growth_rates", []) or [])
    if seg_rates:
        # Prefer live overrides if present
        r_over = st.session_state.get("modelfit_r")
        r_src = r_over if r_over else seg_rates
        r_list_str = ", ".join(f"r{j+1}={r:0.3f}" for j, r in enumerate(r_src))
        st.sidebar.markdown(f"**Segments (r):** {r_list_str}")

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

    # Ad response feature parameters (used in Stage 2 feature building)
    with st.sidebar.expander("Ad response (features)", expanded=False):
        lam_sb = slider_state(
            "Adstock lambda (carryover)",
            min_value=0.0,
            max_value=0.99,
            default_value=float(_get_state("adstock_lambda", 0.5)),
            step=0.01,
            key="adstock_lambda",
        )
        theta_sb = number_input_state(
            "Log transform theta",
            min_value=1.0,
            default_value=float(_get_state("ad_log_theta", 500.0)),
            step=50.0,
            key="ad_log_theta",
        )

    # Model fit parameters (read from fit; allow manual override for what-if scenarios)
    with st.sidebar.expander("Model fit parameters", expanded=False):
        fit = st.session_state.get("pwlog_fit")
        if fit is None:
            st.caption("No model fit available yet. Run Model fit on the Estimators tab.")
        else:
            try:
                k_val = number_input_state(
                    "K (carrying capacity)",
                    min_value=0.0,
                    default_value=float(getattr(fit, "carrying_capacity", 0.0)),
                    step=100.0,
                    key="modelfit_K",
                )
                gp = number_input_state(
                    "gamma_pulse",
                    min_value=-10.0,
                    max_value=10.0,
                    default_value=float(getattr(fit, "gamma_pulse", 0.0)),
                    step=0.001,
                    key="modelfit_gamma_pulse",
                )
                gs = number_input_state(
                    "gamma_step",
                    min_value=-10.0,
                    max_value=10.0,
                    default_value=float(getattr(fit, "gamma_step", 0.0)),
                    step=0.001,
                    key="modelfit_gamma_step",
                )
                gx0 = getattr(fit, "gamma_exog", None)
                if gx0 is not None:
                    gx = number_input_state(
                        "gamma_exog (log ad)",
                        min_value=-10.0,
                        max_value=10.0,
                        default_value=float(gx0),
                        step=0.001,
                        key="modelfit_gamma_exog",
                    )

                # Segment growth rates r_j
                r_list = list(getattr(fit, "segment_growth_rates", []) or [])
                r_over = []
                for j, rj in enumerate(r_list, start=1):
                    r_val = number_input_state(
                        f"r segment {j}",
                        min_value=-10.0,
                        max_value=10.0,
                        default_value=float(rj),
                        step=0.001,
                        key=f"modelfit_r_{j}",
                    )
                    r_over.append(float(r_val))

                # Persist aggregate list (non-widget key) for convenience
                st.session_state["modelfit_r"] = r_over
            except Exception:
                st.caption("Model fit parameters available, but could not render editor.")

    # Map model-fit overrides into simulator: use last segment r as organic growth if available
    _k_now, _r_now, _gp_now, _gs_now, _gx_now = _current_fit_params()
    organic_from_fit = float(_r_now[-1]) if (_r_now and len(_r_now) > 0) else float(organic_growth)

    return SimulationInputs(
        starting_free_subscribers=int(start_free),
        starting_premium_subscribers=int(start_premium),
        horizon_months=int(horizon),
        organic_monthly_growth_rate=float(organic_from_fit),
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
                # Coerce invalid dates to NaT then drop them
                ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
                ev2 = ev2.dropna(subset=["date"])  # keep only rows with valid dates
                rows = []
                for _, r in ev2.iterrows():
                    d = r["date"]
                    # Skip rows where analysis cannot be computed
                    try:
                        pre, post = slope_around(total_series, d, window=6)
                        delta = post - pre
                    except Exception:
                        continue

                    # Safely coerce cost to a numeric value, defaulting to 0.0 on NaN/NaT/invalid
                    raw_cost = r.get("cost", 0.0)
                    cost_num = pd.to_numeric(raw_cost, errors="coerce")
                    cost = 0.0 if pd.isna(cost_num) else float(cost_num)

                    rows.append({"date": d, "type": r.get("type", ""), "slope_delta": delta, "cost": cost})
                if rows:
                    out = pd.DataFrame(rows)
                    st.dataframe(out, width="stretch")

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
                st.session_state["markers_source"] = (
                    "events"
                    if isinstance(st.session_state.get("events_df"), pd.DataFrame)
                    and not st.session_state["events_df"].empty
                    else "detect"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")


def _compute_estimates(
    all_series: Optional[pd.Series], paid_series: Optional[pd.Series], window_months: int = 6
) -> dict:
    return compute_estimates(all_series, paid_series, window_months)


@dataclass
class ImportContext:
    all_series: Optional[pd.Series]
    paid_series: Optional[pd.Series]
    plot_df: pd.DataFrame
    net_only: bool


def _to_monthly_last(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None or s.empty:
        return s
    s2 = pd.to_datetime(pd.Index(s.index), errors="coerce")
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=s2).dropna()
    return s.resample("ME").last()


def _build_plot_df(all_series: Optional[pd.Series], paid_series: Optional[pd.Series]) -> pd.DataFrame:
    plot_df = pd.DataFrame()
    if all_series is not None and not all_series.empty:
        plot_df["Total"] = all_series
    if paid_series is not None and not paid_series.empty:
        plot_df["Paid"] = paid_series
    if {"Total", "Paid"}.issubset(plot_df.columns):
        plot_df["Free"] = pd.to_numeric(plot_df["Total"], errors="coerce") - pd.to_numeric(
            plot_df["Paid"], errors="coerce"
        )
    return plot_df


def _safe_select_columns(head: pd.DataFrame, key_prefix: str) -> tuple[Optional[int], Optional[int]]:
    ncols = head.shape[1]
    if ncols < 2:
        st.error(f"{key_prefix.capitalize()} file needs at least 2 columns (date, count).")
        return None, None
    date_sel = st.selectbox(
        f"{key_prefix.capitalize()}: date column (index)",
        list(range(ncols)),
        index=0,
        key=f"{key_prefix}_date_sel",
    )
    count_sel = st.selectbox(
        f"{key_prefix.capitalize()}: count column (index)",
        list(range(ncols)),
        index=min(1, ncols - 1),
        key=f"{key_prefix}_count_sel",
    )
    return date_sel, count_sel


def _ui_upload_two_files() -> (
    tuple[Optional[Any], bool, Optional[int], Optional[int], Optional[Any], bool, Optional[int], Optional[int]]
):
    logger.info("Uploading two files")
    c_all, c_paid = st.columns(2)
    with c_all:
        all_file, all_has_header, all_date_sel, all_count_sel = upload_panel(
            "All subscribers file (CSV/XLSX, often downloaded as `[blogname]_emails_[date].csv`)",
            help_hint="Pick the time series of all subscribers over time.",
            key_prefix="all",
            default_header=False,
        )

    with c_paid:
        paid_file, paid_has_header, paid_date_sel, paid_count_sel = upload_panel(
            "Paid subscribers file (CSV/XLSX, often downloaded as `[blogname]_subscribers_[date].csv`)",
            help_hint="Pick the time series of paid subscribers over time.",
            key_prefix="paid",
            default_header=False,
        )

    return (
        all_file,
        all_has_header,
        all_date_sel,
        all_count_sel,
        paid_file,
        paid_has_header,
        paid_date_sel,
        paid_count_sel,
    )


def _parse_and_normalize_series(
    all_file,
    all_has_header,
    all_date_sel,
    all_count_sel,
    paid_file,
    paid_has_header,
    paid_date_sel,
    paid_count_sel,
) -> ImportContext:
    all_series = (
        read_series(all_file, all_has_header, all_date_sel, all_count_sel)
        if all_file is not None and all_date_sel is not None and all_count_sel is not None
        else None
    )
    paid_series = (
        read_series(paid_file, paid_has_header, paid_date_sel, paid_count_sel)
        if paid_file is not None and paid_date_sel is not None and paid_count_sel is not None
        else None
    )

    # Normalize to monthly once
    all_series_m = _to_monthly_last(all_series)
    paid_series_m = _to_monthly_last(paid_series)

    plot_df = _build_plot_df(all_series_m, paid_series_m)

    # Persist minimal state for other tabs
    if all_series_m is not None:
        st.session_state["import_total"] = all_series_m
    if paid_series_m is not None:
        st.session_state["import_paid"] = paid_series_m

    net_only = st.checkbox("Use net-only growth (set churn to 0)", value=True)
    return ImportContext(all_series_m, paid_series_m, plot_df, net_only)


def _ui_series_chart(plot_df: pd.DataFrame) -> tuple[bool, bool]:
    logger.info("_ui_series_chart has been called")
    logger.info(f"plot_df: {plot_df}")
    if plot_df.empty:
        return False, False
    use_dual_axis = st.checkbox(
        "Use separate right axis for Paid",
        value=True,
        help="Plots Total/Free on left axis and Paid on right axis for readability.",
    )
    show_total = st.checkbox("Show Total line", value=("Paid" not in plot_df.columns))
    series_title = "Series (Paid is dashed)" if (use_dual_axis and "Paid" in plot_df.columns) else "Series"
    base = plot_series(plot_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title)
    event_rules = _event_rules_from_events() if st.session_state.get("markers_source", "events") == "events" else None
    chart = alt.layer(base, event_rules) if event_rules is not None else base
    st.altair_chart(chart, use_container_width=True)
    return use_dual_axis, show_total


def _stage2_events_and_detection(plot_df: pd.DataFrame) -> tuple[list[int], Optional[str]]:
    target_col = "Total" if "Total" in plot_df.columns else ("Free" if "Free" in plot_df.columns else None)
    events_editor(plot_df, target_col)
    # Map Change events to breakpoints; show detected as reference only
    detected = trend_detection_ui(plot_df, target_col)
    change_dates = _events_change_dates()
    idx = plot_df[target_col].dropna().index if target_col is not None else plot_df.index
    bkps_from_events = _dates_to_breakpoint_indices(change_dates, idx)
    return (bkps_from_events if bkps_from_events else detected), target_col


def _stage5_tail(
    plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, target_col: Optional[str], breakpoints: list[int]
) -> None:
    tail_view_ui(plot_df, use_dual_axis, show_total, target_col, breakpoints)


def render_data_import() -> None:
    """
    Stage 1–5: Import, preview, annotate, feature-build, quick-fit, diagnostics, and handoff.
    """
    st.subheader("Stage 1: Import Substack exports (time series)")
    st.caption(
        "Upload two files: All subscribers over time, and Paid subscribers over time. "
        "We normalize everything to end-of-month (monthly). No headers by default: first column is date, second is count."
    )

    logger.info("Stage 1: entering Data Import")

    # Quick save/load
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

    # Uploads
    (
        all_file,
        all_has_header,
        all_date_sel,
        all_count_sel,
        paid_file,
        paid_has_header,
        paid_date_sel,
        paid_count_sel,
    ) = _ui_upload_two_files()

    # Only proceed if at least one file present
    if all_file is None and paid_file is None:
        return

    try:
        ctx = _parse_and_normalize_series(
            all_file,
            all_has_header,
            all_date_sel,
            all_count_sel,
            paid_file,
            paid_has_header,
            paid_date_sel,
            paid_count_sel,
        )

        if ctx.plot_df.empty:
            st.info("No usable data found after parsing/normalization.")
            return

        st.subheader("Imported series")
        st.caption("Mode: Paid and unpaid" if "Paid" in ctx.plot_df.columns else "Mode: Unpaid only")

        # Stage 1: observations
        if not ctx.plot_df.empty:
            emit_observations(ctx.plot_df)

        # Chart
        use_dual_axis, show_total = _ui_series_chart(ctx.plot_df)

        # Stage 2: events + detection + features
        breakpoints, target_col = _stage2_events_and_detection(ctx.plot_df)
        events_features_ui(ctx.plot_df)

        # Stage 3: Adds & Churn
        adds_and_churn_ui(ctx.plot_df)

        # Stage 4: Fit
        ev_dates_log = _events_change_dates()
        logger.info(
            "Stage 4: inputs — breakpoints=%s; change_dates=%s; plot_df_head=%s",
            breakpoints,
            [str(pd.to_datetime(d).date()) for d in (ev_dates_log or [])],
            ctx.plot_df.head(5).to_dict(orient="records"),
        )
        with st.expander("Stage 4 inputs (data & breakpoints)", expanded=False):
            st.write("Breakpoints (indices from 'Change' events):", breakpoints)
            st.write("Change event dates:", _events_change_dates())
            st.dataframe(ctx.plot_df, width="stretch")
        quick_fit_ui(ctx.plot_df, breakpoints)

        # Stage 5: Diagnostics tail
        _stage5_tail(ctx.plot_df, use_dual_axis, show_total, target_col, breakpoints)

        # Summary metrics + apply to simulator
        metrics_and_apply_ui(ctx.all_series, ctx.paid_series, net_only=ctx.net_only)

    except Exception as e:
        st.error(f"Estimation failed: {e}")


render_brand_header()

# Tabs
with st.container():
    tab_import, tab_sim, tab_est, tab_save, tab_stages, tab_help = st.tabs(
        [
            "Data Import",
            "Simulator",
            "Estimators",
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
    # If available, show the growth equation selected on the data/fit tab
    _eq = st.session_state.get("growth_equation_latex")
    if _eq:
        st.markdown("**Current growth equation**")
        st.latex(_eq)
    else:
        # Fallback: show the simulator's MVP equations
        st.markdown("**Current growth equation**")
        eq_sim = (
            r"F_t = F_{t-1}(1 - c_f) + F_{t-1}\,g + \frac{AdSpend_t}{CAC} - conv_t\\"
            r"P_t = P_{t-1}(1 - c_p) + conv_t\\"
            r"conv_t = (new^{free}_t)\,p_{new} + F_{t-1}\,p_{ongoing},\\"
            r"\quad new^{free}_t = F_{t-1}\,g + \frac{AdSpend_t}{CAC}"
        )
        st.latex(eq_sim)

    # Stage 7 summary: what we've done, what's calculated, what's next
    with st.expander("What we've done so far, what's calculated, and what's next", expanded=True):
        obs = st.session_state.get("observations_df")
        events_df = st.session_state.get("events_df")
        cov_df = st.session_state.get("covariates_df")
        feat_df = st.session_state.get("features_df")
        adds_df = st.session_state.get("adds_df")
        churn_df = st.session_state.get("churn_df")
        fit = st.session_state.get("pwlog_fit")
        lam = st.session_state.get("adstock_lambda")
        theta = st.session_state.get("ad_log_theta")

        st.markdown("**Done so far**")
        bullets = []
        if obs is not None:
            bullets.append(f"- Stage 1 (observations_df): {len(obs)} rows at current granularity")
        if events_df is not None and not getattr(events_df, "empty", True):
            try:
                n_events = len(events_df.dropna(subset=["date"]))
            except Exception:
                n_events = len(events_df)
            bullets.append(f"- Stage 2 (events_df): {n_events} {'event' if n_events == 1 else 'events'} annotated")
        if cov_df is not None:
            bullets.append("- Stage 2 (covariates_df): ad spend indexed monthly")
        if feat_df is not None and not getattr(feat_df, "empty", True):
            lam_str = "?" if lam is None else f"{lam:0.2f}"
            theta_str = "?" if theta is None else f"{theta:0.2f}"
            bullets.append(
                f"- Stage 2 (features_df): adstock a_t = x_t + lambda * a_(t-1) and log response log(1 + a_t/theta) built (lambda={lam_str}, theta={theta_str})"
            )
        if adds_df is not None and not getattr(adds_df, "empty", True):
            bullets.append(f"- Stage 3 (adds_df): {len(adds_df)} monthly rows")
        if churn_df is not None and not getattr(churn_df, "empty", True):
            bullets.append(f"- Stage 3 (churn_df): {len(churn_df)} monthly rows")
        if fit is not None:
            try:
                k_now, r_now, gp_now2, gs_now2, gx_now2 = _current_fit_params()
                k_disp = f"{int(float(k_now)):,}"
                rs = ", ".join(f"{r:0.3f}" for r in (r_now or []))
                bullets.append(f"- Stage 4 (Model fit): K={k_disp}; r by segment=[{rs}]")
                bullets.append(f"  - Events: gamma_pulse={gp_now2:0.3f}, gamma_step={gs_now2:0.3f}")
                if gx_now2 is not None:
                    bullets.append(f"  - Exogenous (log ad effect): gamma_exog={float(gx_now2):0.3f}")
            except Exception:
                bullets.append("- Stage 4 (Model fit): available")
        if not bullets:
            bullets.append("- No prior stages detected yet. Use the Estimators tab to run 1–5.")
        st.markdown("\n".join(bullets))

        st.markdown("**What we'll do next**")
        nxt = [
            "- Set assumptions in the sidebar: growth, churn, conversion, pricing, CAC, ad spend.",
            "- Run the Simulator below to project subscribers, revenue, ROAS, and payback.",
            "- Optional: use the model fit to calibrate growth dynamics; advanced Bayesian fit coming soon.",
        ]
        st.markdown("\n".join(nxt))
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

Status: Stages 1–3 (import, events, quick estimators) and Model fit are implemented. Pro Fit (Bayesian), diagnostics, and scenario planner are in progress.
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
