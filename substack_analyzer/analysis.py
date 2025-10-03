from __future__ import annotations

import math
from contextlib import suppress
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st


def read_series(file_like, has_header: bool, date_sel, count_sel) -> pd.Series:
    """Parse a CSV/XLSX time series into a monthly-indexed pandas Series.

    Expects two columns: date and count (or indices selected by user).
    Resamples to month end using last observation.
    """
    with suppress(Exception):
        file_like.seek(0)

    if getattr(file_like, "name", "").lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_like, header=0 if has_header else None)
    else:
        df = pd.read_csv(file_like, header=0 if has_header else None)

    if has_header:
        date_col = date_sel if isinstance(date_sel, str) else df.columns[int(date_sel)]
        count_col = count_sel if isinstance(count_sel, str) else df.columns[int(count_sel)]
    else:
        date_col = df.columns[int(date_sel)]
        count_col = df.columns[int(count_sel)]

    s = (
        df[[date_col, count_col]]
        .rename(columns={date_col: "date", count_col: "count"})
        .assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce"))
        .dropna(subset=["date"])
        .sort_values("date")
        .set_index("date")["count"]
    )
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not s.empty:
        s = s.resample("M").last().dropna()
    return s


def plot_series(plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, series_title: str) -> alt.Chart:
    base = alt.Chart(plot_df.reset_index().rename(columns={"index": "date"})).encode(
        x=alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(
                labelExpr="timeFormat(datum.value, '%b %Y')",
                labelAngle=0,
                labelPadding=6,
                titlePadding=10,
            ),
        )
    )
    left_series = [c for c in (["Total", "Free"] if show_total else ["Free"]) if c in plot_df.columns]
    layers = []
    if left_series:
        left = (
            base.transform_fold(left_series, as_=["Series", "Value"])
            .mark_line(point=True)
            .encode(
                y=alt.Y("Value:Q", axis=alt.Axis(title="Total / Free")),
                color=alt.Color("Series:N", scale=alt.Scale(scheme="tableau10"), title=series_title),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%b %Y"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Value:Q"),
                ],
            )
        )
        layers.append(left)
    if use_dual_axis and ("Paid" in plot_df.columns):
        right = (
            base.transform_fold(["Paid"], as_=["Series", "Value"])
            .mark_line(strokeDash=[4, 3], point=True)
            .encode(
                y=alt.Y("Value:Q", axis=alt.Axis(title="Paid", orient="right"), scale=alt.Scale(zero=True)),
                color=alt.Color("Series:N", scale=alt.Scale(range=["#DB4437"]), title=series_title),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%b %Y"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Value:Q"),
                ],
            )
        )
        layers.append(right)

    ev = st.session_state.get("events_df")
    if ev is not None and not ev.empty:
        ev2 = ev.dropna(subset=["date"]).copy()
        if not ev2.empty:
            # Use exact event dates for the chart so edits in the table reflect immediately
            with suppress(Exception):
                ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce")
            markers = (
                alt.Chart(ev2)
                .mark_rule(color="#8e44ad", size=3)
                .encode(
                    x="date:T",
                    tooltip=[
                        alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                        "type:N",
                        "notes:N",
                        "cost:Q",
                    ],
                )
            )
            layers.append(markers)

    chart = alt.layer(*layers).properties(height=260, padding={"bottom": 20, "left": 5, "right": 5, "top": 5})
    if use_dual_axis and ("Paid" in plot_df.columns):
        chart = chart.resolve_scale(y="independent")
    return chart


def compute_estimates(
    all_series: Optional[pd.Series], paid_series: Optional[pd.Series], window_months: int = 6
) -> dict:
    """Derive starting levels and rate estimates from monthly subscriber series.

    Computes lightweight, robust estimates to initialize the simulator and
    calibrations. Works in two modes depending on which inputs are provided:

    - All + Paid provided: Aligns both series, derives Free = All - Paid, and
      computes median-of-tail rates on the last `window_months` months for
      free growth and conversion proxy. Returns: `start_free`, `start_premium`,
      `organic_growth`, `conv_ongoing`.
    - Only All provided: Uses the All series and current session state's
      `start_premium` (if any) to back out `start_free`, and computes
      `organic_growth` from All deltas.
    - Only Paid provided: Returns `start_premium` from the last observation.

    Independently ensures churn defaults are present by reading
    `churn_free` and `churn_prem` from `st.session_state` (default 0.01).

    Parameters
    ----------
    all_series : Optional[pd.Series]
        Monthly All (Total) subscribers indexed by month-end timestamps.
    paid_series : Optional[pd.Series]
        Monthly Paid subscribers indexed by month-end timestamps.
    window_months : int, default 6
        Tail window size used for median rate calculations.

    Returns
    -------
    dict
        A dictionary including some or all of the following keys depending on
        inputs available: `start_free`, `start_premium`, `organic_growth`,
        `conv_ongoing`, `churn_free`, `churn_prem`.

    Notes
    -----
    - Rates are computed from first differences divided by previous levels and
      then aggregated via median over the tail window to reduce outlier impact.
    - Series are aligned on overlapping dates; if no overlap exists in the
      All+Paid path, a ValueError is raised.
    - Inputs are not modified; computations are performed on copies.
    """
    estimates: dict = {}

    def _get_state(key: str, default):
        return st.session_state.get(key, default)

    def _median_positive(s: pd.Series) -> float:
        s = s.dropna()
        s = s[s > 0]
        return float(s.median()) if not s.empty else 0.0

    if all_series is not None and paid_series is not None and not all_series.empty and not paid_series.empty:
        aligned = pd.concat([all_series, paid_series], axis=1, keys=["all", "paid"]).dropna()
        if aligned.empty:
            raise ValueError("No overlapping dates between All and Paid series")
        aligned["free"] = aligned["all"] - aligned["paid"]
        for col in ["free", "paid"]:
            aligned[f"{col}_delta"] = aligned[col].diff()
            aligned[f"{col}_rate"] = aligned[f"{col}_delta"] / aligned[col].shift(1)
        aligned["conv_proxy"] = aligned["paid_delta"] / aligned["free"].shift(1)
        tail = aligned.tail(window_months)
        estimates.update(
            {
                "start_free": int(aligned["free"].iloc[-1]),
                "start_premium": int(aligned["paid"].iloc[-1]),
                "organic_growth": _median_positive(tail["free_rate"]),
                "conv_ongoing": _median_positive(tail["conv_proxy"]),
            }
        )
    elif all_series is not None and not all_series.empty:
        total = all_series
        total_delta = total.diff()
        total_rate = total_delta / total.shift(1)
        tail = total_rate.tail(window_months)
        estimates.update(
            {
                "start_free": int(total.iloc[-1] - int(_get_state("start_premium", 0))),
                "organic_growth": _median_positive(tail),
            }
        )
    elif paid_series is not None and not paid_series.empty:
        estimates.update({"start_premium": int(paid_series.iloc[-1])})

    if "churn_free" not in estimates:
        estimates["churn_free"] = float(_get_state("churn_free", 0.01))
    if "churn_prem" not in estimates:
        estimates["churn_prem"] = float(_get_state("churn_prem", 0.01))

    return estimates


def build_events_features(plot_df: pd.DataFrame, lam: float, theta: float, ad_file):
    """Build monthly covariates/features from Events and optional ad spend.

    This function encodes the app's Events table into time-aligned monthly
    features and optionally derives advertising-related covariates from an
    uploaded ad spend file. All outputs are aligned to the monthly index of
    `plot_df` (month-end timestamps).

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame whose index defines the monthly timeline. Only the index is
        used here; columns such as "Total" are not required.
    lam : float
        Adstock carryover parameter in [0, 1). Higher values yield longer
        persistence of past ad spend in the `adstock` feature.
    theta : float
        Scale parameter for the log transform used in `ad_effect_log`.
        The feature is computed as log(1 + adstock/theta).
    ad_file : file-like or None
        Optional CSV/XLSX with columns {"date", "spend"}. Values are grouped to
        month end and aligned to the index as the `ad_spend` covariate.

    Reads
    -----
    st.session_state["events_df"] : pd.DataFrame (optional)
        Expected columns: "date" (any parseable date), "type" (e.g., "Ad spend",
        "Launch"), optional "persistence" ("Transient" or "Persistent"), and
        optional "cost" (used as a weight for ad-type pulses). Dates are
        normalised to month end.

    Returns
    -------
    covariates_df : pd.DataFrame
        Contains a single column `ad_spend` (monthly), zero if no ad file.
    features_df : pd.DataFrame
        Columns:
          - `pulse`: transient spikes at event months; for ad-like events the
            magnitude defaults to `cost` (or 1.0 if missing).
          - `step`: persistent unit-step from event month onward.
          - `adstock`: recursive carryover of `ad_spend` using `lam`.
          - `ad_effect_log`: log(1 + adstock/theta).

    Notes
    -----
    - Event dates falling outside the monthly index are ignored.
    - If an event's persistence is unspecified, it contributes to both `pulse`
      and `step` for backward compatibility.
    - All outputs are float-valued and aligned to the month-end index.
    """
    monthly_index = plot_df.index
    pulse = pd.Series(0.0, index=monthly_index, name="pulse")
    step = pd.Series(0.0, index=monthly_index, name="step")

    ev_src = st.session_state.get("events_df")
    if ev_src is not None and not ev_src.empty:
        ev2 = ev_src.dropna(subset=["date"]).copy()
        with suppress(Exception):
            ev2["date"] = pd.to_datetime(ev2["date"]).dt.to_period("M").dt.to_timestamp("M")
        for _, r in ev2.iterrows():
            d = r.get("date")
            if pd.isna(d) or d not in monthly_index:
                continue
            cost = float(r.get("cost", 1.0) or 1.0)
            kind = str(r.get("type", ""))
            weight = cost if kind.lower() in {"ad spend", "ad"} else 1.0
            persistence = str(r.get("persistence", "")).strip().lower()
            if persistence == "persistent":
                step.loc[d:] += 1.0
            elif persistence == "transient":
                pulse.loc[d] += float(weight)
            else:
                # Backward-compatibility: if unspecified, treat as both
                pulse.loc[d] += float(weight)
                step.loc[d:] += 1.0

    ad_spend = pd.Series(0.0, index=monthly_index, name="ad_spend")
    if ad_file is not None:
        try:
            if ad_file.name.lower().endswith((".xlsx", ".xls")):
                ad_df = pd.read_excel(ad_file)
            else:
                ad_df = pd.read_csv(ad_file)
            if {"date", "spend"}.issubset(ad_df.columns):
                ad_df = ad_df.assign(date=lambda d: pd.to_datetime(d["date"]))
                ad_df = ad_df.dropna(subset=["date"])
                ad_df["date"] = ad_df["date"].dt.to_period("M").dt.to_timestamp("M")
                ad_df = ad_df.groupby("date", as_index=True)["spend"].sum().sort_index()
                ad_spend = ad_df.reindex(monthly_index).fillna(0.0)
        except Exception:
            pass

    adstock = pd.Series(0.0, index=monthly_index, name="adstock")
    if not ad_spend.empty:
        prev = 0.0
        vals: list[float] = []
        for v in ad_spend.to_list():
            s_val = float(v) + float(lam) * float(prev)
            vals.append(s_val)
            prev = s_val
        adstock = pd.Series(vals, index=monthly_index, name="adstock")

    ad_effect_log = pd.Series(
        (adstock / max(theta, 1e-9)).add(1.0).apply(lambda x: float(math.log(x))),
        index=monthly_index,
        name="ad_effect_log",
    )

    covariates_df = pd.DataFrame({"ad_spend": ad_spend})
    features_df = pd.DataFrame({"pulse": pulse, "step": step, "adstock": adstock, "ad_effect_log": ad_effect_log})
    return covariates_df, features_df


def derive_adds_churn(plot_df: pd.DataFrame, churn_free_est: float, churn_paid_est: float):
    total_s = plot_df.get("Total") if "Total" in plot_df.columns else None
    paid_s = plot_df.get("Paid") if "Paid" in plot_df.columns else None
    free_s = (
        (total_s.astype(float) - paid_s.astype(float)).clip(lower=0)
        if (total_s is not None and paid_s is not None)
        else None
    )

    adds_rows, churn_rows = [], []
    idx = plot_df.index
    for i, d in enumerate(idx):
        if i == 0:
            continue
        prev_d = idx[i - 1]
        if free_s is not None:
            free_prev, free_now = float(free_s.loc[prev_d]), float(free_s.loc[d])
            canc_free = max(free_prev * churn_free_est, 0.0)
            adds_free = max((free_now - free_prev) + canc_free, 0.0)
        else:
            canc_free = math.nan
            adds_free = math.nan

        if paid_s is not None:
            paid_prev, paid_now = float(paid_s.loc[prev_d]), float(paid_s.loc[d])
            canc_paid = max(paid_prev * churn_paid_est, 0.0)
            adds_paid = max((paid_now - paid_prev) + canc_paid, 0.0)
        else:
            canc_paid = math.nan
            adds_paid = math.nan

        adds_rows.append({"date": d, "gross_adds_free": adds_free, "gross_adds_paid": adds_paid})
        churn_rows.append({"date": d, "cancels_free": canc_free, "cancels_paid": canc_paid})

    adds_df = pd.DataFrame(adds_rows).set_index("date") if adds_rows else pd.DataFrame()
    churn_df = pd.DataFrame(churn_rows).set_index("date") if churn_rows else pd.DataFrame()
    return adds_df, churn_df
