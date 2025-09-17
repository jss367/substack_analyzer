from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd

# Note: ruptures was previously used; keep import only if needed elsewhere
import ruptures as rpt  # noqa: F401
import streamlit as st
import streamlit.components.v1 as components

from substack_analyzer.model import AdSpendSchedule, SimulationInputs, simulate_growth

# Drag components removed; no custom component declarations needed


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
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">
        <style>
        :root {
            --brand-accent: #B92D24;
            --brand-green-dark: #5F6E60;
            --brand-green-light: #A6C4A7;
            --brand-bg: #FFFCF2;
            --brand-text: #2C2626;
        }
        html, body, .stApp { font-family: Helvetica, Arial, sans-serif; color: var(--brand-text); }
        .stApp { background-color: var(--brand-bg) !important; }
        [data-testid="stSidebar"] { background-color: var(--brand-green-light) !important; }
        [data-testid="stSidebar"] * { color: var(--brand-text); }
        h1, h2, h3, h4, h5, h6 { font-family: 'Source Serif 4', Georgia, serif; color: var(--brand-green-dark); }
        a { color: var(--brand-accent); }
        .stButton>button { background-color: var(--brand-green-dark); color: #fff; border: 0; border-radius: 6px; }
        .stButton>button:hover { background-color: #4d5a50; }
        /* Make header transparent to let theme show */
        .stApp header { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    c1, c2 = st.columns([1, 5])
    with c1:
        if LOGO_FULL.exists():
            st.image(str(LOGO_FULL), use_container_width=True)
        elif LOGO_ICON.exists():
            st.image(str(LOGO_ICON), width=96)
    with c2:
        st.markdown(
            "<div style='padding-top:8px;'><h1 style='margin-bottom:0;'>Substack Ads ROI Simulator</h1></div>",
            unsafe_allow_html=True,
        )
    st.divider()


# Apply brand styles and sidebar logo once
_inject_brand_styles()
if LOGO_FULL.exists() or LOGO_ICON.exists():
    st.sidebar.image(str(LOGO_FULL if LOGO_FULL.exists() else LOGO_ICON), use_container_width=True)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _get_state(key: str, default):
    return st.session_state.get(key, default)


def _format_date_badges(dates: list[pd.Timestamp | str]) -> str:
    items: list[str] = []
    for d in dates:
        try:
            dt = pd.to_datetime(d)
            label = dt.strftime("%b %d, %Y")
        except Exception:
            label = str(d)
        items.append(
            f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:999px;border:1px solid #8e44ad;color:#8e44ad;font-size:12px;'>"  # noqa: E501
            f"{label}</span>"
        )
    return "".join(items)


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


@dataclass(frozen=True)
class SegmentSlope:
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    slope_per_month: float


def _fit_slope_per_month(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = pd.RangeIndex(len(values)).to_numpy(dtype=float)
    y = values.to_numpy(dtype=float)
    denom = float(((x - x.mean()) ** 2).sum())
    if denom == 0.0:
        return 0.0
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
    return slope


def detect_change_points(series: pd.Series, max_changes: int = 4) -> list[int]:
    """Detect change points emphasizing slope changes rather than level shifts.

    Strategy:
    - Work on the first difference (monthly deltas) and then its difference
      (acceleration). Large absolute acceleration indicates a slope change.
    - Pick the top-k local maxima in |acceleration| with a minimum spacing
      between change points to avoid clustered duplicates.
    - Return indices relative to the original monthly series.
    """
    s = series.dropna()
    n = s.shape[0]
    if n < 6:
        return []

    # First and second differences
    delta = s.diff().dropna()
    accel = delta.diff().dropna()
    if accel.empty:
        return []

    # Score by absolute acceleration
    score = accel.abs()

    # Identify candidate peaks (greater than neighbors)
    candidates: list[pd.Timestamp] = []
    values: list[float] = []
    accel_vals = score.to_numpy()
    accel_index = score.index
    for i in range(1, len(accel_vals) - 1):
        if accel_vals[i] >= accel_vals[i - 1] and accel_vals[i] >= accel_vals[i + 1]:
            candidates.append(accel_index[i])
            values.append(float(accel_vals[i]))

    if not candidates:
        return []

    # Sort by magnitude descending and enforce min separation (in months)
    order = sorted(range(len(candidates)), key=lambda k: values[k], reverse=True)
    min_separation = 2  # months
    selected_dates: list[pd.Timestamp] = []
    for idx in order:
        d = candidates[idx]
        # Enforce spacing relative to already selected
        if all(abs(s.index.get_loc(d) - s.index.get_loc(sd)) >= min_separation for sd in selected_dates):
            selected_dates.append(d)
        if len(selected_dates) >= max_changes:
            break

    # Map dates back to series indices
    indices = [int(s.index.get_loc(d)) for d in sorted(selected_dates)]
    # Ensure indices are within bounds [1, n-1]
    indices = [i for i in indices if 0 <= i < n]
    return indices


def compute_segment_slopes(series: pd.Series, breakpoints: list[int]) -> list[SegmentSlope]:
    s = series.dropna()
    if not breakpoints:
        breakpoints = [s.shape[0]]
    segments: list[SegmentSlope] = []
    start = 0
    for bp in breakpoints:
        seg_vals = s.iloc[start:bp]
        slope = _fit_slope_per_month(seg_vals)
        segments.append(
            SegmentSlope(
                start_index=start,
                end_index=bp - 1,
                start_date=s.index[start],
                end_date=s.index[bp - 1],
                slope_per_month=float(slope),
            )
        )
        start = bp
    return segments


def slope_around(series: pd.Series, event_date: pd.Timestamp, window: int = 6) -> tuple[float, float]:
    """Return (pre_slope, post_slope) using +/- window months around event_date."""
    s = series.dropna()
    if s.empty:
        return (0.0, 0.0)
    # Find closest index at or before event_date
    idx = s.index.searchsorted(event_date, side="right") - 1
    idx = max(min(idx, len(s) - 2), 1)
    start_pre = max(0, idx - window + 1)
    end_pre = idx + 1
    start_post = idx + 1
    end_post = min(len(s), idx + 1 + window)
    pre = _fit_slope_per_month(s.iloc[start_pre:end_pre])
    post = _fit_slope_per_month(s.iloc[start_post:end_post])
    return (float(pre), float(post))


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


def _read_series(file, has_header: bool, date_sel, count_sel) -> pd.Series:
    # Reset pointer in case file was read for preview
    with suppress(Exception):
        file.seek(0)

    if getattr(file, "name", "").lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, header=0 if has_header else None)
    else:
        df = pd.read_csv(file, header=0 if has_header else None)

    # Determine column names from selection (index or string)
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
        .dropna(subset=["date"])  # drop rows with invalid dates
        .sort_values("date")
        .set_index("date")["count"]
    )
    s = pd.to_numeric(s, errors="coerce").dropna()
    # Resample to month end using last observation
    if not s.empty:
        s = s.resample("M").last().dropna()
    return s


def _compute_estimates(
    all_series: Optional[pd.Series], paid_series: Optional[pd.Series], window_months: int = 6
) -> dict:
    estimates: dict = {}

    # Helper to median positive
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
                "organic_growth": _median_positive(tail["free_rate"]),  # net free growth
                "conv_ongoing": _median_positive(tail["conv_proxy"]),
            }
        )
    elif all_series is not None and not all_series.empty:
        # Only total subscribers series; assume premium unchanged (use existing state)
        total = all_series
        total_delta = total.diff()
        total_rate = total_delta / total.shift(1)
        tail = total_rate.tail(window_months)
        estimates.update(
            {
                "start_free": int(total.iloc[-1] - int(_get_state("start_premium", 0))),
                "organic_growth": _median_positive(tail),  # net total rate as proxy
            }
        )
    elif paid_series is not None and not paid_series.empty:
        # Only paid series; we can set starting premium
        estimates.update({"start_premium": int(paid_series.iloc[-1])})

    # Keep churn defaults if not computed elsewhere (cannot infer from topline series)
    if "churn_free" not in estimates:
        estimates["churn_free"] = float(_get_state("churn_free", 0.01))
    if "churn_prem" not in estimates:
        estimates["churn_prem"] = float(_get_state("churn_prem", 0.01))

    return estimates


def render_data_import() -> None:
    st.subheader("Import Substack exports (time series)")
    st.caption(
        "Upload two files: All subscribers over time, and Paid subscribers over time. "
        "No headers by default: first column is date, second is count."
    )

    c_all, c_paid = st.columns(2)

    with c_all:
        all_file = st.file_uploader(
            "All subscribers file (CSV/XLSX, often downloaded as `[blogname]_emails_[date].csv`)",
            type=["csv", "xlsx", "xls"],
            key="all_file",
        )
        all_has_header = st.checkbox("All file has header row", value=False, key="all_has_header")
        all_date_sel: Optional[int] = 0
        all_count_sel: Optional[int] = 1
        if all_file is not None:
            try:
                if all_file.name.lower().endswith((".xlsx", ".xls")):
                    tmp = pd.read_excel(all_file, header=0 if all_has_header else None, nrows=5)
                else:
                    tmp = pd.read_csv(all_file, header=0 if all_has_header else None, nrows=5)
                # Reset pointer for later full read
                all_file.seek(0)
                ncols = tmp.shape[1]
                all_date_sel = st.selectbox("All: date column (index)", list(range(ncols)), index=0, key="all_date_sel")
                all_count_sel = st.selectbox(
                    "All: count column (index)", list(range(ncols)), index=min(1, ncols - 1), key="all_count_sel"
                )
            except Exception as e:
                st.error(f"Could not read All file: {e}")
                all_file = None

    with c_paid:
        paid_file = st.file_uploader(
            "Paid subscribers file (CSV/XLSX, often downloaded as `[blogname]_subscribers_[date].csv`)",
            type=["csv", "xlsx", "xls"],
            key="paid_file",
        )
        paid_has_header = st.checkbox("Paid file has header row", value=False, key="paid_has_header")
        paid_date_sel: Optional[int] = 0
        paid_count_sel: Optional[int] = 1
        if paid_file is not None:
            try:
                if paid_file.name.lower().endswith((".xlsx", ".xls")):
                    tmp2 = pd.read_excel(paid_file, header=0 if paid_has_header else None, nrows=5)
                else:
                    tmp2 = pd.read_csv(paid_file, header=0 if paid_has_header else None, nrows=5)
                paid_file.seek(0)
                ncols2 = tmp2.shape[1]
                paid_date_sel = st.selectbox(
                    "Paid: date column (index)", list(range(ncols2)), index=0, key="paid_date_sel"
                )
                paid_count_sel = st.selectbox(
                    "Paid: count column (index)", list(range(ncols2)), index=min(1, ncols2 - 1), key="paid_count_sel"
                )
            except Exception as e:
                st.error(f"Could not read Paid file: {e}")
                paid_file = None

    # Proceed if at least one file is present
    if all_file is not None or paid_file is not None:
        # We'll show the window slider near the tail chart it affects
        window = int(_get_state("est_window", 6))
        net_only = st.checkbox("Use net-only growth (set churn to 0)", value=True)
        debug_mode = st.checkbox("Enable debug logging", value=True, help="Adds console logs and inline diagnostics.")
        try:
            all_series = None
            paid_series = None
            if all_file is not None:
                all_series = _read_series(all_file, all_has_header, all_date_sel, all_count_sel)
            if paid_file is not None:
                paid_series = _read_series(paid_file, paid_has_header, paid_date_sel, paid_count_sel)

            # Preview charts + trend detection
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
                # Dual-axis toggle for visibility when Paid is much smaller
                use_dual_axis = st.checkbox(
                    "Use separate right axis for Paid",
                    value=True,
                    help="Plots Total/Free on left axis and Paid on right axis for readability.",
                )
                # Option to hide/show Total line by default (on when Paid missing)
                default_show_total = "Paid" not in plot_df.columns
                show_total = st.checkbox("Show Total line", value=default_show_total)
                # Legend title reflects whether Paid is plotted
                has_paid = "Paid" in plot_df.columns
                plotting_paid = bool(use_dual_axis) and has_paid
                series_title = "Series (Paid is dashed)" if plotting_paid else "Series"
                # Altair chart
                base = alt.Chart(plot_df.reset_index().rename(columns={"index": "date"})).encode(
                    x=alt.X("date:T", title="Date")
                )

                left = (
                    base.transform_fold(
                        [c for c in (["Total", "Free"] if show_total else ["Free"]) if c in plot_df.columns],
                        as_=["Series", "Value"],
                    )
                    .mark_line(point=True)
                    .encode(
                        y=alt.Y("Value:Q", axis=alt.Axis(title="Total / Free")),
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(scheme="tableau10"),
                            title=series_title,
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Value:Q", title="Value"),
                        ],
                    )
                )

                layers = [left]
                if use_dual_axis and ("Paid" in plot_df.columns):
                    right = (
                        base.transform_fold(["Paid"], as_=["Series", "Value"])
                        .mark_line(strokeDash=[4, 3], point=True)
                        .encode(
                            y=alt.Y(
                                "Value:Q",
                                axis=alt.Axis(title="Paid", orient="right"),
                                scale=alt.Scale(zero=True),
                            ),
                            color=alt.Color(
                                "Series:N",
                                scale=alt.Scale(range=["#DB4437"]),
                                title=series_title,
                            ),
                            tooltip=[
                                alt.Tooltip("date:T", title="Date"),
                                alt.Tooltip("Series:N", title="Series"),
                                alt.Tooltip("Value:Q", title="Value"),
                            ],
                        )
                    )
                    layers.append(right)

                # Overlay event markers on main chart if present
                if (ev := st.session_state.get("events_df")) is not None and not ev.empty:
                    ev2 = ev.dropna(subset=["date"]).copy()
                    if not ev2.empty:
                        ev2["date"] = pd.to_datetime(ev2["date"]).dt.to_period("M").dt.to_timestamp("M")
                        markers = (
                            alt.Chart(ev2)
                            .mark_rule(color="#8e44ad", size=3)
                            .encode(
                                x="date:T",
                                tooltip=["date:T", "type:N", "notes:N", "cost:Q"],
                            )
                        )
                        layers.append(markers)
                chart = alt.layer(*layers)
                if use_dual_axis and ("Paid" in plot_df.columns):
                    chart = chart.resolve_scale(y="independent")
                chart = chart.properties(height=260)
                st.altair_chart(chart, use_container_width=True)

                # Change-point detection (on Total if present, otherwise Free)
                st.caption("Detected trend changes help annotate what happened (e.g., shout‑outs, ad spend).")
                target_col = "Total" if "Total" in plot_df.columns else ("Free" if "Free" in plot_df.columns else None)
                breakpoints: list[int] = []
                if target_col is not None:
                    max_bkps = st.slider(
                        "Max changes to detect",
                        0,
                        8,
                        3,
                        1,
                        key="max_changes_detect",
                    )
                    try:
                        breakpoints = detect_change_points(plot_df[target_col], max_changes=max_bkps)
                    except Exception:
                        breakpoints = []
                if breakpoints:
                    s_idx = plot_df[target_col].dropna().index
                    dates = [pd.to_datetime(s_idx[i]) for i in breakpoints if i < len(s_idx)]
                    st.markdown("**Detected change dates (on %s):**" % target_col)
                    st.markdown(_format_date_badges(dates), unsafe_allow_html=True)
                    st.caption("Tip: To adjust these, use the draggable timeline below (purple bars).")
                    # Offer to seed events table with detected dates
                    if st.button("Add detected change dates to Events"):
                        try:
                            s_idx = plot_df[target_col].dropna().index
                            change_dates = [s_idx[i - 1] if i > 0 else s_idx[i] for i in breakpoints]
                            seeded = pd.DataFrame(
                                {
                                    "date": [pd.to_datetime(d).date() for d in change_dates],
                                    "type": ["Change"] * len(change_dates),
                                    "notes": [f"Detected change in {target_col}"] * len(change_dates),
                                    "cost": [0.0] * len(change_dates),
                                }
                            )
                            existing = st.session_state.get("events_df")
                            if existing is not None and not existing.empty:
                                merged = pd.concat([existing, seeded], ignore_index=True)
                            else:
                                merged = seeded
                            st.session_state["events_df"] = merged
                            st.success("Added detected change dates to Events.")
                        except Exception:
                            st.info("Could not add detected dates. Try again after loading data.")

                # Events table: add/edit annotations
                st.subheader("Events & annotations")
                st.caption("Track shout-outs, ad campaigns, launches, etc. Dates must match the series timeline.")
                default_events = pd.DataFrame(
                    [
                        {"date": None, "type": "Ad spend", "notes": "", "cost": 0.0},
                    ]
                )
                events_df = st.session_state.get("events_df", default_events)
                edited = st.data_editor(
                    events_df,
                    num_rows="dynamic",
                    column_config={
                        "date": st.column_config.DateColumn("Date"),
                        "type": st.column_config.SelectboxColumn(
                            "Type", options=["Ad spend", "Shout-out", "Other"], width="medium"
                        ),
                        "notes": st.column_config.TextColumn("Notes", width="large"),
                        "cost": st.column_config.NumberColumn("Cost ($)", step=10.0, help="For Ad spend ROI calc"),
                    },
                    use_container_width=True,
                    key="events_editor",
                )
                st.session_state["events_df"] = edited

                # draggable timeline component removed

                # drag-edit UI removed
                # Deltas
                deltas = plot_df.diff()
                st.subheader("Monthly change (delta)")
                st.bar_chart(deltas.fillna(0))
                # Tail-only view with window slider placed here
                window = st.slider("Estimation window (last N months)", 3, 12, window, 1, key="est_window")
                st.caption("This window recomputes trailing medians for the estimates and the tail chart below.")
                st.subheader(f"Last {window} months (tail)")
                tail_df = plot_df.tail(window)
                # Tail chart: always use Altair so markers appear regardless of Paid
                base_t = alt.Chart(tail_df.reset_index().rename(columns={"index": "date"})).encode(
                    x=alt.X("date:T", title="Date")
                )
                # Tail legend title reflects whether Paid is plotted in tail
                tail_has_paid = "Paid" in tail_df.columns
                plotting_paid_t = bool(use_dual_axis) and tail_has_paid
                series_title_t = "Series (Paid is dashed)" if plotting_paid_t else "Series"
                left_t = (
                    base_t.transform_fold(
                        [c for c in (["Total", "Free"] if show_total else ["Free"]) if c in tail_df.columns],
                        as_=["Series", "Value"],
                    )
                    .mark_line(point=True)
                    .encode(
                        y=alt.Y("Value:Q", axis=alt.Axis(title="Total / Free")),
                        color=alt.Color(
                            "Series:N",
                            scale=alt.Scale(scheme="tableau10"),
                            title=series_title_t,
                        ),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Value:Q", title="Value"),
                        ],
                    )
                )
                layers_t = [left_t]
                if "Paid" in tail_df.columns and use_dual_axis:
                    right_t = (
                        base_t.transform_fold(["Paid"], as_=["Series", "Value"])
                        .mark_line(strokeDash=[4, 3], point=True)
                        .encode(
                            y=alt.Y("Value:Q", axis=alt.Axis(title="Paid", orient="right")),
                            color=alt.Color(
                                "Series:N",
                                scale=alt.Scale(range=["#DB4437"]),
                                title=series_title_t,
                            ),
                            tooltip=[
                                alt.Tooltip("date:T", title="Date"),
                                alt.Tooltip("Series:N", title="Series"),
                                alt.Tooltip("Value:Q", title="Value"),
                            ],
                        )
                    )
                    layers_t.append(right_t)
                # Markers on tail chart
                if (ev := st.session_state.get("events_df")) is not None and not ev.empty:
                    ev2 = ev.dropna(subset=["date"]).copy()
                    if not ev2.empty:
                        ev2["date"] = pd.to_datetime(ev2["date"]).dt.to_period("M").dt.to_timestamp("M")
                        markers_t = (
                            alt.Chart(ev2)
                            .mark_rule(color="#8e44ad", size=3)
                            .encode(
                                x="date:T",
                                tooltip=["date:T", "type:N", "notes:N", "cost:Q"],
                            )
                        )
                        layers_t.append(markers_t)
                if target_col is not None and breakpoints:
                    s_t = tail_df[target_col].dropna()
                    # Convert breakpoints from full series to tail indices if possible
                    segs_t = []
                    try:
                        full_s = plot_df[target_col].dropna()
                        segs = compute_segment_slopes(full_s, breakpoints)
                        # Keep segments intersecting the tail range
                        tail_start = s_t.index[0]
                        tail_end = s_t.index[-1]
                        for seg in segs:
                            if seg.end_date >= tail_start and seg.start_date <= tail_end:
                                segs_t.append(seg)
                    except Exception:
                        segs_t = []
                    fit_rows_t = []
                    for seg in segs_t:
                        xs = pd.date_range(
                            max(seg.start_date, s_t.index[0]),
                            min(seg.end_date, s_t.index[-1]),
                            freq="M",
                        )
                        start_val = float(full_s.loc[seg.start_date]) if 'full_s' in locals() else float(s_t.iloc[0])
                        for i, d in enumerate(xs):
                            fit_rows_t.append({"date": d, "Fit": start_val + seg.slope_per_month * i})
                    if fit_rows_t:
                        fit_df_t = pd.DataFrame(fit_rows_t)
                        fit_t = alt.Chart(fit_df_t).mark_line(color="#7f8c8d").encode(x="date:T", y="Fit:Q")
                        layers_t.append(fit_t)
                chart_t = alt.layer(*layers_t).resolve_scale(y="independent").properties(height=240)
                st.altair_chart(chart_t, use_container_width=True)

            estimates = _compute_estimates(all_series, paid_series, window)

            # Display whatever we computed
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
                "Notes: From totals alone we can compute net growth and a conversion proxy (when both series present). "
                "Churn and CAC need more detail."
            )

            if st.button("Apply estimates to Simulator"):
                # Apply core estimates
                for k, v in estimates.items():
                    st.session_state[k] = v
                # Net-only churn handling
                if net_only:
                    st.session_state["churn_free"] = 0.0
                    st.session_state["churn_prem"] = 0.0
                # Zero ad spend by default to avoid inflated projections
                st.session_state["ad_stage1"] = 0.0
                st.session_state["ad_stage2"] = 0.0
                st.session_state["ad_const"] = 0.0
                st.session_state["spend_mode_index"] = 1  # Constant
                # Set immediate conversion to 0 unless user overrides
                st.session_state["conv_new"] = 0.0
                st.session_state["horizon_months"] = max(int(_get_state("horizon_months", 60)), 24)
                # Set switch flag and rerun to trigger tab switch
                st.session_state["switch_to_sim"] = True
                st.success("Applied. Switching to Simulator…")
                st.rerun()
        except Exception as e:
            st.error(f"Estimation failed: {e}")


def render_outputs_formulas() -> None:
    st.subheader("Outputs and formulas")

    st.markdown(
        """
### Subscribers
- **Free subscribers (month m)**
  - **Inputs**: starting_free_subscribers, monthly_churn_rate_free, organic_monthly_growth_rate,
    ad_spend_schedule, cost_per_new_free_subscriber, new_subscriber_premium_conv_rate,
    ongoing_premium_conv_rate
  - **Calc**:
    - free_churned = free_prev × churn_free
    - free_after_churn = free_prev − free_churned
    - new_free_organic = free_after_churn × organic_growth
    - ad_spend_m = schedule(m); new_free_paid = ad_spend_m ÷ CAC
    - new_free_total = new_free_organic + new_free_paid
    - convert_from_new = new_free_total × new_subscriber_premium_conv_rate
    - convert_from_existing = max(free_after_churn + new_free_total − new_free_total, 0) × ongoing_rate
    - free_m = free_after_churn + new_free_total − (convert_from_new + convert_from_existing)

- **Premium subscribers (month m)**
  - **Inputs**: starting_premium_subscribers, monthly_churn_rate_premium,
    new_subscriber_premium_conv_rate, ongoing_premium_conv_rate (and free dynamics above)
  - **Calc**:
    - prem_churned = prem_prev × churn_premium
    - prem_after_churn = prem_prev − prem_churned
    - prem_m = prem_after_churn + convert_from_new + convert_from_existing

- **Total subscribers** = free_m + prem_m

### Revenue
- **Net monthly revenue per premium**
  - **Inputs**: premium_monthly_price_gross, substack_fee_pct, stripe_fee_pct, stripe_flat_fee
  - **Calc**: net_monthly_per_premium = gross × (1 − substack − stripe) − stripe_flat

- **Net annual revenue per premium**
  - **Inputs**: premium_annual_price_gross, substack_fee_pct, stripe_fee_pct, stripe_flat_fee
  - **Calc**: net_annual_per_premium = gross × (1 − substack − stripe) − stripe_flat

- **Net MRR (month m)**
  - **Inputs**: premium_subscribers_m, annual_share, net_monthly_per_premium
  - **Calc**: monthly_premium = prem_m × (1 − annual_share)
    - mrr_net = monthly_premium × net_monthly_per_premium

- **Net revenue (month m)**
  - **Inputs**: mrr_net, annual_share, net_annual_per_premium
  - **Calc**: annual_revenue_amortized = (prem_m × annual_share × net_annual_per_premium) ÷ 12
    - net_revenue = mrr_net + annual_revenue_amortized

### Costs and profit
- **Ad spend (month m)**
  - **Inputs**: ad_spend_schedule
  - **Calc**: ad_spend_m = schedule(m)

- **Ad manager fee (month m)**
  - **Inputs**: ad_manager_monthly_fee
  - **Calc**: ad_manager_fee_m = ad_manager_monthly_fee if ad_spend_m > 0 else 0

- **Profit (month m)**
  - **Calc**: profit_m = net_revenue − ad_spend_m − ad_manager_fee_m

- **Cumulative net profit** = Σ profit_m up to m
- **Cumulative ad spend** = Σ ad_spend_m up to m

### Unit economics & milestones (from the time series)
- **ROAS (net)** = Σ net_revenue ÷ Σ ad_spend
- **Blended CAC (paid only)** = Σ ad_spend ÷ Σ new_free_paid
- **Payback month** = first m where cumulative_net_profit > 0
        """
    )


render_brand_header()

# Tabs
with st.container():
    tab_import, tab_sim, tab_est, tab_out, tab_help = st.tabs(
        [
            "Data Import",
            "Simulator",
            "Estimators",
            "Outputs & Formulas",
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
