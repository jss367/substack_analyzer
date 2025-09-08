from __future__ import annotations

import math
from contextlib import suppress
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from substack_analyzer.model import AdSpendSchedule, SimulationInputs, simulate_growth

st.set_page_config(page_title="Substack Ads ROI Simulator", layout="wide")


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _get_state(key: str, default):
    return st.session_state.get(key, default)


def sidebar_inputs() -> SimulationInputs:
    st.sidebar.header("Assumptions")

    with st.sidebar.expander("Starting point", expanded=True):
        start_free = st.number_input(
            "Starting free subscribers",
            min_value=0,
            value=int(_get_state("start_free", 0)),
            step=10,
            key="start_free",
        )
        start_premium = st.number_input(
            "Starting premium subscribers",
            min_value=0,
            value=int(_get_state("start_premium", 0)),
            step=1,
            key="start_premium",
        )

    with st.sidebar.expander("Horizon", expanded=False):
        horizon = st.slider(
            "Months to simulate",
            min_value=12,
            max_value=120,
            value=int(_get_state("horizon_months", 60)),
            step=6,
            key="horizon_months",
        )

    with st.sidebar.expander("Growth & churn", expanded=True):
        organic_growth = st.number_input(
            "Organic monthly growth (free)",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("organic_growth", 0.01)),
            step=0.001,
            format="%0.3f",
            key="organic_growth",
        )
        churn_free = st.number_input(
            "Monthly churn (free)",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("churn_free", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_free",
        )
        churn_prem = st.number_input(
            "Monthly churn (premium)",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("churn_prem", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_prem",
        )

    with st.sidebar.expander("Conversions", expanded=True):
        conv_new = st.number_input(
            "New-subscriber premium conversion",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("conv_new", 0.0)),
            step=0.001,
            format="%0.3f",
            key="conv_new",
        )
        conv_ongoing = st.number_input(
            "Ongoing premium conversion of existing free",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("conv_ongoing", 0.0)),
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
            stage1 = st.number_input(
                "Monthly ad spend (years 1-2)",
                min_value=0.0,
                value=float(_get_state("ad_stage1", 0.0)),
                step=50.0,
                key="ad_stage1",
            )
            stage2 = st.number_input(
                "Monthly ad spend (years 3-5)",
                min_value=0.0,
                value=float(_get_state("ad_stage2", 0.0)),
                step=50.0,
                key="ad_stage2",
            )
            ad_schedule = AdSpendSchedule.two_stage(stage1, stage2)
            st.session_state["spend_mode_index"] = 0
        else:
            const_spend = st.number_input(
                "Monthly ad spend (constant)",
                min_value=0.0,
                value=float(_get_state("ad_const", 0.0)),
                step=50.0,
                key="ad_const",
            )
            ad_schedule = AdSpendSchedule.constant(const_spend)
            st.session_state["spend_mode_index"] = 1

        cac = st.number_input(
            "Cost per new free subscriber (CAC)",
            min_value=0.01,
            value=float(_get_state("cac", 2.0)),
            step=0.1,
            key="cac",
        )
        ad_manager_fee = st.number_input(
            "Ad manager monthly fee",
            min_value=0.0,
            value=float(_get_state("ad_manager_fee", 0.0)),
            step=50.0,
            key="ad_manager_fee",
        )

    with st.sidebar.expander("Pricing & fees", expanded=True):
        price_monthly = st.number_input(
            "Premium monthly price (gross)",
            min_value=0.0,
            value=float(_get_state("price_monthly", 10.0)),
            step=1.0,
            key="price_monthly",
        )
        price_annual = st.number_input(
            "Premium annual price (gross)",
            min_value=0.0,
            value=float(_get_state("price_annual", 70.0)),
            step=5.0,
            key="price_annual",
        )
        substack_pct = st.number_input(
            "Substack fee %",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("substack_pct", 0.10)),
            step=0.01,
            format="%0.2f",
            key="substack_pct",
        )
        stripe_pct = st.number_input(
            "Stripe % (billing + card)",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("stripe_pct", 0.036)),
            step=0.001,
            format="%0.3f",
            key="stripe_pct",
        )
        stripe_flat = st.number_input(
            "Stripe flat per transaction",
            min_value=0.0,
            value=float(_get_state("stripe_flat", 0.30)),
            step=0.05,
            key="stripe_flat",
        )
        annual_share = st.slider(
            "Share of premium on annual plans",
            min_value=0.0,
            max_value=1.0,
            value=float(_get_state("annual_share", 0.0)),
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
        all_file = st.file_uploader("All subscribers file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="all_file")
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
        paid_file = st.file_uploader("Paid subscribers file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="paid_file")
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
        try:
            all_series = None
            paid_series = None
            if all_file is not None:
                all_series = _read_series(all_file, all_has_header, all_date_sel, all_count_sel)
            if paid_file is not None:
                paid_series = _read_series(paid_file, paid_has_header, paid_date_sel, paid_count_sel)

            # Preview charts
            plot_df = pd.DataFrame()
            if all_series is not None and not all_series.empty:
                plot_df["All"] = all_series
            if paid_series is not None and not paid_series.empty:
                plot_df["Paid"] = paid_series
            if not plot_df.empty:
                if "All" in plot_df.columns and "Paid" in plot_df.columns:
                    plot_df["Free"] = plot_df["All"].astype(float) - plot_df["Paid"].astype(float)
                st.subheader("Imported series")
                st.line_chart(plot_df)
                # Deltas
                deltas = plot_df.diff()
                st.subheader("Monthly change (delta)")
                st.bar_chart(deltas)
                # Tail-only view with window slider placed here
                window = st.slider("Estimation window (last N months)", 3, 12, window, 1, key="est_window")
                st.caption("This window recomputes trailing medians for the estimates and the tail chart below.")
                st.subheader(f"Last {window} months (tail)")
                st.line_chart(plot_df.tail(window))

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


st.title("Substack Ads ROI Simulator")

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
