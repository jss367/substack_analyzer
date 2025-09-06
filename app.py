from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from substack_analyzer.model import AdSpendSchedule, SimulationInputs, simulate_growth

st.set_page_config(page_title="Substack Ads ROI Simulator", layout="wide")


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def sidebar_inputs() -> SimulationInputs:
    st.sidebar.header("Assumptions")

    with st.sidebar.expander("Starting point", expanded=True):
        start_free = st.number_input("Starting free subscribers", min_value=0, value=1000, step=10)
        start_premium = st.number_input("Starting premium subscribers", min_value=0, value=30, step=1)

    with st.sidebar.expander("Horizon", expanded=False):
        horizon = st.slider("Months to simulate", min_value=12, max_value=120, value=60, step=6)

    with st.sidebar.expander("Growth & churn", expanded=True):
        organic_growth = st.number_input(
            "Organic monthly growth (free)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%0.3f"
        )
        churn_free = st.number_input(
            "Monthly churn (free)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%0.3f"
        )
        churn_prem = st.number_input(
            "Monthly churn (premium)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%0.3f"
        )

    with st.sidebar.expander("Conversions", expanded=True):
        conv_new = st.number_input(
            "New-subscriber premium conversion", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%0.3f"
        )
        conv_ongoing = st.number_input(
            "Ongoing premium conversion of existing free",
            min_value=0.0,
            max_value=1.0,
            value=0.0003,
            step=0.0001,
            format="%0.4f",
        )

    with st.sidebar.expander("Acquisition", expanded=True):
        spend_mode = st.selectbox("Ad spend schedule", ["Two-stage (Years 1-2 / 3-5)", "Constant"], index=0)
        if spend_mode.startswith("Two-stage"):
            stage1 = st.number_input("Monthly ad spend (years 1-2)", min_value=0.0, value=3000.0, step=100.0)
            stage2 = st.number_input("Monthly ad spend (years 3-5)", min_value=0.0, value=1000.0, step=100.0)
            ad_schedule = AdSpendSchedule.two_stage(stage1, stage2)
        else:
            const_spend = st.number_input("Monthly ad spend (constant)", min_value=0.0, value=3000.0, step=100.0)
            ad_schedule = AdSpendSchedule.constant(const_spend)

        cac = st.number_input("Cost per new free subscriber (CAC)", min_value=0.01, value=2.0, step=0.1)
        ad_manager_fee = st.number_input("Ad manager monthly fee", min_value=0.0, value=1500.0, step=50.0)

    with st.sidebar.expander("Pricing & fees", expanded=True):
        price_monthly = st.number_input("Premium monthly price (gross)", min_value=0.0, value=10.0, step=1.0)
        price_annual = st.number_input("Premium annual price (gross)", min_value=0.0, value=70.0, step=5.0)
        substack_pct = st.number_input(
            "Substack fee %", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%0.2f"
        )
        stripe_pct = st.number_input(
            "Stripe % (billing + card)", min_value=0.0, max_value=1.0, value=0.036, step=0.001, format="%0.3f"
        )
        stripe_flat = st.number_input("Stripe flat per transaction", min_value=0.0, value=0.30, step=0.05)
        annual_share = st.slider("Share of premium on annual plans", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

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
- Use Subscribers over time (or export). Compute organic new = total new free − new free from ads. Monthly organic rate ≈ organic new ÷ (months × average free).

2) Monthly churn (free and premium)
- Use unsubscribes + list cleaning totals. Monthly churn ≈ churned ÷ (months × average cohort size).

3) CAC (cost per new free)
- CAC = ad spend ÷ new free from ads in the same period. Use your ad manager or Substack source tags.

4) New-subscriber premium conversion
- In a recent month, take new premium that arrived from brand‑new free signups ÷ number of new free that month.

5) Ongoing premium conversion of existing free
- Premium upgrades not tied to first‑month signups ÷ (months × average free).

6) Pricing and fees
- Defaults: Substack 10%, Stripe 3.6% + $0.30. Adjust to your setup.

7) Ad spend schedule
- Two-stage lets you set higher months 0–23 and lower 24–59; Constant uses one number.

Use the Estimators tab to compute these from a few copy/paste numbers, then plug them into the Simulator in the sidebar.
        """
    )


st.title("Substack Ads ROI Simulator")

tab_sim, tab_est, tab_help = st.tabs(["Simulator", "Estimators", "Help"])

with tab_sim:
    inputs = sidebar_inputs()
    result = simulate_growth(inputs)
    df = result.monthly
    render_kpis(df)
    with st.expander("Monthly details", expanded=False):
        st.dataframe(df, width="stretch")
    render_charts(df)
    st.caption(
        "MVP model: instant conversion of a share of new free subs, small ongoing conversion of existing free base, and simple net revenue after Substack + Stripe fees."
    )

with tab_est:
    render_estimators()

with tab_help:
    render_help()
