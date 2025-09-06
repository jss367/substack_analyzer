from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdSpendSchedule:
    """Defines monthly ad spend as a function of month index (starting at 0)."""

    get_spend_for_month: Callable[[int], float]

    @staticmethod
    def constant(monthly_spend: float) -> "AdSpendSchedule":
        return AdSpendSchedule(lambda _m: float(monthly_spend))

    @staticmethod
    def two_stage(years_1_to_2: float, years_3_to_5: float) -> "AdSpendSchedule":
        def _spend(month_index: int) -> float:
            # Months are 0-indexed. Years 1-2: months 0..23, Years 3-5: months 24..59
            if month_index < 24:
                return float(years_1_to_2)
            return float(years_3_to_5)

        return AdSpendSchedule(_spend)


@dataclass(frozen=True)
class SimulationInputs:
    # Starting state
    starting_free_subscribers: int = 1000
    starting_premium_subscribers: int = 20

    # Horizon
    horizon_months: int = 60

    # Growth and churn
    organic_monthly_growth_rate: float = 0.01  # 1%
    monthly_churn_rate_free: float = 0.01  # 1%
    monthly_churn_rate_premium: float = 0.01  # 1%

    # Conversions
    new_subscriber_premium_conv_rate: float = 0.02  # 2% of new free subs
    ongoing_premium_conv_rate: float = 0.0003  # 0.03% of existing free base per month

    # Acquisition
    cost_per_new_free_subscriber: float = 2.00
    ad_spend_schedule: AdSpendSchedule = AdSpendSchedule.two_stage(3000.0, 1000.0)
    ad_manager_monthly_fee: float = 1500.0

    # Pricing & fees (monthly plan)
    premium_monthly_price_gross: float = 10.0
    substack_fee_pct: float = 0.10  # 10%
    stripe_fee_pct: float = 0.036  # 3.6%
    stripe_flat_fee: float = 0.30  # $0.30 per transaction

    # Optional: annual pricing share (0..1). If >0, some users pay annually.
    annual_share: float = 0.0
    premium_annual_price_gross: float = 70.0


@dataclass(frozen=True)
class SimulationResult:
    monthly: pd.DataFrame

    @property
    def summary(self) -> dict[str, float]:
        last = self.monthly.iloc[-1]
        return {
            "ending_free": float(last["free_subscribers"]),
            "ending_premium": float(last["premium_subscribers"]),
            "ending_total": float(last["total_subscribers"]),
            "cumulative_net_profit": float(last["cumulative_net_profit"]),
            "cumulative_ad_spend": float(last["cumulative_ad_spend"]),
            "peak_mrr_net": float(self.monthly["mrr_net"].max()),
        }


def _net_monthly_revenue_per_premium(input_params: SimulationInputs) -> float:
    gross = input_params.premium_monthly_price_gross
    net = gross * (1.0 - input_params.substack_fee_pct - input_params.stripe_fee_pct) - input_params.stripe_flat_fee
    return max(net, 0.0)


def _net_annual_revenue_per_premium(input_params: SimulationInputs) -> float:
    gross = input_params.premium_annual_price_gross
    net = gross * (1.0 - input_params.substack_fee_pct - input_params.stripe_fee_pct) - input_params.stripe_flat_fee
    return max(net, 0.0)


def simulate_growth(input_params: SimulationInputs) -> SimulationResult:
    """Run a monthly simulation of subscriber and revenue dynamics.

    Model notes (MVP):
    - Free base grows via organic rate and paid acquisition (ad spend / CAC)
    - Churn applied to beginning-of-month balances
    - Premium conversions:
            - A share of this month's new free subscribers convert immediately
              (new_subscriber_premium_conv_rate)
            - A small ongoing share of existing free base converts monthly
              (ongoing_premium_conv_rate)
    - Premium churn uses monthly_churn_rate_premium
    - Revenue assumes monthly plan for all premium users unless annual_share > 0
    - Net revenue computes Substack and Stripe fees (percentage + flat)
    - Profit = net revenue - ad spend - ad manager fee
    """

    months = np.arange(input_params.horizon_months)

    columns = [
        "month",
        "free_subscribers",
        "premium_subscribers",
        "total_subscribers",
        "new_free_organic",
        "new_free_paid",
        "free_churned",
        "premium_converted_from_new",
        "premium_converted_from_existing",
        "premium_churned",
        "ad_spend",
        "ad_manager_fee",
        "mrr_gross",
        "mrr_net",
        "net_revenue",
        "profit",
        "cumulative_ad_spend",
        "cumulative_net_profit",
    ]
    data: list[list[float]] = []

    free_subs = float(input_params.starting_free_subscribers)
    premium_subs = float(input_params.starting_premium_subscribers)

    net_monthly = _net_monthly_revenue_per_premium(input_params)
    net_annual = _net_annual_revenue_per_premium(input_params)

    cumulative_ad_spend = 0.0
    cumulative_net_profit = 0.0

    for m in months:
        # Beginning-of-month churn
        free_churned = free_subs * input_params.monthly_churn_rate_free
        premium_churned = premium_subs * input_params.monthly_churn_rate_premium

        free_subs -= free_churned
        premium_subs -= premium_churned

        # Organic growth
        new_free_organic = free_subs * input_params.organic_monthly_growth_rate

        # Paid acquisition
        ad_spend = float(input_params.ad_spend_schedule.get_spend_for_month(m))
        paid_new = (
            0.0
            if input_params.cost_per_new_free_subscriber <= 0
            else ad_spend / input_params.cost_per_new_free_subscriber
        )

        # Add new free
        new_free_total = new_free_organic + paid_new
        free_subs += new_free_total

        # Conversions to premium
        convert_from_new = new_free_total * input_params.new_subscriber_premium_conv_rate
        convert_from_existing = max(free_subs - new_free_total, 0.0) * input_params.ongoing_premium_conv_rate

        # Apply conversions: move from free to premium
        total_convert = convert_from_new + convert_from_existing
        free_subs = max(free_subs - total_convert, 0.0)
        premium_subs += total_convert

        # Revenue
        # Split premium base into monthly vs annual cohorts
        monthly_premium = premium_subs * (1.0 - input_params.annual_share)
        annual_premium = premium_subs * input_params.annual_share

        mrr_gross = monthly_premium * input_params.premium_monthly_price_gross
        mrr_net = monthly_premium * net_monthly

        # Annual revenue recognized this month (simplified: evenly amortized)
        annual_revenue_net_month = (annual_premium * net_annual) / 12.0

        net_revenue = mrr_net + annual_revenue_net_month

        ad_manager_fee = input_params.ad_manager_monthly_fee if ad_spend > 0 else 0.0
        profit = net_revenue - ad_spend - ad_manager_fee

        cumulative_ad_spend += ad_spend
        cumulative_net_profit += profit

        total_subscribers = free_subs + premium_subs

        data.append(
            [
                float(m + 1),
                free_subs,
                premium_subs,
                total_subscribers,
                new_free_organic,
                paid_new,
                free_churned,
                convert_from_new,
                convert_from_existing,
                premium_churned,
                ad_spend,
                ad_manager_fee,
                mrr_gross,
                mrr_net,
                net_revenue,
                profit,
                cumulative_ad_spend,
                cumulative_net_profit,
            ]
        )

    monthly_df = pd.DataFrame(data, columns=columns)
    return SimulationResult(monthly=monthly_df)
