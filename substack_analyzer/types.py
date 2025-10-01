from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class SegmentSlope:
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    slope_per_month: float


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
    starting_free_subscribers: int = 0
    starting_premium_subscribers: int = 0

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
