# Substack Growth Modeling — Simplified System Design

This is a minimal version of the growth model: saturation, ads with diminishing returns, and event pulses. No separate adds/churn layers, no seasonality, no change-points.

---

## Core Equation

We model daily change in subscribers as:

[
\Delta S_t = r \cdot S_{t-1}\Bigl(1 - \tfrac{S_{t-1}}{K}\Bigr) + g(a_t) + \gamma \cdot \text{pulse}_t + \varepsilon_t
]

Where:

- **(S_t)**: active subscribers at day _t_ (from Substack export).
- **(r)**: baseline growth rate (to estimate).
- **(K)**: saturation level / carrying capacity (to estimate).
- **(a_t)**: adstocked spend.
- **(g(a_t))**: diminishing returns transform of adstock.
- **(\gamma)**: pulse event effect size.
- **(\varepsilon_t)**: noise.

---

## Stage 1 — Data Input

**Inputs**

- Substack export of active subscribers (CSV/XLSX).
- Optional: ad spend (daily), shoutout/press events (dates).

**Processing**

- Parse dates, sort ascending.
- Normalize to daily frequency; carry forward if sparse.
- Build `observations_df`: `date`, `active_total`, `is_imputed`.

**Outputs**

- Clean daily subscriber time series.

---

## Stage 2 — Events & Ads

**Adstock**

- Compute: (a*t = x_t + \lambda a*{t-1}).
- Fix (\lambda) to represent a 3-day half-life (≈0.79), or estimate later.

**Ad response**

- Use log form: (g(a_t) = \beta \log(1 + a_t/\theta)).

**Pulse events**

- Encode shoutouts/press as short pulses with fixed half-life (e.g., 3 days).

**Outputs**

- `features_df`: `date`, `adstock`, `ad_effect`, `pulse`.

---

## Stage 3 — Model Fitting

**Model**

- Observation: (\Delta S_t) modeled with logistic growth + ads + pulses.
- Parameters to estimate: (r, K, \beta, \theta, \gamma, \sigma).

**Inference options**

- **Quick fit**: grid search over (K, \theta, \lambda); OLS for (r, \beta, \gamma).
- **Bayesian fit**: priors on parameters; NUTS sampling in PyMC/Stan.

**Outputs**

- Fitted parameters.
- Posterior predictive bands (if Bayesian).

---

## Stage 4 — Forecasting & Scenarios

**Inputs**

- Fitted parameters.
- Scenario levers: future ad spend, future pulses.

**Processing**

- Simulate (S_t) forward under scenarios.
- Clip by (K).

**Outputs**

- Forecasted subscribers with intervals.
- Scenario comparisons.

---

## Stage 5 — Implementation Notes

- Dependencies: PyMC/Stan, ArviZ, pandas.
- Start with fixed (\lambda), grid search (K), log-form ads.
- Weekly aggregation may help runtime.

---

## Data Contracts (Simplified)

### `observations_df`

- `date`
- `active_total`
- `is_imputed`

### `features_df`

- `date`
- `adstock`
- `ad_effect`
- `pulse`

### `forecast_df`

- `date`
- `active_total` median and intervals

---

This simplified system drops churn modeling, paid/free split, step events, and seasonality. It keeps only the essentials: logistic growth with saturation, ads, and pulses.
