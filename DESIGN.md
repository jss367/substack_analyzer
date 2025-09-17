## Substack Growth Modeling - System Design

This document describes the staged flow for importing Substack exports, estimating key rates, fitting a piecewise-logistic model with events, and forecasting scenarios. For each stage, we list inputs, processing, and outputs (including artifacts saved in session state).

### Stage 1 — File Upload and Parsing (Data Import)

- **Inputs**:
  - All subscribers export (CSV/XLSX): date, total subscribers.
  - Paid subscribers export (CSV/XLSX): date, paid subscribers.
  - User choice: header or not; column indices.
- **Processing**:
  - Read CSV/XLSX; coerce dates; drop invalid; sort ascending.
  - Resample to month-end; take last observation per month.
  - Align series; derive `Free = Total - Paid` when both present.
- **Outputs**:
  - Session: `import_total` (pd.Series, month-end), `import_paid` (pd.Series), optional `Free` derived in charts.
  - Plots: Total/Free and Paid (dual-axis option), monthly deltas.

### Stage 2 — Event Capture and Change-Point Detection

- **Inputs**:
  - User-labeled events in editor: `date`, `type` (Ad spend, Shout-out, Other), `notes`, `cost`.
  - Optional: detected change points from imported series (Total preferred).
- **Processing**:
  - Change-point detection on monthly deltas/acceleration (returns indices → dates).
  - One-click: seed Events table with detected dates (type=Change, cost=0).
- **Outputs**:
  - Session: `events_df` (pd.DataFrame of events), `max_changes_detect` preference.
  - UI: event markers overlaid on charts; badges listing detected dates.

### Stage 3 — Quick Estimation (Heuristics)

- **Inputs**:
  - Imported series (Total and/or Paid), window (last N months), net-only toggle.
- **Processing**:
  - Compute trailing medians for net growth and conversion proxy when both series are present:
    - `start_free`, `start_premium` from latest values (or derived).
    - `organic_growth` ≈ trailing median of net growth rates (Total or Free).
    - `conv_ongoing` proxy from Paid/Free dynamics (if both present).
  - Churn defaults retained unless user overrides.
- **Outputs**:
  - Metrics displayed and `Apply estimates to Simulator` button to populate sidebar state.
  - Session set on apply: `start_free`, `start_premium`, `organic_growth`, `conv_ongoing`, `churn_free/prem` (0 if net-only), ad spend defaults zero, `horizon_months` ≥ 24.

### Stage 4 — Model Fitting (Piecewise-Logistic with Events)

- **Inputs**:
  - Target series: `Total` (preferred) or `Free` if Total unavailable.
  - Breakpoints: indices from Stage 2 change-point detection (optional).
  - Events table from Stage 2.
- **Processing**:
  - Fit ΔS*t = r_seg(t)·S*{t−1}(1 − S\_{t−1}/K) + γ_pulse·pulse_t + γ_step·step_t + ε_t
    - Grid search over K (carrying capacity), OLS for per-segment r and event γ’s.
    - Pulse: month-of spikes; Step: level lift from event date forward; cost can weight pulse.
  - Reconstruct fitted series and compute R² on deltas.
- **Outputs**:
  - Overlaid chart: Actual vs Fitted; optional forecast extension.
  - Metrics: K, segment growth rates r, R² on ΔS.
  - Session: `pwlog_fit` (parameters and fitted series).

### Stage 5 — Forecasting (Short Horizon)

- **Inputs**:
  - From Stage 4: last fitted value, K, last-segment r, γ_step (optional), horizon months.
- **Processing**:
  - Deterministic forward simulation using last segment parameters; apply step level; pulse only once if specified.
- **Outputs**:
  - Forecast curve appended to overlay; CSV-ready series if exported later.

### Stage 6 — Simulator (Cohort & Finance Model)

- **Inputs**:
  - Sidebar parameters (from user or Stage 3 apply):
    - Starting free/premium, horizon months.
    - Growth and churn: organic monthly growth, monthly churn (free/premium).
    - Conversions: new-subscriber conversion, ongoing conversion of existing free.
    - Acquisition: CAC, ad spend schedule (constant or two-stage), ad manager fee.
    - Pricing & fees: monthly and annual prices, Substack %, Stripe % and flat, annual_share.
- **Processing**:
  - Monthly dynamics for free/premium with churn, conversions, paid adds from ad spend/CAC.
  - Net revenue per premium (fees), amortized annual revenue, profit, cumulative metrics.
- **Outputs**:
  - DataFrame of monthly metrics, KPI tiles, subscriber/revenue charts, ROAS, CAC, payback month.
  - Session: `sim_df` for downstream comparisons.

### Stage 7 — Outputs & Documentation

- **Inputs**: None beyond prior stages.
- **Processing**: Display formal formulas and definitions used by the simulator.
- **Outputs**: Reference for interpretation and sanity checks.

---

## Data Contracts (per stage)

### Imported Series

- `import_total`: pd.Series, DatetimeIndex (month-end), dtype float/int.
- `import_paid`: pd.Series, DatetimeIndex (month-end), dtype float/int.

### Events Table (`events_df`)

- Columns: `date` (date), `type` (str: Ad spend, Shout-out, Other, Change), `notes` (str), `cost` (float).
- Dates normalized to month-end.

### Calibration Fit (`pwlog_fit`)

- `carrying_capacity` (float K), `segment_growth_rates` (list of float r), `breakpoints` (list[int]),
  `gamma_pulse` (float), `gamma_step` (float), `fitted_series` (pd.Series), `r2_on_deltas` (float).

### Simulator Inputs (`SimulationInputs`)

- See `substack_analyzer/model.py`. Populated via sidebar or Stage 3 apply.

---

## Dependencies and Extensibility

- Current: piecewise-logistic fit on monthly series, deterministic forecast.
- Next milestones (optional):
  - Adds vs churn decomposition (NegBin for adds, Binomial/hazard for churn).
  - Adstock and diminishing returns for ad spend.
  - Bayesian estimation with uncertainty (PyMC/Stan) and time-series CV.
  - Scenario sets and side-by-side comparisons.
