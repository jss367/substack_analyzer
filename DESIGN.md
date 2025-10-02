# Substack Growth Modeling — System Design (State‑Space/Bayesian)

[IN PROGRESS]

This updates the original design to a structural, _adds–churn_ model with saturation, interventions (shoutouts/ads), and explicit uncertainty via a Bayesian state‑space approach. It supports realistic phases (slow start → spikes → paid acquisition → saturation), scenario planning, and calibrated forecasts.

---

## What’s Different vs Original

- **Adds vs. churn** modeled separately (better attribution + forecasting) instead of net-only deltas.
- **Saturation** via time‑varying carrying capacity $K_t$ and growth rate $r_t$ with **change‑points** or random walks, replacing fixed piecewise logistic.
- **Events** split into **pulse** (short‑lived bursts) and **step** (ongoing lift), plus **adstock with diminishing returns** for ads.
- **Bayesian estimation** with predictive intervals and time‑series cross‑validation, not just OLS on deltas.
- **Scenario engine** operates on posterior draws (uncertainty propagates), not just deterministic extrapolation.

---

## Stage 1 — File Upload & Parsing (Daily Preferred)

**Inputs**

- Substack exports:

  - _All subscribers_ (CSV/XLSX): date, total subscribers (cumulative active).
  - _Paid subscribers_ (CSV/XLSX): date, paid active subscribers.
  - _(Optional, if available)_ subscriber event logs: signups/cancels by date.

- _(Optional)_ External tables: ad spend (by channel), shoutouts (date, size), posts sent, social metrics, Google Trends.
- User options: header flags, column indices, time zone.

**Processing**

- Read and coerce dates; drop invalid; sort ascending.
- Normalize to **daily index** (preferred). If input is sparser (weekly/monthly), resample to daily using last observation carry‑forward; tag imputed days.
- Derive `Free = Total − Paid` (clip at ≥0).
- Build a canonical **observations table** with columns below (see Data Contracts).

**Outputs**

- `observations_df` (daily): `date`, `active_total`, `active_paid`, `active_free`, with flags for imputation.
- Quick plots: active series (free/paid/total), daily/weekly net changes.

---

## Stage 2 — Event & Covariate Capture

**Inputs**

- User‑labeled events: `date`, `type` (Ad, Shoutout, Press, Other), `notes`, `size` (audience/spend), `channel`, `cost`.
- _(Optional)_ Auto‑detected change‑points suggest candidate dates.
- External covariates: ad spend/impressions by channel, posts sent, social mentions, search interest.

**Processing**

- Normalize all events/covariates to **daily**.
- Compute **adstock** per channel: $a_t = x_t + \lambda a_{t-1}$, $\lambda\in[0,1)$ learned.
- Define **diminishing returns** transform for ads:

  - Log: $g(a_t) = \beta\, \log(1 + a_t/\theta)$

- Encode **shoutouts** as:

  - **Pulse** features (short‑lived): Dirac at date convolved with short decay kernel (1–7 days).
  - **Step** features (level lift): indicator $\mathbf{1}(t\ge d)$.

- Pulse kernel parameterization: specify half‑life $h$ so that the decay factor $\lambda = 0.5^{1/h}$. Start with a fixed $h$ (e.g., 3 days) and later allow learning $h$ hierarchically across shoutouts.
- Step effects: estimate per‑event step sizes with partial pooling (hierarchical prior) to stabilize small events.

**Outputs**

- `events_df` (daily): pulse/step encodings with metadata.
- `covariates_df` (daily): ad spend by channel, posts, social, search.
- `features_df` (daily): adstocked series, transformed ad response, content cadence.

---

## Stage 3 — Adds/Churn Preparation

**Inputs**

- `observations_df` (+ optionally subscriber event logs), `features_df`.

**Processing**

- **Path A (preferred)**: If signup/cancel logs exist, aggregate to daily `gross_adds_*` and `cancels_*` (free/paid).
- **Path B (totals only)**: Use a **Kalman smoother** (structural time‑series) to decompose net changes into latent `gross_adds` and `cancels`, with:

  - Smoothness priors on adds.
  - A **tenure‑based hazard prior** for churn (declining with age), modulated by content cadence. Enforce monotonicity of the age‑hazard spline or use a decreasing basis to avoid pathological shapes. When only totals exist, anchor paid churn with observed paid data and share information to free churn via a prior relationship.

**Outputs**

- `adds_df` (daily): `gross_adds_free`, `gross_adds_paid` (with uncertainty if latent).
- `churn_df` (daily): `cancels_free`, `cancels_paid` (with uncertainty if latent).

---

## Stage 4 — Model Fitting (Bayesian State‑Space with Saturation & Interventions)

**Model sketch (free/total shown; paid can be layered)**

- State (actives): $S_t$.
- **Growth dynamic with saturation**:
  $(\Delta S_t = r_t\, S_{t-1}\bigl(1 - S_{t-1}/K_t\bigr) + u_t - \text{churn}_t + \varepsilon_t)$.
- **Time‑varying parameters**:

  - $r_t$: piecewise‑constant with **unknown change‑points** (0–3) _or_ random walk.
  - $K_t$: random walk on log‑scale; weakly‑informative upper prior (market size proxy).

- **Exogenous input**: $u_t = \gamma_\text{pulse}\,\text{pulse}_t + \gamma_\text{step}\,\text{step}_t + f_\text{ads}(a_t) + \beta' x_t$.

  - **Adstock**: $a_t = x_t + \lambda a_{t-1}$, $\lambda\in[0,1)$ per channel or shared.
  - **Diminishing returns**: $f_\text{ads}$ via log form above.
  - **Other covariates**: content cadence, social/search.

- **Likelihoods** (choose by data availability):

  - If adds/cancels available:

    - `gross_adds_t` $\sim$ NegBin($\mu_{\text{adds},t}$, $\phi_{\text{adds}}$).
    - `cancels_t` $\sim$ Binomial$(S_{t-1}, \pi_t)$ with $\text{logit}(\pi_t) = \alpha_0 + \alpha' z_t$.

  - If only totals: observation equation $S^{\text{obs}}_t \sim \mathcal{N}(S_t, \sigma_{\text{obs}})$ plus robust noise for outliers.

- **Priors**: weakly informative for $r_t$, $K_t$; half‑normal for positive scales; sparsity prior on # of change‑points; $\lambda\in[0,1)$ uniform or Beta.
- **Inference**: PyMC/Stan (NUTS). Store posterior draws for all states/parameters.

**Identifiability guardrails**

- Distinguish market size drifts ($K_t$) from permanent level shifts (step events):
  - Use a small random‑walk variance on $(\log K_t)$ and a spike–slab (or strong sparsity) prior on step coefficients.
  - Penalize frequent change‑points on $(r_t)$ to avoid overfitting pulses as trend breaks.
- Ad response: start with a log form $(f_\text{ads}(a_t) = \beta\, \log(1 + a_t/\theta))$; fix $(\eta=1)$ initially for identifiability. Place $(\lambda\sim\text{Beta}(2,2))$ and share $(\theta)$ across sparse channels with hierarchical pooling.
- Constrain churn hazards to $[0,1]$ and enforce non‑negativity on adds; clip forecasts at capacity draws (see Scenario Constraints).

**Inputs**

- `observations_df`, `adds_df`, `churn_df`, `features_df`, optional change‑point suggestions.

**Outputs**

- `bayes_fit` object: posterior samples (`S_t`, `K_t`, `r_t`, change‑points, event coefficients, adstock λ, ad elasticity, churn params, dispersion), posterior predictive, log‑likelihood.
- Fit plots: Actual vs posterior median with 50/80/95% bands; contribution (shap‑like) breakdown for u_t components.

---

## Stage 5 — Diagnostics & Validation

**Processing**

- **Rolling‑origin time‑series CV** (e.g., 4–8 weekly folds). Refit/train on past, forecast next horizon.
- **Metrics**: MAPE on gross adds, RMSE on actives, coverage for 50/80/95% intervals, CRPS.
- **Checks**: Prior and posterior predictive checks; PSIS‑LOO or WAIC for model comparison; target interval coverage (e.g., 50%/80%/95%) and well‑behaved residual autocorrelation.
- **Event attribution checks**: estimated pulse half‑life; step permanence.
- **Elasticity checks**: slope of $f_\text{ads}$ at current spend; saturation level proximity ($S_t/K_t$).
- Residual diagnostics: autocorrelation.

**Outputs**

- `validation_report`: fold metrics + calibration plots.
- `attribution_summary`: lifts by event/ad channel with uncertainty.

---

## Stage 6 — Forecasting & Scenario Planning

**Inputs**

- `bayes_fit` posterior draws; scenario knobs (future events, ad schedules, cadence changes, market expansion).

**Processing**

- **Posterior simulation**: for each draw, simulate forward `S_t`, `adds`, `cancels` under scenario.
- Scenario levers:

  - **Shoutouts**: date, pulse size, step proportion, assumed decay.
  - **Ads**: spend by channel; channel‑specific adstock/response.
  - **Cadence**: posts/week; impact via $r_t$ or an explicit covariate.
  - **Market expansion**: exogenous shift in $K_t$ level.
  - **Pricing/Paywall**: feeds churn and free→paid conversion.

**Outputs**

- `forecast_df`: median and intervals (50/80/95%) for actives, adds, cancels.
- Scenario comparisons: side‑by‑side KPIs; fan charts.

**Constraints**

- Clip simulated actives by each draw’s $(K_t)$ path; prevent negative adds/churn.
- Report the share of draws approaching capacity (e.g., $(S_t/K_t > 0.9)$) to flag saturation risk.
- Bound ad elasticities to reasonable ranges in scenarios to avoid implausible lifts.

---

## Stage 7 — Cohort & Finance Simulator (Paid/Free)

**Inputs**

- Starting states (from `bayes_fit` last day), pricing & fees, CAC and ad schedules, conversion assumptions.

**Processing**

- **Cohort survival**: tenure‑based hazard for churn (age spline + price/cadence flags).
- **Free→Paid conversion**: function of cadence/exposure and promotions; can be hierarchical if multiple publications.
- **Revenue**: monthly & annual pricing, Substack/Stripe fees, annual amortization; ROAS, CAC, payback.

**Outputs**

- `sim_df`: monthly KPIs (subs, adds, churn, MRR/ARR, ROAS, CAC, payback month), charts and tiles.

---

## Stage 8 — Outputs & Documentation

**Processing**

- Render formulas and definitions for each module.
- Persist run configs and random seeds for reproducibility.

**Outputs**

- Markdown/HTML reference; downloadable CSVs for fitted states and forecasts; serialized `bayes_fit`.

---

## Data Contracts

### `observations_df` (daily)

- `date` (date, unique, sorted)
- `active_total` (int ≥0)
- `active_paid` (int ≥0, optional)
- `active_free` (int ≥0, derived if needed)
- `is_imputed` (bool) — resampled day indicator

### `events_df` (daily)

- `date` (date)
- `type` (str: Ad, Shoutout, Press, Other)
- `channel` (str, optional)
- `pulse_raw` (float) — raw size (e.g., est. audience)
- `pulse` (float) — convolved pulse feature
- `step` (int {0,1}) — step indicator from date onward
- `cost` (float, optional)
- `notes` (str)

### `covariates_df` (daily)

- `date`
- `ad_spend_<channel>` (float ≥0)
- `posts_sent` (int ≥0)
- `social_mentions` (int ≥0, optional)
- `search_index` (float ≥0, optional)

### `features_df` (daily)

- `date`
- `adstock_<channel>` (float ≥0)
- `ad_effect_<channel>` (float) — transformed via log
- `cadence` (float) — posts/week

### `adds_df` (daily)

- `date`
- `gross_adds_free` (int ≥0)
- `gross_adds_paid` (int ≥0, optional)
- `adds_ci_low`, `adds_ci_high` (if latent)

### `churn_df` (daily)

- `date`
- `cancels_free` (int ≥0)
- `cancels_paid` (int ≥0, optional)
- `churn_ci_low`, `churn_ci_high` (if latent)

### `bayes_fit`

- Posterior draws for: `S_t`, `K_t`, `r_t`, change‑point locations, `gamma_pulse`, `gamma_step`, `lambda_adstock`, `beta_ads`, `beta_covariates`, churn parameters, dispersion params.
- Posterior predictive distributions for observed targets.
- Log‑likelihood per time point (for PSIS‑LOO).

### `validation_report`

- Fold‑level metrics table; coverage charts; residual diagnostics.

### `forecast_df`

- `date`, median and interval columns for `active_*`, `adds_*`, `cancels_*` under each scenario id.

### `sim_df`

- Monthly aggregates with subscriber/revenue KPIs, ROAS/CAC/payback.

---

## Implementation Notes & MVP Path

- **Dependencies**: PyMC or Stan; ArviZ; optional: numpyro.
- **Computation**: Start with 1–2 change‑points, shared adstock λ across channels (simplify), NegBin for adds.
- **Runtime & caching**: Target 2–5 minutes per fit (2 chains, ~1k draws post‑warmup). Cache fits keyed by a data hash of inputs/features. If runtime exceeds budget, fall back to Quick Fit (piecewise‑logistic) and offer to enqueue Pro Fit.

**Modes**

- Quick Fit (deterministic): piecewise‑logistic with events (already implemented) for instant overlays and rough forecasts.
- Pro Fit (Bayesian): full state‑space with uncertainty and scenario simulation.

**MVP order**:

1. Stages 1–2 (ingest + events/covariates) and simple diagnostics.
2. Stage 3 Path B (latent adds/churn) + quick heuristic estimator for churn if needed.
3. Stage 4 with single‑segment $r$ + fixed $K$ grid search to sanity‑check.
4. Swap in full Bayesian with change‑points + $K_t$ drift.
5. Stage 6 scenarios (ad spend & shoutouts), then Stage 7 finance.

- **Extensibility**:

  - **Bass diffusion** alternative for adds ($p,q,K$) + adstock input.
  - **Hawkes** option for referrals (self‑exciting adds) bounded by $K_t$.
  - **Hierarchical** pooling across multiple publications.

---

## UI/UX Notes

- Inline event editor with “seed from detected change‑points.”
- Toggle: _Totals‑only_ vs _Adds/Churn available_; show uncertainty bands when latent.
- Quick Fit vs Pro Fit toggle with ETA/progress; warn about runtime; cache and allow restoring last fit.
- Parameter table: medians and 80/95% CrIs for $(r_t)$ levels, $(K_t)$, adstock $(\lambda)$, $(\theta)$, $(\beta)$, and churn hazards.
- Contribution chart that decomposes forecasted adds into organic (S‑curve), shoutouts (pulse/step), and ads by channel.
- Scenario panel with reusable templates (e.g., “$2k/mo Meta spend,” “one large shoutout in Q2”).
- Export buttons for CSV (fitted & forecast), and serialized `bayes_fit` plus compact posterior summaries (JSON).
