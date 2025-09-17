# substack_analyzer

This is available at https://substackanalyzer.streamlit.app/

## Quick start

1. Create a virtual environment (Python 3.12):

```bash
cd /Users/julius/git/substack_analyzer
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## What this does

- Simulates monthly subscriber growth for free and premium cohorts
- Combines organic growth, paid acquisition (ad spend / CAC), and churn
- Converts a share of new free subs to premium immediately, plus a small ongoing conversion of existing free
- Computes net revenue after Substack (10%) and Stripe (3.6% + $0.30) fees
- Tracks profit = net revenue − ad spend − ad manager fee
- Visualizes KPIs, subscribers over time, spend vs revenue

## Mapping Substack stats to inputs

- Organic monthly growth (free):
  - Use Substack's "Subscribers over time" or exports to estimate average monthly net new free subs excluding paid acquisition. Divide by the free base to get a rate.
- Cost per new free subscriber (CAC):
  - From your ad platform data or Substack "Where subscribers came from" exports when tagged. CAC = ad spend / new free subs attributed to ads.
- New-subscriber premium conversion:
  - Estimate the share of newly acquired free subs who convert to premium within the first month. If unknown, start with 1–3%.
- Ongoing premium conversion of existing free:
  - Small monthly rate applied to the existing free base. If unknown, start with 0.02–0.05% (0.0002–0.0005).
- Churn (free and premium):
  - Use list cleaning + unsubscribes divided by cohort size monthly. If you only have paid churn, apply that to premium and set free churn around 0.5–2%.
- Pricing and fees:
  - Substack fee 10%, Stripe 3.6% + $0.30 are defaults; update as needed.
- Ad spend schedule:
  - Two-stage lets you specify a higher budget in years 1–2 and lower in years 3–5; constant uses a flat monthly spend.

## Notes and limitations (MVP)

- Conversions and churn are applied at a monthly granularity with simplified timing.
- Annual plan revenue is amortized evenly across months if enabled.
- Attribution/organic separation will vary by how you tag campaigns and sources.

## Next ideas

- Pull Substack exports (CSV) to auto-populate baselines
- Scenario comparison and sensitivity analysis
- Cohort-based conversion and churn curves
- Funnel from traffic by source to free signups to premium

The way to use this is to export your csvs from substack for both paid and full subscribers.

Those spreadsheets have two columns... days and number of subscribers...
