import pandas as pd

from substack_analyzer.analysis import compute_estimates, derive_adds_churn, read_series


def test_compute_estimates_with_both_series():
    idx = pd.period_range("2024-01", periods=6, freq="ME").to_timestamp("ME")
    all_series = pd.Series([100, 110, 120, 135, 150, 170], index=idx)
    paid_series = pd.Series([10, 12, 13, 15, 17, 20], index=idx)
    est = compute_estimates(all_series, paid_series, window_months=3)
    assert est["start_free"] == int(all_series.iloc[-1] - paid_series.iloc[-1])
    assert est["start_premium"] == int(paid_series.iloc[-1])
    assert 0 <= est["organic_growth"] <= 1
    assert 0 <= est["conv_ongoing"] <= 1


def test_derive_adds_churn_happy_path():
    idx = pd.period_range("2024-01", periods=4, freq="ME").to_timestamp("ME")
    total = pd.Series([100, 110, 120, 130], index=idx)
    paid = pd.Series([10, 12, 13, 15], index=idx)
    plot_df = pd.DataFrame({"Total": total, "Paid": paid})
    adds_df, churn_df = derive_adds_churn(plot_df, churn_free_est=0.02, churn_paid_est=0.03)
    assert not adds_df.empty and not churn_df.empty
    # Each output should align to the same timeline except first row (diff-based)
    assert len(adds_df) == len(plot_df) - 1
    assert len(churn_df) == len(plot_df) - 1


def test_read_series_csv(tmp_path):
    # Create a small CSV with header row
    csv = "date,count\n2024-01-15,100\n2024-01-31,110\n2024-02-28,120\n"
    p = tmp_path / "series.csv"
    p.write_text(csv)
    with p.open("rb") as fh:
        s = read_series(fh, has_header=True, date_sel="date", count_sel="count")
    # Should be monthly-end indexed, last observation per month
    assert not s.empty
    assert s.index[-1].day >= 28


# end
