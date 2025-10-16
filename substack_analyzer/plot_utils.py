"""
A simple plotting tool for visualizing series and fits.
"""

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from substack_analyzer.calibration import PiecewiseLogisticFit, _ensure_month_end_index


def plot_fit_vs_actual(
    input_series: pd.Series,
    fit: PiecewiseLogisticFit,
    title: Optional[str] = None,
    show_breakpoints: bool = True,
):
    """Overlay actual `input_series` and fitted series using matplotlib.

    This creates a standard pop-up window via matplotlib when run in a local
    Python session (e.g., from a script or REPL).

    Parameters
    ----------
    input_series : pd.Series
        Original monthly series (will be normalized to month-end index).
    fit : PiecewiseLogisticFit
        Result from `fit_piecewise_logistic` containing `fitted_series` and
        `breakpoints`.
    title : Optional[str]
        Optional chart title. Defaults to a summary with R^2 and SSE.
    show_breakpoints : bool
        If True, draw vertical rules at the model's breakpoints.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object for further customization.
    """

    actual = _ensure_month_end_index(input_series).astype(float)
    fitted = fit.fitted_series.reindex(actual.index).astype(float)

    if title is None:
        title = f"Actual vs Fitted (R^2 on deltas: {fit.r2_on_deltas:.3f}, SSE: {fit.sse:,.0f})"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual.values, marker="o", linewidth=1.8, label="Actual", color="#1f77b4")
    ax.plot(actual.index, fitted.values, marker="o", linewidth=1.8, label="Fitted", color="#2ca02c")

    # Optional vertical lines for breakpoints
    if show_breakpoints and getattr(fit, "breakpoints", None):
        idx = actual.index
        for b in fit.breakpoints:
            if isinstance(b, int) and 0 <= b < len(idx):
                ax.axvline(idx[b], color="#DB4437", linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_ylabel("Subscribers")
    ax.set_xlabel("Date")

    # Format x-axis as monthly with readable labels
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax
