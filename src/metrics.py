from typing import Dict

import numpy as np
import pandas as pd


def perf_stats(
    r: pd.Series,
    periods_per_year: int = 12,
) -> Dict[str, float]:
    """
    Compute simple performance statistics on a log-return series.

    Returns a dict with:
      'Ann. Return', 'Ann. Vol', 'Sharpe', 'Max Drawdown'.
    """
    r = r.dropna()

    mu = float(r.mean() * periods_per_year)
    sigma = float(r.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = mu / sigma if sigma > 0 else float("nan")

    cum = np.exp(r.cumsum())
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    mdd = float(dd.min())

    return {
        "Ann. Return": mu,
        "Ann. Vol": sigma,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
    }


def window_stats(
    r: pd.Series,
    start: str,
    end: str,
    label: str,
) -> Dict[str, object]:
    """
    Compute cumulative return and max drawdown on a given window
    for a log-return series r.
    """
    r_win = r.loc[start:end].dropna()
    if r_win.empty:
        return {
            "Label": label,
            "Start": start,
            "End": end,
            "CumReturn": float("nan"),
            "MaxDD": float("nan"),
        }

    cum = np.exp(r_win.cumsum())
    cum_ret = float(cum.iloc[-1] - 1.0)
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    mdd = float(dd.min())

    return {
        "Label": label,
        "Start": start,
        "End": end,
        "CumReturn": cum_ret,
        "MaxDD": mdd,
    }
