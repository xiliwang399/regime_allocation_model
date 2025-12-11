from typing import Tuple

import numpy as np
import pandas as pd


def backtest_with_rebal(
    target_weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    tc: float = 0.0005,
) -> pd.Series:
    """
    Backtest a strategy that rebalances to given target weights with trading costs.

    target_weights_df: DataFrame of target weights over time (rows = dates)
    returns_df: DataFrame of asset log returns (same assets)
    tc: per-unit trading cost applied to one-way turnover

    Returns a Series of portfolio log returns.
    """
    tw = target_weights_df.copy()
    rets = returns_df.copy()

    common_idx = tw.index.intersection(rets.index)
    tw = tw.loc[common_idx]
    rets = rets.loc[common_idx]

    if len(tw) < 2:
        return pd.Series(dtype=float)

    tw = tw.div(tw.sum(axis=1), axis=0)

    out = []
    idx = []

    w_prev = tw.iloc[0].values

    for i in range(1, len(tw)):
        date_t = tw.index[i]
        r_t = rets.loc[date_t].values

        r_port = float(np.dot(w_prev, r_t))

        gross = np.exp(r_t)
        w_drift = w_prev * gross
        w_drift = w_drift / w_drift.sum()

        w_target = tw.iloc[i].values

        turnover_t = float(0.5 * np.abs(w_target - w_drift).sum())
        cost_t = tc * turnover_t

        r_net = r_port - cost_t

        out.append(r_net)
        idx.append(date_t)

        w_prev = w_target

    return pd.Series(out, index=idx)


def compute_turnover(
    weights_df: pd.DataFrame,
    periods_per_year: int = 12,
) -> Tuple[float, float]:
    """
    Compute average monthly and annualised turnover from a weight time series.

    Turnover definition: 0.5 * sum |w_t - w_{t-1}|
    """
    W_diff = weights_df.diff().abs()
    turnover_monthly = 0.5 * W_diff.sum(axis=1)
    avg_turnover_monthly = float(turnover_monthly.mean())
    avg_turnover_annual = avg_turnover_monthly * periods_per_year
    return avg_turnover_monthly, avg_turnover_annual
