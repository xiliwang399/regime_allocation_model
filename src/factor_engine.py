from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MACRO_COLS


def zscore(series: pd.Series) -> pd.Series:
    """
    Simple z-score, with population-standard deviation (ddof=0).
    """
    s = series.astype(float)
    return (s - s.mean()) / s.std(ddof=0)


def build_macro_panel(
    data: pd.DataFrame,
    macro_cols: List[str] = MACRO_COLS,
) -> pd.DataFrame:
    """
    Extract the macro / policy / geo panel from the full data.
    """
    panel = data[macro_cols]
    return panel


def align_macro_with_regimes(
    macro_panel: pd.DataFrame,
    post_smooth: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align macro panel and smoothed posteriors on common dates,
    dropping rows with missing macro data.
    """
    macro_complete_idx = macro_panel.dropna().index
    common_idx = macro_complete_idx.intersection(post_smooth.index)

    macro_panel_aligned = macro_panel.loc[common_idx].dropna()
    post_smooth_aligned = post_smooth.loc[common_idx]

    return macro_panel_aligned, post_smooth_aligned


def compute_macro_state_means(
    macro_panel: pd.DataFrame,
    post_smooth: pd.DataFrame,
    state_mapping: Dict[str, str],
) -> pd.DataFrame:
    """
    Compute regime-conditional macro means using smoothed posteriors.

    Returns a DataFrame indexed by regime label.
    """
    state_cols = [c for c in post_smooth.columns if c.startswith("State_")]
    K = len(state_cols)

    macro_state_means = []

    for k in range(K):
        col_name = f"State_{k}"
        p_k = post_smooth[col_name].values
        w_k = p_k / p_k.sum()
        m_k = (macro_panel.mul(w_k, axis=0)).sum(axis=0)
        macro_state_means.append(m_k)

    macro_state_df = pd.DataFrame(
        macro_state_means,
        index=[f"State {k}" for k in range(K)],
    )

    regime_labels = [state_mapping[f"State {k}"] for k in range(K)]
    macro_state_df.index = regime_labels

    return macro_state_df


def compute_risk_signals(
    macro_panel: pd.DataFrame,
    panel_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute macro-based risk signals used for tilting the allocation.

    Returns a DataFrame indexed by panel_index with:
      'infl_risk', 'growth_risk', 'policy_risk', 'geo_risk',
      'combo_risk', 'z_EPU', 'z_GPR'.
    """
    signals = macro_panel[["5Y Breakeven", "ISM", "US EPU", "GPR"]].reindex(panel_index)
    signals = signals.ffill().bfill()

    z_infl = zscore(signals["5Y Breakeven"])
    z_ISM = zscore(signals["ISM"])
    z_EPU = zscore(signals["US EPU"])
    z_GPR = zscore(signals["GPR"])

    infl_risk = np.clip(z_infl, 0, None)
    growth_risk = np.clip(-z_ISM, 0, None)
    policy_risk = np.clip(z_EPU, 0, None)
    geo_risk = np.clip(z_GPR, 0, None)
    combo_risk = 0.5 * policy_risk + 0.5 * geo_risk

    out = pd.DataFrame(
        {
            "infl_risk": infl_risk,
            "growth_risk": growth_risk,
            "policy_risk": policy_risk,
            "geo_risk": geo_risk,
            "combo_risk": combo_risk,
            "z_EPU": z_EPU,
            "z_GPR": z_GPR,
        },
        index=panel_index,
    )

    return out
