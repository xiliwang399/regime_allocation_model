from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    ASSETS,
    ASSET_MIN,
    ASSET_MAX,
    GAMMA_BY_LABEL,
    BASE_SAFE_SHARE_BY_LABEL,
    K_INFL,
    K_GROWTH,
    K_POLICY_GEO,
)


def compute_regime_asset_stats(
    returns: pd.DataFrame,
    posteriors: pd.DataFrame,
    assets: List[str] = ASSETS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute regime-conditional mean returns and covariance matrices.

    returns: T x N DataFrame of asset returns
    posteriors: T x K DataFrame of state probabilities (columns 'State_0', ...)

    Returns (state_means, state_covs) with shapes:
      state_means: K x N
      state_covs: K x N x N
    """
    R = returns[assets].copy()
    state_cols = [c for c in posteriors.columns if c.startswith("State_")]

    P = posteriors[state_cols].values
    T_final = len(R)
    N = len(assets)
    K = P.shape[1]

    state_means = []
    state_covs = []

    for k in range(K):
        p_k = P[:, k]
        weight_sum = p_k.sum()

        if weight_sum <= 0:
            mu_k = np.zeros(N)
            Sigma_k = np.eye(N)
        else:
            w_k = p_k / weight_sum
            mu_k = (R.mul(w_k, axis=0)).sum(axis=0).values

            X = R.values - mu_k[None, :]
            X_w = X * np.sqrt(w_k)[:, None]

            lw = LedoitWolf().fit(X_w)
            Sigma_k = lw.covariance_

        state_means.append(mu_k)
        state_covs.append(Sigma_k)

    return np.array(state_means), np.array(state_covs)


def build_gamma_by_state(
    state_mapping: Dict[str, str],
) -> Dict[int, float]:
    """
    Map numerical state index k to risk-aversion gamma using GAMMA_BY_LABEL.
    """
    gamma_regime: Dict[int, float] = {}

    for state_name, label in state_mapping.items():
        k = int(state_name.split()[-1])
        gamma_regime[k] = GAMMA_BY_LABEL.get(label, 5.0)

    return gamma_regime


def construct_regime_portfolios(
    state_means: np.ndarray,
    state_covs: np.ndarray,
    state_mapping: Dict[str, str],
    asset_min: np.ndarray = ASSET_MIN,
    asset_max: np.ndarray = ASSET_MAX,
) -> np.ndarray:
    """
    For each regime, construct a constrained mean–variance portfolio:

        w* = Σ^{-1} μ / γ

    Subject to:
      long-only, IPS-like min/max bounds, weights renormalised to sum to 1.
    """
    K, N = state_means.shape
    gamma_regime = build_gamma_by_state(state_mapping)

    state_weights = []

    for k in range(K):
        mu = state_means[k]
        Sigma = state_covs[k]
        gamma_k = gamma_regime.get(k, 5.0)

        try:
            w_raw = np.linalg.solve(Sigma, mu) / gamma_k
        except np.linalg.LinAlgError:
            w_raw = np.linalg.pinv(Sigma) @ mu / gamma_k

        w_clipped = np.clip(w_raw, 0, None)
        if w_clipped.sum() == 0:
            w_clipped = np.ones(N)

        w_bounded = np.minimum(np.maximum(w_clipped, asset_min), asset_max)

        s = w_bounded.sum()
        if s <= 0:
            w_bounded = np.ones(N) / N
        else:
            w_bounded = w_bounded / s

        state_weights.append(w_bounded)

    return np.array(state_weights)


def mix_regime_portfolios_over_time(
    state_weights: np.ndarray,
    posteriors: pd.DataFrame,
    index: pd.DatetimeIndex,
    assets: List[str] = ASSETS,
) -> pd.DataFrame:
    """
    Build the base time-varying portfolio as a posterior-weighted mixture
    of regime-specific portfolios.

    state_weights: K x N
    posteriors: T x K DataFrame of state probabilities ('State_0', ...)
    """
    state_cols = [c for c in posteriors.columns if c.startswith("State_")]
    P = posteriors[state_cols].values
    T_final = len(index)
    K, N = state_weights.shape

    if P.shape[0] != T_final or P.shape[1] != K:
        raise ValueError("Dimensions of posteriors and state weights do not match.")

    W_base = np.zeros((T_final, N))

    for t in range(T_final):
        W_base[t, :] = P[t, :] @ state_weights

    return pd.DataFrame(W_base, index=index, columns=assets)


def build_base_safe_share_by_state(
    state_mapping: Dict[str, str],
) -> Dict[int, float]:
    """
    Map numerical state index to base safe-asset share (Bills + Tail).
    """
    base_safe_regime: Dict[int, float] = {}

    for state_name, label in state_mapping.items():
        k = int(state_name.split()[-1])
        base_safe_regime[k] = BASE_SAFE_SHARE_BY_LABEL.get(label, 0.45)

    return base_safe_regime


def tail_floor(z_epu: float, z_gpr: float) -> float:
    """
    Tail-hedge floor as a function of policy and geopolitical risk z-scores.
    """
    risk = max(0.0, 0.5 * z_epu + 0.5 * z_gpr)
    floor = 0.05 + 0.05 * risk
    return float(np.clip(floor, 0.05, 0.25))


def build_final_allocation(
    W_base: pd.DataFrame,
    posteriors: pd.DataFrame,
    risk_signals: pd.DataFrame,
    state_mapping: Dict[str, str],
    assets: List[str] = ASSETS,
    asset_min: np.ndarray = ASSET_MIN,
    asset_max: np.ndarray = ASSET_MAX,
    k_infl: float = K_INFL,
    k_growth: float = K_GROWTH,
    k_risk: float = K_POLICY_GEO,
) -> pd.DataFrame:
    """
    Apply macro and regime-driven tilts, safe-share control,
    and tail-hedge floor on top of the base regime-mix portfolio.

    This function reproduces the logic from the notebook:
      1) regime-based safe-share
      2) inflation tilt
      3) growth tilt
      4) policy/geo tilt
      5) enforce safe-share target
      6) apply tail-hedge floor
      7) reapply bounds and renormalise
    """
    W_base = W_base.copy()
    panel_index = W_base.index
    state_cols = [c for c in posteriors.columns if c.startswith("State_")]
    P = posteriors[state_cols].loc[panel_index].values

    N = len(assets)
    T_final = len(panel_index)

    i_rates = assets.index("Rates_10Y")
    i_bills = assets.index("Bills_3M")
    i_ig = assets.index("IG_Credit")
    i_hy = assets.index("HY_Credit")
    i_infl = assets.index("Infl_Linked")
    i_tail = assets.index("Tail_Hedge")

    base_safe_regime = build_base_safe_share_by_state(state_mapping)

    W_final = np.zeros_like(W_base.values)

    infl_risk = risk_signals["infl_risk"]
    growth_risk = risk_signals["growth_risk"]
    combo_risk = risk_signals["combo_risk"]
    z_EPU = risk_signals["z_EPU"]
    z_GPR = risk_signals["z_GPR"]

    K_states = P.shape[1]

    for t, date in enumerate(panel_index):
        w = W_base.iloc[t].values.copy()

        base_safe_t = 0.0
        for k in range(K_states):
            base_safe_t += P[t, k] * base_safe_regime[k]
        safe_target_t = float(np.clip(base_safe_t, 0.20, 0.80))

        tilt_infl = float(np.clip(k_infl * infl_risk.loc[date], 0.0, 0.06))
        shift = min(tilt_infl, w[i_rates])
        if shift > 0:
            w[i_rates] -= shift
            w[i_infl] += 0.7 * shift
            w[i_tail] += 0.3 * shift

        tilt_growth = float(np.clip(k_growth * growth_risk.loc[date], 0.0, 0.07))
        shift = min(tilt_growth, w[i_hy])
        if shift > 0:
            w[i_hy] -= shift
            w[i_ig] += 0.5 * shift
            w[i_bills] += 0.5 * shift

        tilt_risk = float(np.clip(k_risk * combo_risk.loc[date], 0.0, 0.08))
        if tilt_risk > 0:
            from_hy = min(0.7 * tilt_risk, w[i_hy])
            from_rates = min(0.3 * tilt_risk, w[i_rates])
            total_fund = from_hy + from_rates
            if total_fund > 0:
                w[i_hy] -= from_hy
                w[i_rates] -= from_rates
                w[i_bills] += 0.6 * total_fund
                w[i_tail] += 0.4 * total_fund

        safe_cur = w[i_bills] + w[i_tail]
        diff_safe = safe_target_t - safe_cur

        if diff_safe > 0:
            risky_idx = [i_rates, i_ig, i_hy, i_infl]
            risky_weights = w[risky_idx].copy()
            risky_sum = risky_weights.sum()
            if risky_sum > 0:
                move = min(diff_safe, risky_sum)
                frac = risky_weights / risky_sum
                w[risky_idx] -= move * frac
                w[i_bills] += 0.7 * move
                w[i_tail] += 0.3 * move
        elif diff_safe < 0:
            safe_weights = np.array([w[i_bills], w[i_tail]])
            safe_sum = safe_weights.sum()
            need = min(-diff_safe, safe_sum)
            if need > 0:
                if safe_sum > 0:
                    frac_safe = safe_weights / safe_sum
                else:
                    frac_safe = np.array([0.5, 0.5])
                take_bills = need * frac_safe[0]
                take_tail = need * frac_safe[1]
                w[i_bills] -= take_bills
                w[i_tail] -= take_tail

                risky_idx = [i_rates, i_ig, i_hy, i_infl]
                risky_weights = np.clip(w[risky_idx], 0, None)
                risky_sum = risky_weights.sum()
                if risky_sum > 0:
                    frac_risky = risky_weights / risky_sum
                else:
                    frac_risky = np.ones(len(risky_idx)) / len(risky_idx)
                w[risky_idx] += need * frac_risky

        w_tail_min = tail_floor(z_EPU.loc[date], z_GPR.loc[date])
        if w[i_tail] < w_tail_min:
            extra = w_tail_min - w[i_tail]
            take_bills = min(extra, w[i_bills])
            w[i_bills] -= take_bills
            extra -= take_bills
            if extra > 0 and w[i_hy] > 0:
                take_hy = min(extra, w[i_hy])
                w[i_hy] -= take_hy
                extra -= take_hy
            w[i_tail] = w_tail_min

        w = np.minimum(np.maximum(w, asset_min), asset_max)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w[:] = 1.0 / N
        else:
            w /= w.sum()

        W_final[t, :] = w

    return pd.DataFrame(W_final, index=panel_index, columns=assets)
