from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def fit_hmm_with_bic(
    X: pd.DataFrame,
    n_components_list: List[int] = (2, 3, 4),
    random_state: int = 42,
) -> Tuple[GaussianHMM, int, float]:
    """
    Fit Gaussian HMM models for different numbers of states and select the
    best one based on BIC.

    Returns the best HMM, number of states, and the BIC value.
    """
    X_values = X.values
    X_scaled = StandardScaler().fit_transform(X_values)
    T_hmm, d = X_scaled.shape

    best_bic = np.inf
    best_hmm = None
    best_n = None

    for n_components in n_components_list:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=500,
            random_state=random_state,
        )
        model.fit(X_scaled)
        logL = model.score(X_scaled)

        k = (n_components - 1) + n_components * (n_components - 1) + 2 * n_components * d
        bic = k * np.log(T_hmm) - 2 * logL

        if bic < best_bic:
            best_bic = bic
            best_hmm = model
            best_n = n_components

    if best_hmm is None:
        raise RuntimeError("HMM fitting failed for all candidate state counts.")

    return best_hmm, int(best_n), float(best_bic)


def compute_state_profiles(
    X: pd.DataFrame,
    posteriors: np.ndarray,
) -> pd.DataFrame:
    """
    Compute posterior-weighted state characteristics (means of each feature).

    Returns a DataFrame indexed by 'State i' with columns equal to X.columns.
    """
    best_n = posteriors.shape[1]
    state_info = []

    for i in range(best_n):
        p = posteriors[:, i]
        weighted_means = (X.mul(p, axis=0).sum(axis=0) / p.sum())
        state_info.append(weighted_means)

    state_df = pd.DataFrame(state_info)
    state_df.index = [f"State {i}" for i in range(best_n)]
    return state_df


def label_states_by_severity(
    state_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Label regimes based on a simple severity score using VIX, HY OAS and MOVE.

    Returns a mapping from 'State i' to regime label.
    """
    severity_scores: Dict[str, float] = {}

    for state_name in state_df.index:
        row = state_df.loc[state_name]
        score = (
            0.6 * row["VIX_MonthlyAvg"]
            + 0.3 * row["HY OAS"]
            + 0.1 * row["MOVE Proxy (σ Δ10Y, ann.)"]
        )
        severity_scores[state_name] = float(score)

    # Sort from least to most severe
    sorted_states = sorted(severity_scores.items(), key=lambda x: x[1])

    # Up to four labels, truncated if fewer states
    label_names = [
        "Carry-Friendly",
        "Volatility Transition",
        "Defensive",
        "Capital-Protection Mode",
    ][: len(sorted_states)]

    state_mapping = {
        state_name: label_names[i]
        for i, (state_name, _) in enumerate(sorted_states)
    }

    return state_mapping


def smooth_posteriors(
    posteriors: np.ndarray,
    index: pd.DatetimeIndex,
    span: int = 3,
) -> pd.DataFrame:
    """
    Exponentially smooth state posteriors and renormalise.

    Returns a DataFrame with columns State_0, State_1, ... and given index.
    """
    best_n = posteriors.shape[1]
    post_raw = pd.DataFrame(
        posteriors,
        index=index,
        columns=[f"State_{i}" for i in range(best_n)],
    )

    post_smooth = post_raw.ewm(span=span, min_periods=1).mean()
    post_smooth = post_smooth.div(post_smooth.sum(axis=1), axis=0)
    return post_smooth


def estimate_regimes(
    data: pd.DataFrame,
    feature_cols: List[str],
    span: int = 3,
    n_components_list: List[int] = (2, 3, 4),
    random_state: int = 42,
) -> Dict[str, object]:
    """
    High-level wrapper:
    1) select features
    2) fit HMM with BIC selection
    3) compute state profiles
    4) auto-label regimes by severity
    5) smooth posteriors

    Returns a dict with:
      'hmm', 'n_states', 'bic', 'X', 'posteriors', 'posteriors_smooth',
      'state_profile', 'state_mapping'.
    """
    X = data[feature_cols].dropna()
    hmm, n_states, bic = fit_hmm_with_bic(X, n_components_list, random_state)

    posteriors = hmm.predict_proba(StandardScaler().fit_transform(X.values))
    state_profile = compute_state_profiles(X, posteriors)
    state_mapping = label_states_by_severity(state_profile)
    post_smooth = smooth_posteriors(posteriors, X.index, span=span)

    result: Dict[str, object] = {
        "hmm": hmm,
        "n_states": n_states,
        "bic": bic,
        "X": X,
        "posteriors": posteriors,
        "posteriors_smooth": post_smooth,
        "state_profile": state_profile,
        "state_mapping": state_mapping,
    }
    return result
