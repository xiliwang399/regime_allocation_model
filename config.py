import numpy as np
import os

# Paths
FILE_PATH_INPUTS = os.path.join(os.path.dirname(__file__), "inputs.xlsx")
INPUTS_SHEET = "Inputs Monthly"
PRICES_SHEET = "Asset Prices"

# HMM feature columns for regime detection
HMM_FEATURE_COLS = [
    "VIX_MonthlyAvg",
    "MOVE Proxy (σ Δ10Y, ann.)",
    "HY OAS",
    "IG OAS",
    "Slope_10Y_3M",
]

# Full macro / geo panel for profiling regimes
MACRO_COLS = [
    "VIX_MonthlyAvg",
    "HY OAS",
    "IG OAS",
    "DGS10",
    "DGS3M",
    "DGS2Y",
    "TED Spread",
    "EFFR",
    "5Y Breakeven",
    "10Y Breakeven",
    "WTI Oil",
    "PPI All Commodities",
    "US EPU",
    "MOVE Proxy (σ Δ10Y, ann.)",
    "T-Bill 3M",
    "Slope_10Y_3M",
    "Slope_10Y_2Y",
    "GPR",
    "ISM",
    "LIBOR",
]

# Asset mapping from raw Excel columns to model names
ASSET_RENAME_MAP = {
    "US Treasury 7-10Y": "Rates_10Y",
    "US Bills 1-3M": "Bills_3M",
    "TIPS": "Infl_Linked",
    "Corporate IG": "IG_Credit",
    "Corporate HY": "HY_Credit",
    "CHFUSD SPOT": "CHFUSD",
    "GOLD": "GOLD",
    "MBS": "MBS",
    "GOLD USD SPOT Curncy": "XAUUSD",
    "JPY USD SPOT": "JPYUSD",
}

# Core asset universe for allocation
ASSETS = [
    "Rates_10Y",
    "Bills_3M",
    "IG_Credit",
    "HY_Credit",
    "Infl_Linked",
    "Tail_Hedge",
]

# Choose which series is used as tail hedge
TAIL_HEDGE_SOURCE = "GOLD"

# Min / max bounds per asset (same order as ASSETS)
ASSET_MIN = np.array([0.15, 0.15, 0.05, 0.00, 0.00, 0.00])
ASSET_MAX = np.array([0.60, 0.60, 0.30, 0.30, 0.30, 0.30])

# Barbell benchmark weights (same order as ASSETS)
BARBELL_WEIGHTS = np.array([
    0.35,  # Rates_10Y
    0.30,  # Bills_3M
    0.10,  # IG_Credit
    0.05,  # HY_Credit
    0.05,  # Infl_Linked
    0.05,  # Tail_Hedge
])

# Trading costs
TC_STRUCTURAL = 0.0005
TC_BARBELL = 0.001

# Regime-dependent risk aversion gamma, by regime label
GAMMA_BY_LABEL = {
    "Carry-Friendly": 3.0,
    "Volatility Transition": 5.0,
    "Defensive": 8.0,
    "Capital-Protection Mode": 10.0,
}

# Base safe-asset share by regime (Bills + Tail_Hedge)
BASE_SAFE_SHARE_BY_LABEL = {
    "Carry-Friendly": 0.40,
    "Volatility Transition": 0.45,
    "Defensive": 0.50,
    "Capital-Protection Mode": 0.55,
}

# Macro tilt sensitivities
K_INFL = 0.02
K_GROWTH = 0.02
K_POLICY_GEO = 0.02

# Crisis windows for reporting
CRISIS_WINDOWS = {
    "GFC_2007_2009": ("2007-07-01", "2009-06-30"),
    "COVID_2020": ("2020-02-01", "2020-12-31"),
}
