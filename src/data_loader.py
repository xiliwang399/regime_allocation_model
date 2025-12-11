from typing import Tuple, List

import numpy as np
import pandas as pd

from config import (
    FILE_PATH_INPUTS,
    INPUTS_SHEET,
    PRICES_SHEET,
    ASSET_RENAME_MAP,
    ASSETS,
    TAIL_HEDGE_SOURCE,
)


def load_macro_data(
    filepath: str = FILE_PATH_INPUTS,
    sheet_name: str = INPUTS_SHEET,
) -> pd.DataFrame:
    """
    Load and time-align the macro / factor panel.

    Returns a monthly- frequency DataFrame indexed by Date.
    """
    data = pd.read_excel(filepath, sheet_name=sheet_name)
    data = data.rename(columns={"DATE": "Date"})
    data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y", errors="coerce")
    data = data.sort_values("Date")
    data = data.set_index("Date").asfreq("M")
    return data


def load_asset_prices(
    filepath: str = FILE_PATH_INPUTS,
    sheet_name: str = PRICES_SHEET,
) -> pd.DataFrame:
    """
    Load raw asset price data and resample to monthly frequency (last observation).
    """
    prices = pd.read_excel(filepath, sheet_name=sheet_name)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").set_index("Date")
    prices = prices.resample("M").last()
    prices = prices.rename(columns=ASSET_RENAME_MAP)
    return prices


def prepare_asset_returns(
    prices: pd.DataFrame,
    assets: List[str] = ASSETS,
    tail_hedge_source: str = TAIL_HEDGE_SOURCE,
) -> pd.DataFrame:
    """
    Compute asset log returns and select the core allocation universe.

    The tail-hedge asset is set based on `tail_hedge_source`.
    """
    prices = prices.copy()

    if tail_hedge_source not in prices.columns:
        raise KeyError(f"Tail hedge source '{tail_hedge_source}' not found in prices.")

    prices["Tail_Hedge"] = prices[tail_hedge_source]

    asset_returns = np.log(prices / prices.shift(1))
    asset_returns = asset_returns.dropna()

    missing = [a for a in assets if a not in asset_returns.columns]
    if missing:
        raise KeyError(f"Missing required assets in returns: {missing}")

    return asset_returns[assets]
