import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Price Fetching
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Args:
        tickers:    List of asset ticker symbols (e.g., ['AAPL', 'BTC-USD'])
        period:     yfinance period string ('1y', '2y', '5y', 'max')
        interval:   Data granularity ('1d', '1wk', '1mo')
        use_cache:  If True, read from local cache when available

    Returns:
        DataFrame of adjusted close prices (rows=dates, cols=tickers)
    """
    cache_key = "_".join(sorted(tickers)) + f"_{period}_{interval}.parquet"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        logger.info(f"Loading price data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Fetching price data for: {tickers} | period={period} interval={interval}")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = _clean_prices(prices, tickers)

    prices.to_parquet(cache_path)
    logger.info(f"Cached price data to: {cache_path}")

    return prices


def _clean_prices(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Validate and clean the price DataFrame.
    - Forward-fill up to 5 consecutive NaN (weekends, holidays)
    - Drop rows where all values are NaN
    - Ensure all requested tickers are present
    """
    prices = prices.ffill(limit=5)
    prices = prices.dropna(how="all")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"No data returned for tickers: {missing}. Check ticker symbols.")

    return prices[tickers]


# ---------------------------------------------------------------------------
# Return Computation
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: ln(P_t / P_{t-1}).
    Preferred for multi-period compounding and normality assumptions.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns: (P_t - P_{t-1}) / P_{t-1}.
    Used for portfolio aggregation (VaR, CVaR calculations).
    """
    simple_returns = prices.pct_change()
    return simple_returns.dropna()


# ---------------------------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------------------------

def get_returns(
    tickers: List[str],
    period: str = "2y",
    return_type: str = "simple",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch prices and compute returns in one call.

    Args:
        tickers:     List of ticker symbols
        period:      Historical data period
        return_type: 'simple' or 'log'
        use_cache:   Use local cache if available

    Returns:
        DataFrame of daily returns (rows=dates, cols=tickers)
    """
    prices = fetch_price_data(tickers, period=period, use_cache=use_cache)

    if return_type == "log":
        return compute_log_returns(prices)
    elif return_type == "simple":
        return compute_simple_returns(prices)
    else:
        raise ValueError(f"Unknown return_type '{return_type}'. Use 'simple' or 'log'.")