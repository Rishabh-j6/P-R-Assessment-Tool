import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Portfolio Return Series
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> pd.Series:
    """
    Compute weighted daily portfolio return series.

    Args:
        returns_df: DataFrame of daily asset returns (rows=dates, cols=tickers)
        weights:    Array of portfolio weights aligned to returns_df columns

    Returns:
        pd.Series of daily portfolio returns
    """
    if len(weights) != returns_df.shape[1]:
        raise ValueError("Weights length must match number of assets.")
    if abs(sum(weights) - 1.0) > 1e-4:
        raise ValueError("Weights must sum to 1.0")
    return returns_df.dot(weights)


# ---------------------------------------------------------------------------
# Value at Risk
# ---------------------------------------------------------------------------

def historical_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Historical (non-parametric) VaR.
    Sorts realized returns and takes the loss at the given percentile.

    Args:
        portfolio_returns: Daily portfolio return series
        confidence:        Confidence level (0.95 → 95%)
        portfolio_value:   Total portfolio value in USD

    Returns:
        VaR as a positive dollar loss figure
    """
    cutoff = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return abs(cutoff) * portfolio_value


def parametric_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    Parametric (Gaussian) VaR — assumes normally distributed returns.

    Args:
        portfolio_returns: Daily portfolio return series
        confidence:        Confidence level
        portfolio_value:   Total portfolio value in USD

    Returns:
        VaR as a positive dollar loss figure
    """
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return max(var, 0.0) * portfolio_value


# ---------------------------------------------------------------------------
# Conditional VaR (Expected Shortfall)
# ---------------------------------------------------------------------------

def conditional_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """
    CVaR / Expected Shortfall: mean loss beyond the VaR threshold.
    More robust than VaR for fat-tailed distributions.

    Args:
        portfolio_returns: Daily portfolio return series
        confidence:        Confidence level (0.95 → 95%)
        portfolio_value:   Total portfolio value in USD

    Returns:
        CVaR as a positive dollar loss figure
    """
    cutoff = np.percentile(portfolio_returns, (1 - confidence) * 100)
    tail_losses = portfolio_returns[portfolio_returns <= cutoff]
    if tail_losses.empty:
        return 0.0
    return abs(tail_losses.mean()) * portfolio_value


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """
    Annualized Sharpe Ratio.

    Args:
        portfolio_returns: Daily portfolio return series
        risk_free_rate:    Annual risk-free rate (e.g., 0.05 for 5%)

    Returns:
        Annualized Sharpe ratio (float)
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = portfolio_returns - daily_rf
    if excess_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Portfolio Volatility
# ---------------------------------------------------------------------------

def portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Annualized portfolio volatility using covariance matrix.

    Args:
        returns_df: DataFrame of daily asset returns
        weights:    Portfolio weights array

    Returns:
        Annualized volatility (float)
    """
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
    variance = weights @ cov_matrix.values @ weights
    return float(np.sqrt(variance))


def annualized_return(
    portfolio_returns: pd.Series,
) -> float:
    """
    Compound annualized portfolio return.

    Args:
        portfolio_returns: Daily portfolio return series

    Returns:
        Annualized return (float)
    """
    total_return = (1 + portfolio_returns).prod()
    n_days = len(portfolio_returns)
    if n_days == 0:
        return 0.0
    return float(total_return ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------

def maximum_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown over the entire return series.

    Args:
        portfolio_returns: Daily portfolio return series

    Returns:
        Max drawdown as a negative fraction (e.g., -0.35 = -35%)
    """
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return float(drawdowns.min())


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------

def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix of asset returns.

    Args:
        returns_df: DataFrame of daily asset returns

    Returns:
        Correlation matrix as a DataFrame
    """
    return returns_df.corr()


# ---------------------------------------------------------------------------
# Aggregate Risk Report
# ---------------------------------------------------------------------------

def compute_all_metrics(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float = 1_000_000.0,
    risk_free_rate: float = 0.05,
) -> Dict:
    """
    Compute all risk metrics in one call.

    Args:
        returns_df:      DataFrame of daily asset returns
        weights:         Portfolio weights aligned to returns_df columns
        portfolio_value: Total portfolio value in USD
        risk_free_rate:  Annual risk-free rate

    Returns:
        Dictionary containing all computed risk metrics
    """
    port_returns = compute_portfolio_returns(returns_df, weights)

    return {
        "var_95":               historical_var(port_returns, 0.95, portfolio_value),
        "var_99":               historical_var(port_returns, 0.99, portfolio_value),
        "cvar_95":              conditional_var(port_returns, 0.95, portfolio_value),
        "cvar_99":              conditional_var(port_returns, 0.99, portfolio_value),
        "sharpe_ratio":         sharpe_ratio(port_returns, risk_free_rate),
        "portfolio_volatility": portfolio_volatility(returns_df, weights),
        "portfolio_return":     annualized_return(port_returns),
        "max_drawdown":         maximum_drawdown(port_returns),
        "correlation_matrix":   correlation_matrix(returns_df).to_dict(),
    }