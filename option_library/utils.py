"""
Utility Functions

This module provides utility functions for option pricing and analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')


def calculate_greeks(S0: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate option Greeks
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Dictionary with Greeks
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    
    # Vega (same for calls and puts)
    vega = S0 * np.sqrt(T) * norm.pdf(d1)
    
    # Theta
    if option_type.lower() == 'call':
        theta = -S0 * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S0 * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Rho
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def implied_volatility(S0: float, K: float, T: float, r: float, market_price: float, 
                      option_type: str = 'call', max_iter: int = 100, tolerance: float = 1e-6) -> float:
    """
    Calculate implied volatility using Newton-Raphson method
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        market_price: Market option price
        option_type: "call" or "put"
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatility
    """
    sigma = 0.3  # Initial guess
    
    for _ in range(max_iter):
        # Calculate option price and vega
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        
        vega = S0 * np.sqrt(T) * norm.pdf(d1)
        
        # Update sigma
        sigma_new = sigma - (price - market_price) / vega
        
        if abs(sigma_new - sigma) < tolerance:
            break
        
        sigma = sigma_new
    
    return sigma


def option_payoff(S: float, K: float, option_type: str = 'call') -> float:
    """
    Calculate option payoff
    
    Args:
        S: Stock price at expiry
        K: Strike price
        option_type: "call" or "put"
        
    Returns:
        Option payoff
    """
    if option_type.lower() == 'call':
        return max(S - K, 0)
    else:
        return max(K - S, 0)


def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        
    Returns:
        Option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def monte_carlo_price(S0: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call', n_simulations: int = 10000, n_steps: int = 252) -> Dict[str, float]:
    """
    Calculate option price using Monte Carlo simulation
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        n_simulations: Number of simulations
        n_steps: Number of time steps
        
    Returns:
        Dictionary with price and confidence interval
    """
    dt = T / n_steps
    
    # Generate random paths
    Z = np.random.normal(0, 1, (n_simulations, n_steps))
    returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(returns, axis=1))
    
    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(paths[:, -1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[:, -1], 0)
    
    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
    
    return {
        'price': price,
        'std_error': std_error,
        'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
    }


def binomial_tree_price(S0: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = 'call', n_steps: int = 100) -> float:
    """
    Calculate option price using binomial tree
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        n_steps: Number of time steps
        
    Returns:
        Option price
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize option values at maturity
    option_values = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        S = S0 * (u ** (n_steps - i)) * (d ** i)
        if option_type.lower() == 'call':
            option_values[i] = max(S - K, 0)
        else:
            option_values[i] = max(K - S, 0)
    
    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
    
    return option_values[0]


def calculate_historical_volatility(prices: pd.Series, window: int = 252) -> float:
    """
    Calculate historical volatility
    
    Args:
        prices: Series of stock prices
        window: Rolling window size
        
    Returns:
        Historical volatility (annualized)
    """
    returns = prices.pct_change().dropna()
    return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)


def calculate_realized_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate realized volatility
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Series of realized volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR at specified confidence level
    """
    return np.percentile(returns, (1 - confidence_level) * 100)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall)
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR at specified confidence level
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.inf
    
    return excess_returns.mean() / downside_deviation * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate maximum drawdown
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary with maximum drawdown statistics
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_date': max_dd_date,
        'drawdown_series': drawdown
    }


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix
    
    Args:
        returns_df: DataFrame with returns for multiple assets
        
    Returns:
        Correlation matrix
    """
    return returns_df.corr()


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        
    Returns:
        Beta
    """
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance


def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series, 
                           risk_free_rate: float = 0.02) -> float:
    """
    Calculate Treynor ratio
    
    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Treynor ratio
    """
    beta = calculate_beta(asset_returns, market_returns)
    excess_return = asset_returns.mean() - risk_free_rate / 252
    return excess_return / beta


def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio
    
    Args:
        asset_returns: Asset returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    active_returns = asset_returns - benchmark_returns
    return active_returns.mean() / active_returns.std() * np.sqrt(252)


def calculate_calmar_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Calmar ratio
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Calmar ratio
    """
    excess_return = returns.mean() - risk_free_rate / 252
    max_dd = calculate_max_drawdown(returns)['max_drawdown']
    
    if max_dd == 0:
        return np.inf
    
    return excess_return / abs(max_dd) * 252


def format_currency(value: float, currency: str = 'USD') -> str:
    """
    Format value as currency
    
    Args:
        value: Value to format
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    elif currency == 'EUR':
        return f"€{value:,.2f}"
    elif currency == 'GBP':
        return f"£{value:,.2f}"
    else:
        return f"{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (as decimal)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def print_summary_statistics(data: pd.Series, name: str = "Data") -> None:
    """
    Print summary statistics
    
    Args:
        data: Series of data
        name: Name of the data series
    """
    print(f"\n📊 {name} Summary Statistics")
    print("=" * 40)
    print(f"Count: {len(data):,}")
    print(f"Mean: {data.mean():.4f}")
    print(f"Std: {data.std():.4f}")
    print(f"Min: {data.min():.4f}")
    print(f"Max: {data.max():.4f}")
    print(f"25%: {data.quantile(0.25):.4f}")
    print(f"50%: {data.quantile(0.50):.4f}")
    print(f"75%: {data.quantile(0.75):.4f}") 