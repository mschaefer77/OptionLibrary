"""
Market Data Module

This module provides tools for fetching and analyzing real market option data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MarketDataProvider:
    """Market data provider using YFinance"""
    
    def __init__(self, ticker: str):
        """
        Initialize market data provider
        
        Args:
            ticker: Stock symbol (e.g., "AAPL", "TSLA")
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.current_price = self._get_current_price()
        self.volatility = self._estimate_volatility()
    
    def _get_current_price(self) -> float:
        """Get current stock price"""
        try:
            return self.stock.info.get('regularMarketPrice', 100)
        except:
            return 100
    
    def _estimate_volatility(self, days: int = 252) -> float:
        """Estimate volatility from historical data"""
        try:
            hist = self.stock.history(period=f"{days}d")
            returns = hist['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)
        except:
            return 0.25  # Default 25% volatility
    
    def get_option_chain(self, expiry_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get option chain data
        
        Args:
            expiry_date: Option expiry date (YYYY-MM-DD)
            
        Returns:
            Dictionary with calls and puts DataFrames
        """
        try:
            if expiry_date:
                chain = self.stock.option_chain(expiry_date)
                return {
                    'calls': chain.calls,
                    'puts': chain.puts,
                    'expiry_date': expiry_date
                }
            else:
                # Get first available expiry
                available_dates = self.stock.options
                if available_dates:
                    chain = self.stock.option_chain(available_dates[0])
                    return {
                        'calls': chain.calls,
                        'puts': chain.puts,
                        'expiry_date': available_dates[0]
                    }
        except Exception as e:
            print(f"Error fetching option chain: {e}")
        
        return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_all_options(self, max_expiries: int = 10) -> pd.DataFrame:
        """
        Get all available options
        
        Args:
            max_expiries: Maximum number of expiry dates to fetch
            
        Returns:
            DataFrame with all options
        """
        all_options = []
        
        try:
            available_dates = self.stock.options[:max_expiries]
            
            for date in available_dates:
                chain = self.stock.option_chain(date)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                calls['option_type'] = 'call'
                puts['option_type'] = 'put'
                calls['expiry_date'] = date
                puts['expiry_date'] = date
                
                all_options.extend([calls, puts])
            
            if all_options:
                return pd.concat(all_options, ignore_index=True)
                
        except Exception as e:
            print(f"Error fetching options: {e}")
        
        return pd.DataFrame()
    
    def get_historical_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            
        Returns:
            DataFrame with historical data
        """
        try:
            return self.stock.history(period=period)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()


class OptionChainAnalyzer:
    """Analyze option chain data"""
    
    def __init__(self, market_data: pd.DataFrame):
        """
        Initialize option chain analyzer
        
        Args:
            market_data: DataFrame with option data
        """
        self.market_data = market_data
    
    def calculate_implied_volatility(self, S0: float, r: float = 0.05) -> pd.DataFrame:
        """
        Calculate implied volatility for options
        
        Args:
            S0: Current stock price
            r: Risk-free rate
            
        Returns:
            DataFrame with implied volatilities
        """
        if self.market_data.empty:
            return pd.DataFrame()
        
        # Calculate time to expiry
        self.market_data['expiry_date'] = pd.to_datetime(self.market_data['expiry_date'])
        today = pd.Timestamp.now()
        self.market_data['T'] = (self.market_data['expiry_date'] - today).dt.days / 365
        
        # Calculate implied volatility
        implied_vols = []
        
        for _, row in self.market_data.iterrows():
            try:
                K = row['strike']
                T = row['T']
                market_price = row['lastPrice']
                option_type = row['option_type']
                
                # Use Newton-Raphson to find implied volatility
                sigma = self._newton_raphson_iv(S0, K, T, r, market_price, option_type)
                implied_vols.append(sigma)
            except:
                implied_vols.append(np.nan)
        
        self.market_data['implied_volatility'] = implied_vols
        return self.market_data
    
    def _newton_raphson_iv(self, S0: float, K: float, T: float, r: float, 
                           market_price: float, option_type: str, max_iter: int = 100) -> float:
        """
        Newton-Raphson method to find implied volatility
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            market_price: Market option price
            option_type: "call" or "put"
            max_iter: Maximum iterations
            
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
            
            if abs(sigma_new - sigma) < 1e-6:
                break
            
            sigma = sigma_new
        
        return sigma
    
    def analyze_volatility_smile(self) -> Dict[str, Any]:
        """
        Analyze volatility smile
        
        Returns:
            Dictionary with volatility smile analysis
        """
        if self.market_data.empty:
            return {}
        
        # Filter for calls and puts separately
        calls = self.market_data[self.market_data['option_type'] == 'call']
        puts = self.market_data[self.market_data['option_type'] == 'put']
        
        analysis = {
            'calls': {
                'strikes': calls['strike'].tolist() if not calls.empty else [],
                'implied_vols': calls['implied_volatility'].tolist() if not calls.empty else [],
                'moneyness': (calls['strike'] / self.S0).tolist() if not calls.empty else []
            },
            'puts': {
                'strikes': puts['strike'].tolist() if not puts.empty else [],
                'implied_vols': puts['implied_volatility'].tolist() if not puts.empty else [],
                'moneyness': (puts['strike'] / self.S0).tolist() if not puts.empty else []
            }
        }
        
        return analysis


def fetch_market_data(ticker: str, max_expiries: int = 3) -> Dict[str, Any]:
    """
    Convenience function to fetch market data
    
    Args:
        ticker: Stock symbol
        max_expiries: Maximum number of expiry dates
        
    Returns:
        Dictionary with market data
    """
    provider = MarketDataProvider(ticker)
    
    # Get current price and volatility
    current_price = provider.current_price
    volatility = provider.volatility
    
    # Get option chain
    options = provider.get_all_options(max_expiries)
    
    # Analyze option chain
    analyzer = OptionChainAnalyzer(options)
    if not options.empty:
        analyzer.S0 = current_price
        options_with_iv = analyzer.calculate_implied_volatility(current_price)
        volatility_smile = analyzer.analyze_volatility_smile()
    else:
        options_with_iv = pd.DataFrame()
        volatility_smile = {}
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'volatility': volatility,
        'options': options_with_iv,
        'volatility_smile': volatility_smile
    } 