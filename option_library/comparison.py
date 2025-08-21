"""
Model vs Market Comparison Module

This module provides tools for comparing model-implied prices with market prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

from .pricing import BlackScholes, MonteCarloPricer, BinomialTree, TrinomialTree
from .dynamics import create_dynamics_model


class ModelMarketComparator:
    """Compare model prices with market prices"""
    
    def __init__(self, S0: float, r: float = 0.05):
        """
        Initialize comparator
        
        Args:
            S0: Current stock price
            r: Risk-free rate
        """
        self.S0 = S0
        self.r = r
    
    def compare_prices(self, market_data: pd.DataFrame, sigma: float = 0.25) -> pd.DataFrame:
        """
        Compare model prices with market prices
        
        Args:
            market_data: DataFrame with market option data
            sigma: Volatility to use for models
            
        Returns:
            DataFrame with comparison results
        """
        if market_data.empty:
            return pd.DataFrame()
        
        results = []
        
        for _, row in market_data.iterrows():
            K = row['strike']
            T = row['time_to_expiry']
            market_price = row['lastPrice']
            option_type = row['option_type']
            
            # Calculate model prices
            bs = BlackScholes(self.S0, K, T, self.r, sigma)
            # For Monte Carlo and Binomial Tree, pass None as dynamics_model to use GBM with the specified sigma
            mc = MonteCarloPricer(self.S0, K, T, self.r, None)
            bt = BinomialTree(self.S0, K, T, self.r, None)
            
            # Set the sigma manually for these models
            mc.sigma = sigma
            bt.sigma = sigma
            
            if option_type.lower() == 'call':
                bs_price = bs.price_call()
                mc_price = mc.price_call()['price']
                bt_price = bt.price_call()
            else:
                bs_price = bs.price_put()
                mc_price = mc.price_put()['price']
                bt_price = bt.price_put()
            
            # Calculate differences
            bs_diff = bs_price - market_price
            mc_diff = mc_price - market_price
            bt_diff = bt_price - market_price
            
            results.append({
                'strike': K,
                'expiry': row.get('expiry_date', 'Unknown'),
                'option_type': option_type,
                'market_price': market_price,
                'bs_price': bs_price,
                'mc_price': mc_price,
                'bt_price': bt_price,
                'bs_difference': bs_diff,
                'mc_difference': mc_diff,
                'bt_difference': bt_diff,
                'bs_percentage': (bs_diff / market_price) * 100 if market_price > 0 else 0,
                'mc_percentage': (mc_diff / market_price) * 100 if market_price > 0 else 0,
                'bt_percentage': (bt_diff / market_price) * 100 if market_price > 0 else 0,
                'sigma': sigma
            })
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, comparison_data: pd.DataFrame):
        """Plot model vs market price comparison"""
        if comparison_data.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model vs Market Price Comparison', fontsize=16)
        
        # 1. Price comparison
        axes[0, 0].scatter(comparison_data['market_price'], comparison_data['bs_price'], 
                           alpha=0.6, label='Black-Scholes')
        axes[0, 0].scatter(comparison_data['market_price'], comparison_data['mc_price'], 
                           alpha=0.6, label='Monte Carlo')
        axes[0, 0].scatter(comparison_data['market_price'], comparison_data['bt_price'], 
                           alpha=0.6, label='Binomial Tree')
        axes[0, 0].plot([0, comparison_data['market_price'].max()], 
                        [0, comparison_data['market_price'].max()], 'r--', label='Perfect Match')
        axes[0, 0].set_xlabel('Market Price')
        axes[0, 0].set_ylabel('Model Price')
        axes[0, 0].set_title('Price Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Price differences
        axes[0, 1].hist(comparison_data['bs_percentage'], bins=20, alpha=0.7, 
                        label='Black-Scholes', color='blue')
        axes[0, 1].hist(comparison_data['mc_percentage'], bins=20, alpha=0.7, 
                        label='Monte Carlo', color='orange')
        axes[0, 1].hist(comparison_data['bt_percentage'], bins=20, alpha=0.7, 
                        label='Binomial Tree', color='green')
        axes[0, 1].set_xlabel('Percentage Difference (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Price Difference Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Errors by strike
        axes[1, 0].scatter(comparison_data['strike'], comparison_data['bs_percentage'], 
                           alpha=0.6, label='Black-Scholes')
        axes[1, 0].scatter(comparison_data['strike'], comparison_data['mc_percentage'], 
                           alpha=0.6, label='Monte Carlo')
        axes[1, 0].scatter(comparison_data['strike'], comparison_data['bt_percentage'], 
                           alpha=0.6, label='Binomial Tree')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Strike Price')
        axes[1, 0].set_ylabel('Percentage Error (%)')
        axes[1, 0].set_title('Errors by Strike Price')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model comparison
        models = ['bs', 'mc', 'bt']
        model_names = ['Black-Scholes', 'Monte Carlo', 'Binomial Tree']
        errors = []
        
        for model in models:
            error = comparison_data[f'{model}_percentage'].abs().mean()
            errors.append(error)
        
        axes[1, 1].bar(model_names, errors, color=['blue', 'orange', 'green'])
        axes[1, 1].set_ylabel('Mean Absolute Percentage Error (%)')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, comparison_data: pd.DataFrame):
        """Print summary statistics"""
        if comparison_data.empty:
            print("No data to analyze")
            return
        
        print(f"\n📊 Model vs Market Price Comparison Summary")
        print("=" * 60)
        
        # Calculate statistics for each model
        models = {
            'Black-Scholes': 'bs',
            'Monte Carlo': 'mc', 
            'Binomial Tree': 'bt'
        }
        
        for model_name, model_code in models.items():
            mae = comparison_data[f'{model_code}_difference'].abs().mean()
            mape = comparison_data[f'{model_code}_percentage'].abs().mean()
            correlation = comparison_data['market_price'].corr(comparison_data[f'{model_code}_price'])
            
            print(f"{model_name}:")
            print(f"  Mean Absolute Error: ${mae:.4f}")
            print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
            print(f"  Correlation with Market: {correlation:.4f}")
            print()
        
        print(f"Market Data Summary:")
        print(f"  Number of Options: {len(comparison_data)}")
        print(f"  Price Range: ${comparison_data['market_price'].min():.2f} - ${comparison_data['market_price'].max():.2f}")
        print(f"  Strike Range: ${comparison_data['strike'].min():.2f} - ${comparison_data['strike'].max():.2f}")


class GreeksAnalyzer:
    """Analyze option Greeks"""
    
    def __init__(self, S0: float, r: float = 0.05):
        """
        Initialize Greeks analyzer
        
        Args:
            S0: Current stock price
            r: Risk-free rate
        """
        self.S0 = S0
        self.r = r
    
    def calculate_greeks_for_options(self, options_data: pd.DataFrame, sigma: float = 0.25) -> pd.DataFrame:
        """
        Calculate Greeks for a set of options
        
        Args:
            options_data: DataFrame with option data
            sigma: Volatility
            
        Returns:
            DataFrame with Greeks
        """
        if options_data.empty:
            return pd.DataFrame()
        
        greeks_data = []
        
        for _, row in options_data.iterrows():
            K = row['strike']
            T = row['time_to_expiry']
            option_type = row['option_type']
            
            # Calculate Greeks using Black-Scholes
            bs = BlackScholes(self.S0, K, T, self.r, sigma)
            greeks = bs.calculate_greeks()
            
            if option_type.lower() == 'call':
                delta = greeks['delta_call']
                theta = greeks['theta_call']
                rho = greeks['rho_call']
            else:
                delta = greeks['delta_put']
                theta = greeks['theta_put']
                rho = greeks['rho_put']
            
            greeks_data.append({
                'strike': K,
                'option_type': option_type,
                'delta': delta,
                'gamma': greeks['gamma'],
                'vega': greeks['vega'],
                'theta': theta,
                'rho': rho,
                'sigma': sigma
            })
        
        return pd.DataFrame(greeks_data)
    
    def plot_greeks(self, greeks_data: pd.DataFrame):
        """Plot Greeks analysis"""
        if greeks_data.empty:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Option Greeks Analysis', fontsize=16)
        
        # Separate calls and puts
        calls = greeks_data[greeks_data['option_type'] == 'call']
        puts = greeks_data[greeks_data['option_type'] == 'put']
        
        # 1. Delta
        if not calls.empty:
            axes[0, 0].plot(calls['strike'], calls['delta'], 'o-', label='Calls', color='blue')
        if not puts.empty:
            axes[0, 0].plot(puts['strike'], puts['delta'], 's-', label='Puts', color='red')
        axes[0, 0].set_xlabel('Strike Price')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Delta vs Strike')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gamma
        all_options = pd.concat([calls, puts])
        if not all_options.empty:
            axes[0, 1].plot(all_options['strike'], all_options['gamma'], 'o-', color='green')
        axes[0, 1].set_xlabel('Strike Price')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Gamma vs Strike')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Vega
        if not all_options.empty:
            axes[0, 2].plot(all_options['strike'], all_options['vega'], 'o-', color='purple')
        axes[0, 2].set_xlabel('Strike Price')
        axes[0, 2].set_ylabel('Vega')
        axes[0, 2].set_title('Vega vs Strike')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Theta
        if not calls.empty:
            axes[1, 0].plot(calls['strike'], calls['theta'], 'o-', label='Calls', color='blue')
        if not puts.empty:
            axes[1, 0].plot(puts['strike'], puts['theta'], 's-', label='Puts', color='red')
        axes[1, 0].set_xlabel('Strike Price')
        axes[1, 0].set_ylabel('Theta')
        axes[1, 0].set_title('Theta vs Strike')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Rho
        if not calls.empty:
            axes[1, 1].plot(calls['strike'], calls['rho'], 'o-', label='Calls', color='blue')
        if not puts.empty:
            axes[1, 1].plot(puts['strike'], puts['rho'], 's-', label='Puts', color='red')
        axes[1, 1].set_xlabel('Strike Price')
        axes[1, 1].set_ylabel('Rho')
        axes[1, 1].set_title('Rho vs Strike')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Greeks summary
        if not all_options.empty:
            greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
            greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
            greek_ranges = []
            
            for col in greek_cols:
                greek_range = all_options[col].max() - all_options[col].min()
                greek_ranges.append(greek_range)
            
            axes[1, 2].bar(greek_names, greek_ranges, color=['blue', 'green', 'purple', 'orange', 'red'])
            axes[1, 2].set_ylabel('Range')
            axes[1, 2].set_title('Greeks Sensitivity Range')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class VolatilityAnalyzer:
    """Analyze volatility patterns"""
    
    def __init__(self, S0: float, r: float = 0.05):
        """
        Initialize volatility analyzer
        
        Args:
            S0: Current stock price
            r: Risk-free rate
        """
        self.S0 = S0
        self.r = r
    
    def calculate_implied_volatility(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied volatility for options
        
        Args:
            market_data: DataFrame with market option data
            
        Returns:
            DataFrame with implied volatilities
        """
        if market_data.empty:
            return pd.DataFrame()
        
        implied_vols = []
        
        for _, row in market_data.iterrows():
            try:
                K = row['strike']
                T = row['time_to_expiry']
                market_price = row['lastPrice']
                option_type = row['option_type']
                
                # Use Newton-Raphson to find implied volatility
                sigma = self._newton_raphson_iv(K, T, market_price, option_type)
                implied_vols.append(sigma)
            except:
                implied_vols.append(np.nan)
        
        result = market_data.copy()
        result['implied_volatility'] = implied_vols
        return result
    
    def _newton_raphson_iv(self, K: float, T: float, market_price: float, 
                           option_type: str, max_iter: int = 100) -> float:
        """
        Newton-Raphson method to find implied volatility
        
        Args:
            K: Strike price
            T: Time to expiry
            market_price: Market option price
            option_type: "call" or "put"
            max_iter: Maximum iterations
            
        Returns:
            Implied volatility
        """
        from scipy.stats import norm
        
        sigma = 0.3  # Initial guess
        
        for _ in range(max_iter):
            # Calculate option price and vega
            d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-self.r * T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
            
            vega = self.S0 * np.sqrt(T) * norm.pdf(d1)
            
            # Update sigma
            sigma_new = sigma - (price - market_price) / vega
            
            if abs(sigma_new - sigma) < 1e-6:
                break
            
            sigma = sigma_new
        
        return sigma
    
    def plot_volatility_smile(self, volatility_data: pd.DataFrame):
        """Plot volatility smile"""
        if volatility_data.empty:
            print("No data to plot")
            return
        
        # Separate calls and puts
        calls = volatility_data[volatility_data['option_type'] == 'call']
        puts = volatility_data[volatility_data['option_type'] == 'put']
        
        plt.figure(figsize=(12, 8))
        
        if not calls.empty:
            plt.scatter(calls['strike'], calls['implied_volatility'], 
                       alpha=0.7, label='Calls', color='blue', s=50)
        
        if not puts.empty:
            plt.scatter(puts['strike'], puts['implied_volatility'], 
                       alpha=0.7, label='Puts', color='red', s=50)
        
        plt.axvline(x=self.S0, color='black', linestyle='--', alpha=0.7, label='Current Price')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title('Volatility Smile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_volatility_patterns(self, volatility_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility patterns
        
        Args:
            volatility_data: DataFrame with implied volatilities
            
        Returns:
            Dictionary with volatility analysis
        """
        if volatility_data.empty:
            return {}
        
        # Calculate moneyness
        volatility_data['moneyness'] = volatility_data['strike'] / self.S0
        
        # Separate by moneyness
        itm = volatility_data[volatility_data['moneyness'] < 0.95]  # In-the-money
        atm = volatility_data[(volatility_data['moneyness'] >= 0.95) & (volatility_data['moneyness'] <= 1.05)]  # At-the-money
        otm = volatility_data[volatility_data['moneyness'] > 1.05]  # Out-of-the-money
        
        analysis = {
            'itm_vol': itm['implied_volatility'].mean() if not itm.empty else np.nan,
            'atm_vol': atm['implied_volatility'].mean() if not atm.empty else np.nan,
            'otm_vol': otm['implied_volatility'].mean() if not otm.empty else np.nan,
            'vol_skew': otm['implied_volatility'].mean() - itm['implied_volatility'].mean() if not otm.empty and not itm.empty else np.nan,
            'total_options': len(volatility_data),
            'itm_count': len(itm),
            'atm_count': len(atm),
            'otm_count': len(otm)
        }
        
        return analysis 