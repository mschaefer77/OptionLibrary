"""
Price Simulation Module

This module provides tools for simulating underlying asset prices using three core dynamics models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .dynamics import (
    GeometricBrownianMotion, HestonModel, MertonJumpDiffusion,
    create_dynamics_model
)


class PriceSimulator:
    """Price simulator with three core dynamics models"""
    
    def __init__(self, dynamics_model):
        """
        Initialize price simulator
        
        Args:
            dynamics_model: Dynamics model instance (from dynamics.py)
        """
        self.dynamics_model = dynamics_model
        self.model_type = dynamics_model.get_parameters()['model']
    
    def simulate_paths(self, S0: float, T: float, n_steps: int = 252, 
                      n_paths: int = 1, return_time: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate price paths using the dynamics model
        
        Args:
            S0: Initial price
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            return_time: Whether to return time points
            
        Returns:
            Price paths array and optionally time points
        """
        if self.model_type == 'Heston':
            S, v, t = self.dynamics_model.simulate(S0, v0=0.04, T=T, n_steps=n_steps, n_paths=n_paths)
            if return_time:
                return S, v, t
            else:
                return S, v
        else:
            # GBM and Merton models return (S, t)
            S, t = self.dynamics_model.simulate(S0, T, n_steps, n_paths)
            if return_time:
                return S, t
            else:
                return S
    
    def plot_paths(self, S: np.ndarray, t: np.ndarray = None, 
                   title: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot simulated price paths
        
        Args:
            S: Price paths array
            t: Time points (optional)
            title: Plot title
            figsize: Figure size
        """
        if t is None:
            t = np.linspace(0, 1, S.shape[1])
        
        plt.figure(figsize=figsize)
        
        if S.shape[0] == 1:
            plt.plot(t, S[0], 'b-', linewidth=2, label='Price Path')
        else:
            for i in range(min(10, S.shape[0])):  # Plot first 10 paths
                plt.plot(t, S[i], alpha=0.7, linewidth=1)
            plt.plot(t, np.mean(S, axis=0), 'r-', linewidth=2, label='Mean Path')
        
        plt.xlabel('Time (years)')
        plt.ylabel('Price')
        plt.title(title or f'{self.model_type} Price Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_statistics(self, S: np.ndarray, t: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate statistics from simulated paths
        
        Args:
            S: Price paths array
            t: Time points (optional)
            
        Returns:
            Dictionary with statistics
        """
        if t is None:
            t = np.linspace(0, 1, S.shape[1])
        
        # Calculate returns
        returns = np.diff(S, axis=1) / S[:, :-1]
        
        stats = {
            'model_type': self.model_type,
            'n_paths': S.shape[0],
            'n_steps': S.shape[1],
            'final_prices': S[:, -1],
            'mean_final_price': np.mean(S[:, -1]),
            'std_final_price': np.std(S[:, -1]),
            'min_price': np.min(S),
            'max_price': np.max(S),
            'mean_returns': np.mean(returns, axis=0),
            'volatility': np.std(returns, axis=0) * np.sqrt(252),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        }
        
        return stats


def simulate_paths(dynamics_model, S0: float, T: float, n_steps: int = 252, 
                  n_paths: int = 1, plot: bool = False) -> Dict[str, Any]:
    """
    Convenience function to simulate paths
    
    Args:
        dynamics_model: Dynamics model instance
        S0: Initial price
        T: Time horizon (years)
        n_steps: Number of time steps
        n_paths: Number of paths
        plot: Whether to plot the paths
        
    Returns:
        Dictionary with simulation results
    """
    simulator = PriceSimulator(dynamics_model)
    
    if dynamics_model.get_parameters()['model'] == 'Heston':
        S, v, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        if plot:
            simulator.plot_paths(S, t)
        return {
            'prices': S,
            'variance': v,
            'time': t,
            'statistics': simulator.get_statistics(S, t)
        }
    else:
        S, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        if plot:
            simulator.plot_paths(S, t)
        return {
            'prices': S,
            'time': t,
            'statistics': simulator.get_statistics(S, t)
        }


def generate_price_data(dynamics_model, S0: float, T: float, n_steps: int = 252, 
                       n_paths: int = 1, frequency: str = 'D') -> pd.DataFrame:
    """
    Generate price data as a pandas DataFrame
    
    Args:
        dynamics_model: Dynamics model instance
        S0: Initial price
        T: Time horizon (years)
        n_steps: Number of time steps
        n_paths: Number of paths
        frequency: Data frequency ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        DataFrame with price data
    """
    simulator = PriceSimulator(dynamics_model)
    
    if dynamics_model.get_parameters()['model'] == 'Heston':
        S, v, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        
        # Create date range
        start_date = datetime.now()
        dates = pd.date_range(start=start_date, periods=len(t), freq=frequency)
        
        # Create DataFrame
        data = []
        for i in range(n_paths):
            for j, (date, price, variance) in enumerate(zip(dates, S[i], v[i])):
                data.append({
                    'date': date,
                    'path': i,
                    'price': price,
                    'variance': variance,
                    'time': t[j]
                })
        
        return pd.DataFrame(data)
    
    else:
        S, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        
        # Create date range
        start_date = datetime.now()
        dates = pd.date_range(start=start_date, periods=len(t), freq=frequency)
        
        # Create DataFrame
        data = []
        for i in range(n_paths):
            for j, (date, price) in enumerate(zip(dates, S[i])):
                data.append({
                    'date': date,
                    'path': i,
                    'price': price,
                    'time': t[j]
                })
        
        return pd.DataFrame(data)


def compare_dynamics_models(S0: float = 100, T: float = 1.0, n_steps: int = 252, 
                           n_paths: int = 5) -> Dict[str, Any]:
    """
    Compare different dynamics models
    
    Args:
        S0: Initial price
        T: Time horizon
        n_steps: Number of time steps
        n_paths: Number of paths per model
        
    Returns:
        Dictionary with comparison results
    """
    models = {
        'GBM': create_dynamics_model('GBM', mu=0.05, sigma=0.2),
        'Heston': create_dynamics_model('Heston', mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7),
        'Merton': create_dynamics_model('Merton', mu=0.05, sigma=0.2, lambda_jump=0.1, mu_jump=-0.1, sigma_jump=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        simulator = PriceSimulator(model)
        if name == 'Heston':
            S, v, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        else:
            S, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        stats = simulator.get_statistics(S, t)
        results[name] = {
            'prices': S,
            'time': t,
            'statistics': stats
        }
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)
        
        S = result['prices']
        t = result['time']
        
        for j in range(min(3, S.shape[0])):
            plt.plot(t, S[j], alpha=0.7, linewidth=1)
        
        plt.plot(t, np.mean(S, axis=0), 'r-', linewidth=2, label='Mean')
        plt.title(f'{name} Model')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results 