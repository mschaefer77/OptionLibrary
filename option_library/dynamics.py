"""
Underlying Price Dynamics Models

This module provides three core stochastic processes for modeling underlying asset prices:
- Geometric Brownian Motion (GBM) - Standard Black-Scholes model
- Heston Model (Stochastic Volatility) - Stochastic volatility with mean reversion
- Merton Jump-Diffusion - Continuous diffusion with random jumps
"""

import numpy as np
from scipy.stats import norm, expon
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings
warnings.filterwarnings('ignore')


class GeometricBrownianMotion:
    """Geometric Brownian Motion (Black-Scholes model)"""
    
    def __init__(self, mu: float, sigma: float):
        """
        Initialize GBM model
        
        Args:
            mu: Drift parameter (annualized)
            sigma: Volatility parameter (annualized)
        """
        self.mu = mu
        self.sigma = sigma
    
    def simulate(self, S0: float, T: float, n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate GBM paths
        
        Args:
            S0: Initial price
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, time_points)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Generate random increments
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Calculate price paths
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        
        # Cumulative sum of returns
        returns = drift + diffusion
        log_prices = np.log(S0) + np.cumsum(returns, axis=1)
        
        # Add initial price
        prices = np.exp(log_prices)
        prices = np.insert(prices, 0, S0, axis=1)
        
        return prices, t
    
    def get_parameters(self) -> Dict[str, float]:
        """Get model parameters"""
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'model': 'GBM'
        }


class HestonModel:
    """Heston Stochastic Volatility Model"""
    
    def __init__(self, mu: float, kappa: float, theta: float, sigma_v: float, rho: float):
        """
        Initialize Heston model
        
        Args:
            mu: Drift parameter
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma_v: Volatility of volatility
            rho: Correlation between price and volatility
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
    
    def simulate(self, S0: float, v0: float, T: float, n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths
        
        Args:
            S0: Initial price
            v0: Initial variance
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, variance_paths, time_points)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, (n_paths, n_steps))
        Z2 = np.random.normal(0, 1, (n_paths, n_steps))
        Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        for i in range(n_steps):
            # Variance process
            v[:, i + 1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + 
                self.sigma_v * np.sqrt(v[:, i]) * np.sqrt(dt) * Z2[:, i],
                0.0001  # Ensure positive variance
            )
            
            # Price process
            drift = (self.mu - 0.5 * v[:, i]) * dt
            diffusion = np.sqrt(v[:, i]) * np.sqrt(dt) * Z1[:, i]
            
            S[:, i + 1] = S[:, i] * np.exp(drift + diffusion)
        
        return S, v, t
    
    def get_parameters(self) -> Dict[str, float]:
        """Get model parameters"""
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma_v': self.sigma_v,
            'rho': self.rho,
            'model': 'Heston'
        }


class MertonJumpDiffusion:
    """Merton Jump-Diffusion Model"""
    
    def __init__(self, mu: float, sigma: float, lambda_jump: float, mu_jump: float, sigma_jump: float):
        """
        Initialize Merton jump-diffusion model
        
        Args:
            mu: Drift parameter
            sigma: Volatility parameter
            lambda_jump: Jump intensity (jumps per year)
            mu_jump: Mean jump size
            sigma_jump: Jump size volatility
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
    
    def simulate(self, S0: float, T: float, n_steps: int, n_paths: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate jump-diffusion paths
        
        Args:
            S0: Initial price
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, time_points)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Initialize price array
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        
        # Generate random numbers
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Jump parameters
        jump_prob = self.lambda_jump * dt
        
        for i in range(n_steps):
            # Check for jumps
            jumps = np.random.poisson(jump_prob, n_paths)
            jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, n_paths)
            
            # Calculate total jump effect
            jump_effect = np.sum(jumps[:, np.newaxis] * jump_sizes[:, np.newaxis], axis=1)
            
            # Diffusion component
            drift = (self.mu - 0.5 * self.sigma**2 - self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1)) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z[:, i]
            
            # Update prices
            S[:, i + 1] = S[:, i] * np.exp(drift + diffusion + jump_effect)
        
        return S, t
    
    def get_parameters(self) -> Dict[str, float]:
        """Get model parameters"""
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'lambda_jump': self.lambda_jump,
            'mu_jump': self.mu_jump,
            'sigma_jump': self.sigma_jump,
            'model': 'Merton'
        }


def create_dynamics_model(model_type: str, **kwargs) -> Union[GeometricBrownianMotion, HestonModel, MertonJumpDiffusion]:
    """
    Factory function to create dynamics models
    
    Args:
        model_type: Type of model ('GBM', 'Heston', 'Merton')
        **kwargs: Model-specific parameters
        
    Returns:
        Dynamics model instance
    """
    if model_type.upper() == 'GBM':
        return GeometricBrownianMotion(kwargs.get('mu', 0.05), kwargs.get('sigma', 0.2))
    
    elif model_type.upper() == 'HESTON':
        return HestonModel(
            kwargs.get('mu', 0.05),
            kwargs.get('kappa', 2.0),
            kwargs.get('theta', 0.04),
            kwargs.get('sigma_v', 0.3),
            kwargs.get('rho', -0.7)
        )
    
    elif model_type.upper() == 'MERTON':
        return MertonJumpDiffusion(
            kwargs.get('mu', 0.05),
            kwargs.get('sigma', 0.2),
            kwargs.get('lambda_jump', 0.1),
            kwargs.get('mu_jump', -0.1),
            kwargs.get('sigma_jump', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported models: 'GBM', 'Heston', 'Merton'") 