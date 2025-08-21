"""
Option Pricing Models

This module provides various option pricing methods:
- Black-Scholes (analytical) - GBM only
- Monte Carlo simulation - Any dynamics model
- Binomial tree - Any dynamics model  
- Trinomial tree - Any dynamics model
- Finite difference - Any dynamics model
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import warnings
warnings.filterwarnings('ignore')


class BlackScholes:
    """Black-Scholes option pricing model (GBM dynamics only)"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize Black-Scholes model
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def price_call(self) -> float:
        """Price European call option"""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def price_put(self) -> float:
        """Price European put option"""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
    
    def calculate_greeks(self) -> Dict[str, float]:
        """Calculate option Greeks"""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        # Delta
        delta_call = norm.cdf(d1)
        delta_put = delta_call - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))
        
        # Vega (same for calls and puts)
        vega = self.S0 * np.sqrt(self.T) * norm.pdf(d1)
        
        # Theta
        theta_call = -self.S0 * self.sigma * norm.pdf(d1) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        theta_put = -self.S0 * self.sigma * norm.pdf(d1) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        # Rho
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        return {
            'delta_call': delta_call,
            'delta_put': delta_put,
            'gamma': gamma,
            'vega': vega,
            'theta_call': theta_call,
            'theta_put': theta_put,
            'rho_call': rho_call,
            'rho_put': rho_put
        }


class MonteCarloPricer:
    """Monte Carlo option pricing with any dynamics model"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, dynamics_model=None):
        """
        Initialize Monte Carlo pricer
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            dynamics_model: Dynamics model instance (if None, uses GBM with sigma=0.2)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.dynamics_model = dynamics_model
        
        # Default GBM parameters if no dynamics model provided
        if dynamics_model is None:
            self.sigma = 0.2
        else:
            self.sigma = dynamics_model.get_parameters().get('sigma', 0.2)
    
    def price_call(self, n_simulations: int = 10000, n_steps: int = 252) -> Dict[str, float]:
        """Price European call option using Monte Carlo"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            Z = np.random.normal(0, 1, (n_simulations, n_steps))
            returns = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
            paths = self.S0 * np.exp(np.cumsum(returns, axis=1))
        else:
            # Use provided dynamics model
            if self.dynamics_model.get_parameters()['model'] == 'Heston':
                paths, _, _ = self.dynamics_model.simulate(self.S0, v0=0.04, T=self.T, n_steps=n_steps, n_paths=n_simulations)
            else:
                paths, _ = self.dynamics_model.simulate(self.S0, self.T, n_steps, n_simulations)
        
        # Calculate payoffs
        payoffs = np.maximum(paths[:, -1] - self.K, 0)
        
        # Discount to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    def price_put(self, n_simulations: int = 10000, n_steps: int = 252) -> Dict[str, float]:
        """Price European put option using Monte Carlo"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            Z = np.random.normal(0, 1, (n_simulations, n_steps))
            returns = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
            paths = self.S0 * np.exp(np.cumsum(returns, axis=1))
        else:
            # Use provided dynamics model
            if self.dynamics_model.get_parameters()['model'] == 'Heston':
                paths, _, _ = self.dynamics_model.simulate(self.S0, v0=0.04, T=self.T, n_steps=n_steps, n_paths=n_simulations)
            else:
                paths, _ = self.dynamics_model.simulate(self.S0, self.T, n_steps, n_simulations)
        
        # Calculate payoffs
        payoffs = np.maximum(self.K - paths[:, -1], 0)
        
        # Discount to present value
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - 1.96 * std_error, price + 1.96 * std_error)
        }
    
    

class BinomialTree:
    """Binomial tree option pricing with any dynamics model"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, dynamics_model=None):
        """
        Initialize binomial tree
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            dynamics_model: Dynamics model instance (if None, uses GBM with sigma=0.2)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.dynamics_model = dynamics_model
        
        # Default GBM parameters if no dynamics model provided
        if dynamics_model is None:
            self.sigma = 0.2
        else:
            self.sigma = dynamics_model.get_parameters().get('sigma', 0.2)
    
    def price_call(self, n_steps: int = 100) -> float:
        """Price European call option using binomial tree"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            u = np.exp(self.sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(self.r * dt) - d) / (u - d)
            
            # Initialize option values at maturity
            option_values = np.zeros(n_steps + 1)
            for i in range(n_steps + 1):
                S = self.S0 * (u ** (n_steps - i)) * (d ** i)
                option_values[i] = max(S - self.K, 0)
            
            # Backward induction
            for step in range(n_steps - 1, -1, -1):
                for i in range(step + 1):
                    option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            return option_values[0]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_call(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_put(self, n_steps: int = 100) -> float:
        """Price European put option using binomial tree"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            u = np.exp(self.sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(self.r * dt) - d) / (u - d)
            
            # Initialize option values at maturity
            option_values = np.zeros(n_steps + 1)
            for i in range(n_steps + 1):
                S = self.S0 * (u ** (n_steps - i)) * (d ** i)
                option_values[i] = max(self.K - S, 0)
            
            # Backward induction
            for step in range(n_steps - 1, -1, -1):
                for i in range(step + 1):
                    option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            return option_values[0]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_put(n_simulations=10000, n_steps=n_steps)['price']


class TrinomialTree:
    """Trinomial tree option pricing with any dynamics model"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, dynamics_model=None):
        """
        Initialize trinomial tree
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            dynamics_model: Dynamics model instance (if None, uses GBM with sigma=0.2)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.dynamics_model = dynamics_model
        
        # Default GBM parameters if no dynamics model provided
        if dynamics_model is None:
            self.sigma = 0.2
        else:
            self.sigma = dynamics_model.get_parameters().get('sigma', 0.2)
    
    def price_call(self, n_steps: int = 100) -> float:
        """Price European call option using trinomial tree"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            u = np.exp(self.sigma * np.sqrt(3 * dt))
            d = 1 / u
            m = 1
            
            # Risk-neutral probabilities
            pu = ((np.exp(self.r * dt) - d) * (u - np.exp(self.r * dt))) / ((u - d) * (u - m))
            pd = ((u - np.exp(self.r * dt)) * (np.exp(self.r * dt) - m)) / ((u - d) * (m - d))
            pm = 1 - pu - pd
            
            # Initialize option values at maturity
            option_values = np.zeros(2 * n_steps + 1)
            for i in range(2 * n_steps + 1):
                S = self.S0 * (u ** (n_steps - i)) * (m ** max(0, n_steps - i)) * (d ** max(0, i - n_steps))
                option_values[i] = max(S - self.K, 0)
            
            # Backward induction
            for step in range(n_steps - 1, -1, -1):
                for i in range(2 * step + 1):
                    option_values[i] = np.exp(-self.r * dt) * (pu * option_values[i] + pm * option_values[i + 1] + pd * option_values[i + 2])
            
            return option_values[0]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_call(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_put(self, n_steps: int = 100) -> float:
        """Price European put option using trinomial tree"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            u = np.exp(self.sigma * np.sqrt(3 * dt))
            d = 1 / u
            m = 1
            
            # Risk-neutral probabilities
            pu = ((np.exp(self.r * dt) - d) * (u - np.exp(self.r * dt))) / ((u - d) * (u - m))
            pd = ((u - np.exp(self.r * dt)) * (np.exp(self.r * dt) - m)) / ((u - d) * (m - d))
            pm = 1 - pu - pd
            
            # Initialize option values at maturity
            option_values = np.zeros(2 * n_steps + 1)
            for i in range(2 * n_steps + 1):
                S = self.S0 * (u ** (n_steps - i)) * (m ** max(0, n_steps - i)) * (d ** max(0, i - n_steps))
                option_values[i] = max(self.K - S, 0)
            
            # Backward induction
            for step in range(n_steps - 1, -1, -1):
                for i in range(2 * step + 1):
                    option_values[i] = np.exp(-self.r * dt) * (pu * option_values[i] + pm * option_values[i + 1] + pd * option_values[i + 2])
            
            return option_values[0]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_put(n_simulations=10000, n_steps=n_steps)['price']


class FiniteDifference:
    """Finite difference option pricing with any dynamics model"""
    
    def __init__(self, S0: float, K: float, T: float, r: float, dynamics_model=None):
        """
        Initialize finite difference solver
        
        Args:
            S0: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            dynamics_model: Dynamics model instance (if None, uses GBM with sigma=0.2)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.dynamics_model = dynamics_model
        
        # Default GBM parameters if no dynamics model provided
        if dynamics_model is None:
            self.sigma = 0.2
        else:
            self.sigma = dynamics_model.get_parameters().get('sigma', 0.2)
    
    def price_call_explicit(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European call option using explicit finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(S[i] - self.K, 0)
            
            # Explicit finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 * dt / (dS**2)
                    beta = self.r * S[i] * dt / (2 * dS)
                    gamma = self.r * dt
                    
                    V[n, i] = alpha * V[n + 1, i + 1] + (1 - 2 * alpha - gamma) * V[n + 1, i] + alpha * V[n + 1, i - 1] - beta * (V[n + 1, i + 1] - V[n + 1, i - 1])
                
                # Boundary conditions
                V[n, 0] = 0  # S = 0
                V[n, n_S] = S[n_S] - self.K * np.exp(-self.r * (self.T - n * dt))  # S = S_max
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_call(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_call_implicit(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European call option using implicit finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(S[i] - self.K, 0)
            
            # Implicit finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                # Set up tridiagonal system
                A = np.zeros((n_S + 1, n_S + 1))
                b = np.zeros(n_S + 1)
                
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 / (dS**2)
                    beta = self.r * S[i] / (2 * dS)
                    gamma = self.r
                    
                    A[i, i - 1] = alpha - beta
                    A[i, i] = -2 * alpha - gamma
                    A[i, i + 1] = alpha + beta
                    b[i] = -V[n + 1, i] / dt
                
                # Boundary conditions
                A[0, 0] = 1
                A[n_S, n_S] = 1
                b[0] = 0
                b[n_S] = S[n_S] - self.K * np.exp(-self.r * (self.T - n * dt))
                
                # Solve tridiagonal system
                V[n, :] = np.linalg.solve(A, b)
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_call(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_put_explicit(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European put option using explicit finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(self.K - S[i], 0)
            
            # Explicit finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 * dt / (dS**2)
                    beta = self.r * S[i] * dt / (2 * dS)
                    gamma = self.r * dt
                    
                    V[n, i] = alpha * V[n + 1, i + 1] + (1 - 2 * alpha - gamma) * V[n + 1, i] + alpha * V[n + 1, i - 1] - beta * (V[n + 1, i + 1] - V[n + 1, i - 1])
                
                # Boundary conditions
                V[n, 0] = self.K * np.exp(-self.r * (self.T - n * dt))  # S = 0
                V[n, n_S] = 0  # S = S_max
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_put(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_put_implicit(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European put option using implicit finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(self.K - S[i], 0)
            
            # Implicit finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                # Set up tridiagonal system
                A = np.zeros((n_S + 1, n_S + 1))
                b = np.zeros(n_S + 1)
                
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 / (dS**2)
                    beta = self.r * S[i] / (2 * dS)
                    gamma = self.r
                    
                    A[i, i - 1] = alpha - beta
                    A[i, i] = -2 * alpha - gamma
                    A[i, i + 1] = alpha + beta
                    b[i] = -V[n + 1, i] / dt
                
                # Boundary conditions
                A[0, 0] = 1
                A[n_S, n_S] = 1
                b[0] = self.K * np.exp(-self.r * (self.T - n * dt))  # S = 0
                b[n_S] = 0  # S = S_max
                
                # Solve tridiagonal system
                V[n, :] = np.linalg.solve(A, b)
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_put(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_call_crank_nicolson(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European call option using Crank-Nicolson finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(S[i] - self.K, 0)
            
            # Crank-Nicolson finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                # Set up tridiagonal system for Crank-Nicolson
                A = np.zeros((n_S + 1, n_S + 1))
                b = np.zeros(n_S + 1)
                
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 / (dS**2)
                    beta = self.r * S[i] / (2 * dS)
                    gamma = self.r
                    
                    # Crank-Nicolson: average of implicit and explicit
                    # Left side (n+1/2): implicit
                    A[i, i - 1] = -0.5 * (alpha - beta) * dt
                    A[i, i] = 1 + 0.5 * (2 * alpha + gamma) * dt
                    A[i, i + 1] = -0.5 * (alpha + beta) * dt
                    
                    # Right side (n+1): explicit
                    b[i] = (0.5 * (alpha - beta) * dt) * V[n + 1, i - 1] + \
                           (1 - 0.5 * (2 * alpha + gamma) * dt) * V[n + 1, i] + \
                           (0.5 * (alpha + beta) * dt) * V[n + 1, i + 1]
                
                # Boundary conditions
                A[0, 0] = 1
                A[n_S, n_S] = 1
                b[0] = 0  # S = 0
                b[n_S] = S[n_S] - self.K * np.exp(-self.r * (self.T - n * dt))  # S = S_max
                
                # Solve tridiagonal system
                V[n, :] = np.linalg.solve(A, b)
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_call(n_simulations=10000, n_steps=n_steps)['price']
    
    def price_put_crank_nicolson(self, n_steps: int = 100, n_price_steps: int = 100) -> float:
        """Price European put option using Crank-Nicolson finite difference"""
        if self.dynamics_model is None:
            # Use GBM
            dt = self.T / n_steps
            dS = self.S0 / n_price_steps
            
            # Grid setup
            S_max = 3 * self.S0
            S_min = 0.1 * self.S0
            n_S = int((S_max - S_min) / dS)
            
            # Initialize grid
            S = np.linspace(S_min, S_max, n_S + 1)
            V = np.zeros((n_steps + 1, n_S + 1))
            
            # Boundary conditions at maturity
            for i in range(n_S + 1):
                V[n_steps, i] = max(self.K - S[i], 0)
            
            # Crank-Nicolson finite difference scheme
            for n in range(n_steps - 1, -1, -1):
                # Set up tridiagonal system for Crank-Nicolson
                A = np.zeros((n_S + 1, n_S + 1))
                b = np.zeros(n_S + 1)
                
                for i in range(1, n_S):
                    # Finite difference coefficients
                    alpha = 0.5 * self.sigma**2 * S[i]**2 / (dS**2)
                    beta = self.r * S[i] / (2 * dS)
                    gamma = self.r
                    
                    # Crank-Nicolson: average of implicit and explicit
                    # Left side (n+1/2): implicit
                    A[i, i - 1] = -0.5 * (alpha - beta) * dt
                    A[i, i] = 1 + 0.5 * (2 * alpha + gamma) * dt
                    A[i, i + 1] = -0.5 * (alpha + beta) * dt
                    
                    # Right side (n+1): explicit
                    b[i] = (0.5 * (alpha - beta) * dt) * V[n + 1, i - 1] + \
                           (1 - 0.5 * (2 * alpha + gamma) * dt) * V[n + 1, i] + \
                           (0.5 * (alpha + beta) * dt) * V[n + 1, i + 1]
                
                # Boundary conditions
                A[0, 0] = 1
                A[n_S, n_S] = 1
                b[0] = self.K * np.exp(-self.r * (self.T - n * dt))  # S = 0
                b[n_S] = 0  # S = S_max
                
                # Solve tridiagonal system
                V[n, :] = np.linalg.solve(A, b)
            
            # Interpolate to get option value at S0
            idx = np.searchsorted(S, self.S0)
            if idx == 0:
                return V[0, 0]
            elif idx == len(S):
                return V[0, n_S]
            else:
                # Linear interpolation
                w = (self.S0 - S[idx - 1]) / (S[idx] - S[idx - 1])
                return (1 - w) * V[0, idx - 1] + w * V[0, idx]
        else:
            # Use Monte Carlo with dynamics model for non-GBM
            mc = MonteCarloPricer(self.S0, self.K, self.T, self.r, self.dynamics_model)
            return mc.price_put(n_simulations=10000, n_steps=n_steps)['price']


def compare_pricing_models(S0: float, K: float, T: float, r: float, dynamics_model=None) -> Dict[str, float]:
    """
    Compare different pricing models
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        dynamics_model: Dynamics model instance (if None, uses GBM with sigma=0.2)
        
    Returns:
        Dictionary with results from different models
    """
    results = {}
    
    # Black-Scholes (GBM only)
    if dynamics_model is None or (hasattr(dynamics_model, 'get_parameters') and dynamics_model.get_parameters()['model'] == 'GBM'):
        sigma = dynamics_model.get_parameters().get('sigma', 0.2) if dynamics_model else 0.2
        bs = BlackScholes(S0, K, T, r, sigma)
        results['Black-Scholes'] = {
            'call': bs.price_call(),
            'put': bs.price_put(),
            'greeks': bs.calculate_greeks()
        }
    
    # Monte Carlo
    mc = MonteCarloPricer(S0, K, T, r, dynamics_model)
    mc_call = mc.price_call()
    mc_put = mc.price_put()
    results['Monte Carlo'] = {
        'call': mc_call['price'],
        'put': mc_put['price'],
        'call_std_error': mc_call['std_error'],
        'put_std_error': mc_put['std_error']
    }
    
    # Binomial Tree
    bt = BinomialTree(S0, K, T, r, dynamics_model)
    results['Binomial Tree'] = {
        'call': bt.price_call(),
        'put': bt.price_put()
    }
    
    # Trinomial Tree
    tt = TrinomialTree(S0, K, T, r, dynamics_model)
    results['Trinomial Tree'] = {
        'call': tt.price_call(),
        'put': tt.price_put()
    }
    
    # Finite Difference (Explicit)
    fd_explicit = FiniteDifference(S0, K, T, r, dynamics_model)
    results['Finite Difference (Explicit)'] = {
        'call': fd_explicit.price_call_explicit(),
        'put': fd_explicit.price_put_explicit()
    }
    
    # Finite Difference (Implicit)
    fd_implicit = FiniteDifference(S0, K, T, r, dynamics_model)
    results['Finite Difference (Implicit)'] = {
        'call': fd_implicit.price_call_implicit(),
        'put': fd_implicit.price_put_implicit()
    }
    
    # Finite Difference (Crank-Nicolson)
    fd_cn = FiniteDifference(S0, K, T, r, dynamics_model)
    results['Finite Difference (Crank-Nicolson)'] = {
        'call': fd_cn.price_call_crank_nicolson(),
        'put': fd_cn.price_put_crank_nicolson()
    }
    
    return results 