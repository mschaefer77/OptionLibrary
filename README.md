#Option Pricing Library

> **An extension of my Numerical Methods for Option Pricing class** - implementing the theoretical models covered in coursework into a practical, computational library.

##Overview

This library provides option analysis tools include

## Features

- **Three Core Dynamics Models**: Geometric Brownian Motion (GBM), Heston Stochastic Volatility, Merton Jump-Diffusion
- **Multiple Pricing Methods**: Black-Scholes, Monte Carlo, Binomial Trees, Trinomial Trees, Finite Difference Schemes
- **Advanced Numerical Methods**: Explicit, Implicit, and Crank-Nicolson finite difference schemes
- **Real Market Data Integration**: YFinance API for option chain analysis
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho
- **Volatility Analysis**: Implied volatility calculation and volatility skew visualization

##Mathematical Foundation

### Underlying Dynamics

#### Geometric Brownian Motion (GBM)
The classic Black-Scholes model with constant volatility:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

where:
- $S_t$ = asset price at time $t$
- $\mu$ = drift rate
- $\sigma$ = volatility
- $W_t$ = Wiener process

#### Heston Stochastic Volatility Model
$$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1$$
$$dv_t = \kappa(\theta - v_t)dt + \sigma_v \sqrt{v_t} dW_t^2$$

where:
- $v_t$ = instantaneous variance
- $\kappa$ = mean reversion speed
- $\theta$ = long-term variance
- $\sigma_v$ = volatility of volatility
- $dW_t^1, dW_t^2$ = correlated Wiener processes

#### Merton Jump-Diffusion Model
$$dS_t = (\mu - \lambda \bar{J})S_t dt + \sigma S_t dW_t + (J-1)S_t dN_t$$

where:
- $\lambda$ = jump intensity
- $J$ = jump size (lognormally distributed)
- $\bar{J} = E[J-1]$ = expected jump size
- $N_t$ = Poisson process

### Pricing Methods

#### Black-Scholes Formula
For call options:
$$C(S_t, K, T) = S_t N(d_1) - Ke^{-rT}N(d_2)$$

For put options:
$$P(S_t, K, T) = Ke^{-rT}N(-d_2) - S_t N(-d_1)$$

where:
$$d_1 = \frac{\ln(S_t/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

#### Monte Carlo Simulation
$$V(S_0, K, T) = e^{-rT} \mathbb{E}[\max(S_T - K, 0)]$$

#### Finite Difference Schemes
Solving the Black-Scholes PDE:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**Discretization Approach:**
We discretize the continuous PDE using finite differences on a grid:
- **S-grid**: Asset price points from $S_{min}$ to $S_{max}$
- **t-grid**: Time points from $t=0$ to $t=T$
- **Grid spacing**: $\Delta S$ for price, $\Delta t$ for time

**1. Explicit Finite Difference Scheme:**
Forward Euler method in time, central differences in space:
$$\frac{V_{i,j+1} - V_{i,j}}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{\Delta S^2} + rS_i\frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} - rV_{i,j} = 0$$

**Advantages:**
- Simple implementation
- Direct solution at each time step
- No matrix inversion required

**Disadvantages:**
- Conditional stability: requires $\Delta t \leq \frac{\Delta S^2}{2\sigma^2S_{max}^2}$
- First-order accuracy in time

**2. Implicit Finite Difference Scheme:**
Backward Euler method in time, central differences in space:
$$\frac{V_{i,j+1} - V_{i,j}}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{V_{i+1,j+1} - 2V_{i,j+1} + V_{i-1,j+1}}{\Delta S^2} + rS_i\frac{V_{i+1,j+1} - V_{i-1,j+1}}{2\Delta S} - rV_{i,j+1} = 0$$

**Advantages:**
- Unconditionally stable
- Better stability properties

**Disadvantages:**
- Requires solving tridiagonal system at each time step
- First-order accuracy in time
- More computationally intensive

**3. Crank-Nicolson Scheme:**
Averaging explicit and implicit schemes for second-order accuracy:
$$\frac{V_{i,j+1} - V_{i,j}}{\Delta t} + \frac{1}{2}\sigma^2S_i^2\frac{1}{2}\left[\frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{\Delta S^2} + \frac{V_{i+1,j+1} - 2V_{i,j+1} + V_{i-1,j+1}}{\Delta S^2}\right] + rS_i\frac{1}{2}\left[\frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} + \frac{V_{i+1,j+1} - V_{i-1,j+1}}{2\Delta S}\right] - r\frac{V_{i,j} + V_{i,j+1}}{2} = 0$$

**Advantages:**
- Second-order accuracy in both time and space
- Unconditionally stable
- Better convergence properties

**Disadvantages:**
- More complex implementation
- Requires solving tridiagonal system
- Slightly more computationally intensive than implicit

**Boundary Conditions:**
- **At $S = 0$**: $V(0,t) = 0$ for calls, $V(0,t) = Ke^{-r(T-t)}$ for puts
- **At $S = S_{max}$**: $V(S_{max},t) = S_{max} - Ke^{-r(T-t)}$ for calls, $V(S_{max},t) = 0$ for puts
- **At $t = T$**: $V(S,T) = \max(S-K, 0)$ for calls, $V(S,T) = \max(K-S, 0)$ for puts

**Implementation Notes:**
- All schemes handle both call and put options
- Non-GBM dynamics use Monte Carlo fallback
- Grid refinement improves accuracy but increases computation time

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/optiontracka.git
cd optiontracka

# Install dependencies
pip install -r requirements.txt

# Install the library
pip install -e .
```

### Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
yfinance>=0.1.70
scikit-learn>=1.0.0
```


## Running Demos 

This repo contains a demo.ipynb file which containes individual demonstrations of the majority of the libaries main functions
