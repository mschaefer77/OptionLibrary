#!/usr/bin/env python3
"""
Configuration File for Option Pricing Library

Easy parameter adjustment without changing code.
"""

# =============================================================================
# 📊 OPTION PARAMETERS
# =============================================================================
OPTION_PARAMS = {
    'S0': 100,        # Current stock price
    'K': 100,         # Strike price
    'T': 0.25,        # Time to expiry (years)
    'r': 0.05,        # Risk-free rate
    'sigma': 0.25,    # Volatility
}

# =============================================================================
# 📈 GBM MODEL PARAMETERS
# =============================================================================
GBM_PARAMS = {
    'mu': 0.05,       # Drift parameter
    'sigma': 0.25,    # Volatility parameter
}

# =============================================================================
# 🔄 HESTON MODEL PARAMETERS
# =============================================================================
HESTON_PARAMS = {
    'mu': 0.05,       # Drift parameter
    'kappa': 2.0,     # Mean reversion speed
    'theta': 0.04,    # Long-term variance
    'sigma_v': 0.3,   # Volatility of volatility
    'rho': -0.7,      # Correlation between price and volatility
}

# =============================================================================
# ⚡ MERTON JUMP-DIFFUSION PARAMETERS
# =============================================================================
MERTON_PARAMS = {
    'mu': 0.05,       # Drift parameter
    'sigma': 0.25,    # Volatility parameter
    'lambda_jump': 0.1,    # Jump intensity (jumps per year)
    'mu_jump': -0.1,       # Mean jump size
    'sigma_jump': 0.1,     # Jump size volatility
}

# =============================================================================
# 🎲 SIMULATION PARAMETERS
# =============================================================================
SIMULATION_PARAMS = {
    'n_steps': 252,   # Number of time steps
    'n_paths': 5,     # Number of paths to simulate
    'n_simulations': 10000,  # Number of Monte Carlo simulations
}

# =============================================================================
# 📊 MARKET DATA PARAMETERS
# =============================================================================
MARKET_PARAMS = {
    'ticker': 'AAPL',     # Stock symbol
    'max_expiries': 2,    # Maximum number of expiry dates to fetch
    'risk_free_rate': 0.05,  # Risk-free rate for market data analysis
}

# =============================================================================
# 🎨 PLOTTING PARAMETERS
# =============================================================================
PLOT_PARAMS = {
    'figsize': (12, 8),   # Default figure size
    'dpi': 100,           # Figure DPI
    'style': 'default',   # Matplotlib style
    'save_plots': False,  # Whether to save plots
    'plot_format': 'png', # Plot file format
}

# =============================================================================
# 🔧 ADVANCED PARAMETERS
# =============================================================================
ADVANCED_PARAMS = {
    'tolerance': 1e-6,    # Numerical tolerance for convergence
    'max_iterations': 100, # Maximum iterations for iterative methods
    'seed': 42,           # Random seed for reproducibility
    'parallel': False,    # Whether to use parallel processing
}

# =============================================================================
# 📋 PRESET CONFIGURATIONS
# =============================================================================
PRESETS = {
    'conservative': {
        'OPTION_PARAMS': {'S0': 100, 'K': 95, 'T': 0.5, 'r': 0.02, 'sigma': 0.15},
        'GBM_PARAMS': {'mu': 0.02, 'sigma': 0.15},
        'HESTON_PARAMS': {'mu': 0.02, 'kappa': 1.5, 'theta': 0.02, 'sigma_v': 0.2, 'rho': -0.5},
        'MERTON_PARAMS': {'mu': 0.02, 'sigma': 0.15, 'lambda_jump': 0.05, 'mu_jump': -0.05, 'sigma_jump': 0.05}
    },
    
    'moderate': {
        'OPTION_PARAMS': {'S0': 100, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.25},
        'GBM_PARAMS': {'mu': 0.05, 'sigma': 0.25},
        'HESTON_PARAMS': {'mu': 0.05, 'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.7},
        'MERTON_PARAMS': {'mu': 0.05, 'sigma': 0.25, 'lambda_jump': 0.1, 'mu_jump': -0.1, 'sigma_jump': 0.1}
    },
    
    'aggressive': {
        'OPTION_PARAMS': {'S0': 100, 'K': 105, 'T': 0.1, 'r': 0.08, 'sigma': 0.4},
        'GBM_PARAMS': {'mu': 0.08, 'sigma': 0.4},
        'HESTON_PARAMS': {'mu': 0.08, 'kappa': 3.0, 'theta': 0.06, 'sigma_v': 0.5, 'rho': -0.8},
        'MERTON_PARAMS': {'mu': 0.08, 'sigma': 0.4, 'lambda_jump': 0.2, 'mu_jump': -0.15, 'sigma_jump': 0.2}
    }
}

def get_preset(preset_name):
    """Get a preset configuration"""
    if preset_name not in PRESETS:
        print(f"⚠️  Preset '{preset_name}' not found. Available presets: {list(PRESETS.keys())}")
        return None
    
    preset = PRESETS[preset_name]
    print(f"✅ Loaded '{preset_name}' preset configuration")
    return preset

def print_current_config():
    """Print current configuration"""
    print("📋 Current Configuration:")
    print("=" * 50)
    
    print("\n📊 Option Parameters:")
    for key, value in OPTION_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n📈 GBM Parameters:")
    for key, value in GBM_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n🔄 Heston Parameters:")
    for key, value in HESTON_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n⚡ Merton Parameters:")
    for key, value in MERTON_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n🎲 Simulation Parameters:")
    for key, value in SIMULATION_PARAMS.items():
        print(f"  {key}: {value}")

def update_config(new_params):
    """Update configuration with new parameters"""
    global OPTION_PARAMS, GBM_PARAMS, HESTON_PARAMS, MERTON_PARAMS, SIMULATION_PARAMS
    
    if 'OPTION_PARAMS' in new_params:
        OPTION_PARAMS.update(new_params['OPTION_PARAMS'])
    if 'GBM_PARAMS' in new_params:
        GBM_PARAMS.update(new_params['GBM_PARAMS'])
    if 'HESTON_PARAMS' in new_params:
        HESTON_PARAMS.update(new_params['HESTON_PARAMS'])
    if 'MERTON_PARAMS' in new_params:
        MERTON_PARAMS.update(new_params['MERTON_PARAMS'])
    if 'SIMULATION_PARAMS' in new_params:
        SIMULATION_PARAMS.update(new_params['SIMULATION_PARAMS'])
    
    print("✅ Configuration updated successfully!")

if __name__ == "__main__":
    print_current_config()
    
    print("\n🎯 To use a preset configuration:")
    print("   preset_config = get_preset('moderate')")
    print("   update_config(preset_config)")
    
    print("\n🔧 To update individual parameters:")
    print("   OPTION_PARAMS['S0'] = 150")
    print("   GBM_PARAMS['sigma'] = 0.3") 