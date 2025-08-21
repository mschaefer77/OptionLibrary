#!/usr/bin/env python3
"""
Interactive Demo with Parameter Input Boxes

This demo provides interactive widgets for easy parameter adjustment.
Requires: pip install ipywidgets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the library
from option_library import (
    # Dynamics models
    create_dynamics_model, GeometricBrownianMotion, HestonModel, MertonJumpDiffusion,
    
    # Simulation
    PriceSimulator, simulate_paths, generate_price_data,
    
    # Pricing models
    BlackScholes, MonteCarloPricer, BinomialTree, TrinomialTree, FiniteDifference,
    compare_pricing_models,
    
    # Market data
    MarketDataProvider, OptionChainAnalyzer, fetch_market_data,
    
    # Comparison and analysis
    ModelMarketComparator, GreeksAnalyzer, VolatilityAnalyzer,
    
    # Utilities
    calculate_greeks, implied_volatility, option_payoff, print_summary_statistics
)

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️  ipywidgets not available. Install with: pip install ipywidgets")
    print("   For now, using basic input() function instead.")


def create_parameter_widgets():
    """Create interactive parameter widgets"""
    if not WIDGETS_AVAILABLE:
        return None
    
    # Option parameters
    option_params = widgets.VBox([
        widgets.HTML(value="<h3>📊 Option Parameters</h3>"),
        widgets.HBox([
            widgets.FloatText(value=100, description="S0 (Current Price):", style={'description_width': '150px'}),
            widgets.FloatText(value=100, description="K (Strike):", style={'description_width': '150px'})
        ]),
        widgets.HBox([
            widgets.FloatText(value=0.25, description="T (Time to Expiry):", style={'description_width': '150px'}),
            widgets.FloatText(value=0.05, description="r (Risk-free Rate):", style={'description_width': '150px'})
        ]),
        widgets.FloatText(value=0.25, description="σ (Volatility):", style={'description_width': '150px'})
    ])
    
    # GBM parameters
    gbm_params = widgets.VBox([
        widgets.HTML(value="<h3>📈 GBM Model Parameters</h3>"),
        widgets.HBox([
            widgets.FloatText(value=0.05, description="μ (Drift):", style={'description_width': '150px'}),
            widgets.FloatText(value=0.25, description="σ (Volatility):", style={'description_width': '150px'})
        ])
    ])
    
    # Heston parameters
    heston_params = widgets.VBox([
        widgets.HTML(value="<h3>🔄 Heston Model Parameters</h3>"),
        widgets.HBox([
            widgets.FloatText(value=0.05, description="μ (Drift):", style={'description_width': '150px'}),
            widgets.FloatText(value=2.0, description="κ (Mean Reversion):", style={'description_width': '150px'})
        ]),
        widgets.HBox([
            widgets.FloatText(value=0.04, description="θ (Long-term Var):", style={'description_width': '150px'}),
            widgets.FloatText(value=0.3, description="σ_v (Vol of Vol):", style={'description_width': '150px'})
        ]),
        widgets.FloatText(value=-0.7, description="ρ (Correlation):", style={'description_width': '150px'})
    ])
    
    # Merton parameters
    merton_params = widgets.VBox([
        widgets.HTML(value="<h3>⚡ Merton Jump-Diffusion Parameters</h3>"),
        widgets.HBox([
            widgets.FloatText(value=0.05, description="μ (Drift):", style={'description_width': '150px'}),
            widgets.FloatText(value=0.25, description="σ (Volatility):", style={'description_width': '150px'})
        ]),
        widgets.HBox([
            widgets.FloatText(value=0.1, description="λ (Jump Intensity):", style={'description_width': '150px'}),
            widgets.FloatText(value=-0.1, description="μ_jump (Jump Mean):", style={'description_width': '150px'})
        ]),
        widgets.FloatText(value=0.1, description="σ_jump (Jump Vol):", style={'description_width': '150px'})
    ])
    
    # Simulation parameters
    sim_params = widgets.VBox([
        widgets.HTML(value="<h3>🎲 Simulation Parameters</h3>"),
        widgets.HBox([
            widgets.IntText(value=252, description="Time Steps:", style={'description_width': '150px'}),
            widgets.IntText(value=5, description="Paths:", style={'description_width': '150px'})
        ])
    ])
    
    # Action buttons
    action_buttons = widgets.VBox([
        widgets.HTML(value="<h3>🚀 Actions</h3>"),
        widgets.HBox([
            widgets.Button(description="Simulate Paths", button_style='primary'),
            widgets.Button(description="Price Options", button_style='success'),
            widgets.Button(description="Compare Models", button_style='info')
        ])
    ])
    
    # Combine all widgets
    all_widgets = widgets.VBox([
        option_params,
        gbm_params,
        heston_params,
        merton_params,
        sim_params,
        action_buttons
    ])
    
    return all_widgets


def get_parameters_from_widgets(widgets_container):
    """Extract parameters from widgets"""
    if not WIDGETS_AVAILABLE:
        return get_parameters_from_input()
    
    # Extract values from widgets (this would need to be customized based on your widget structure)
    # For now, return default values
    return {
        'S0': 100,
        'K': 100,
        'T': 0.25,
        'r': 0.05,
        'sigma': 0.25,
        'gbm_mu': 0.05,
        'gbm_sigma': 0.25,
        'heston_mu': 0.05,
        'heston_kappa': 2.0,
        'heston_theta': 0.04,
        'heston_sigma_v': 0.3,
        'heston_rho': -0.7,
        'merton_mu': 0.05,
        'merton_sigma': 0.25,
        'merton_lambda': 0.1,
        'merton_mu_jump': -0.1,
        'merton_sigma_jump': 0.1,
        'n_steps': 252,
        'n_paths': 5
    }


def get_parameters_from_input():
    """Get parameters using basic input() function"""
    print("\n📊 Enter Option Parameters:")
    print("=" * 40)
    
    S0 = float(input("S0 (Current Price) [100]: ") or 100)
    K = float(input("K (Strike) [100]: ") or 100)
    T = float(input("T (Time to Expiry in years) [0.25]: ") or 0.25)
    r = float(input("r (Risk-free Rate) [0.05]: ") or 0.05)
    sigma = float(input("σ (Volatility) [0.25]: ") or 0.25)
    
    print("\n📈 GBM Model Parameters:")
    print("-" * 30)
    gbm_mu = float(input("μ (Drift) [0.05]: ") or 0.05)
    gbm_sigma = float(input("σ (Volatility) [0.25]: ") or 0.25)
    
    print("\n🔄 Heston Model Parameters:")
    print("-" * 30)
    heston_mu = float(input("μ (Drift) [0.05]: ") or 0.05)
    heston_kappa = float(input("κ (Mean Reversion) [2.0]: ") or 2.0)
    heston_theta = float(input("θ (Long-term Variance) [0.04]: ") or 0.04)
    heston_sigma_v = float(input("σ_v (Vol of Vol) [0.3]: ") or 0.3)
    heston_rho = float(input("ρ (Correlation) [-0.7]: ") or -0.7)
    
    print("\n⚡ Merton Jump-Diffusion Parameters:")
    print("-" * 30)
    merton_mu = float(input("μ (Drift) [0.05]: ") or 0.05)
    merton_sigma = float(input("σ (Volatility) [0.25]: ") or 0.25)
    merton_lambda = float(input("λ (Jump Intensity) [0.1]: ") or 0.1)
    merton_mu_jump = float(input("μ_jump (Jump Mean) [-0.1]: ") or -0.1)
    merton_sigma_jump = float(input("σ_jump (Jump Vol) [0.1]: ") or 0.1)
    
    print("\n🎲 Simulation Parameters:")
    print("-" * 30)
    n_steps = int(input("Time Steps [252]: ") or 252)
    n_paths = int(input("Paths [5]: ") or 5)
    
    return {
        'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
        'gbm_mu': gbm_mu, 'gbm_sigma': gbm_sigma,
        'heston_mu': heston_mu, 'heston_kappa': heston_kappa,
        'heston_theta': heston_theta, 'heston_sigma_v': heston_sigma_v,
        'heston_rho': heston_rho,
        'merton_mu': merton_mu, 'merton_sigma': merton_sigma,
        'merton_lambda': merton_lambda, 'merton_mu_jump': merton_mu_jump,
        'merton_sigma_jump': merton_sigma_jump,
        'n_steps': n_steps, 'n_paths': n_paths
    }


def interactive_simulation(params):
    """Run simulation with given parameters"""
    print(f"\n🎯 Running Simulation with Parameters:")
    print(f"  S0: ${params['S0']}, K: ${params['K']}, T: {params['T']} years")
    print(f"  r: {params['r']:.2%}, σ: {params['sigma']:.2%}")
    
    # Create dynamics models
    models = {
        'GBM': create_dynamics_model('GBM', mu=params['gbm_mu'], sigma=params['gbm_sigma']),
        'Heston': create_dynamics_model('Heston', mu=params['heston_mu'], kappa=params['heston_kappa'], 
                                      theta=params['heston_theta'], sigma_v=params['heston_sigma_v'], 
                                      rho=params['heston_rho']),
        'Merton': create_dynamics_model('Merton', mu=params['merton_mu'], sigma=params['merton_sigma'], 
                                      lambda_jump=params['merton_lambda'], mu_jump=params['merton_mu_jump'], 
                                      sigma_jump=params['merton_sigma_jump'])
    }
    
    # Simulate and plot paths
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Interactive Simulation with Different Dynamics', fontsize=16)
    
    for i, (name, model) in enumerate(models.items()):
        simulator = PriceSimulator(model)
        
        if name == 'Heston':
            S, v, t = simulator.simulate_paths(params['S0'], params['T'], params['n_steps'], params['n_paths'])
            # Plot price paths
            for j in range(min(3, S.shape[0])):
                axes[i].plot(t, S[j], alpha=0.7, linewidth=1)
            axes[i].plot(t, np.mean(S, axis=0), 'r-', linewidth=2, label='Mean Path')
        else:
            S, t = simulator.simulate_paths(params['S0'], params['T'], params['n_steps'], params['n_paths'])
            # Plot price paths
            for j in range(min(3, S.shape[0])):
                axes[i].plot(t, S[j], alpha=0.7, linewidth=1)
            axes[i].plot(t, np.mean(S, axis=0), 'r-', linewidth=2, label='Mean Path')
        
        axes[i].set_title(f'{name} Model')
        axes[i].set_xlabel('Time (years)')
        axes[i].set_ylabel('Price')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return models


def interactive_pricing(params):
    """Run option pricing with given parameters"""
    print(f"\n💰 Running Option Pricing with Parameters:")
    print(f"  S0: ${params['S0']}, K: ${params['K']}, T: {params['T']} years")
    print(f"  r: {params['r']:.2%}, σ: {params['sigma']:.2%}")
    
    # Test pricing with different dynamics models
    dynamics_models = {
        'GBM': create_dynamics_model('GBM', mu=params['r'], sigma=params['sigma']),
        'Heston': create_dynamics_model('Heston', mu=params['r'], kappa=params['heston_kappa'], 
                                      theta=params['sigma']**2, sigma_v=params['heston_sigma_v'], 
                                      rho=params['heston_rho']),
        'Merton': create_dynamics_model('Merton', mu=params['r'], sigma=params['sigma'], 
                                      lambda_jump=params['merton_lambda'], mu_jump=params['merton_mu_jump'], 
                                      sigma_jump=params['merton_sigma_jump'])
    }
    
    for model_name, dynamics_model in dynamics_models.items():
        print(f"\n🔍 Testing with {model_name} dynamics:")
        
        try:
            # Compare pricing models with this dynamics model
            results = compare_pricing_models(params['S0'], params['K'], params['T'], params['r'], dynamics_model)
            
            print(f"  📊 Pricing Results:")
            for pricing_model, result in results.items():
                if 'call' in result:
                    print(f"    {pricing_model} Call: ${result['call']:.4f}")
                if 'put' in result:
                    print(f"    {pricing_model} Put: ${result['put']:.4f}")
                if 'call_std_error' in result:
                    print(f"    {pricing_model} Call Std Error: ${result['call_std_error']:.4f}")
                    print(f"    {pricing_model} Put Std Error: ${result['put_std_error']:.4f}")
                    
        except Exception as e:
            print(f"  ❌ Error with {model_name}: {e}")


def main():
    """Main interactive demo function"""
    print("🚀 Interactive Option Pricing Library Demo")
    print("=" * 60)
    
    if WIDGETS_AVAILABLE:
        print("✅ Jupyter widgets available - creating interactive interface...")
        widgets_container = create_parameter_widgets()
        display(widgets_container)
        
        print("\n📝 Use the widgets above to adjust parameters, then run the functions below:")
        print("   interactive_simulation(params)")
        print("   interactive_pricing(params)")
        
    else:
        print("📝 Using basic input interface...")
        params = get_parameters_from_input()
        
        print(f"\n✅ Parameters loaded:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Run simulations
        interactive_simulation(params)
        
        # Run pricing
        interactive_pricing(params)


if __name__ == "__main__":
    main() 