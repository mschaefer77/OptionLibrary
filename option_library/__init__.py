"""
Option Pricing Library

A comprehensive library for:
1. Underlying price simulation with three core dynamics models
2. Option pricing with multiple models
3. Market data comparison and analysis
"""

from .dynamics import (
    GeometricBrownianMotion,
    HestonModel,
    MertonJumpDiffusion,
    create_dynamics_model
)

from .simulation import (
    PriceSimulator,
    simulate_paths,
    generate_price_data
)

from .pricing import (
    BlackScholes,
    MonteCarloPricer,
    BinomialTree,
    TrinomialTree,
    FiniteDifference,
    compare_pricing_models
)

from .market_data import (
    MarketDataProvider,
    OptionChainAnalyzer,
    fetch_market_data
)

from .comparison import (
    ModelMarketComparator,
    GreeksAnalyzer,
    VolatilityAnalyzer
)

from .utils import (
    calculate_greeks,
    implied_volatility,
    option_payoff,
    black_scholes_price,
    monte_carlo_price,
    binomial_tree_price,
    print_summary_statistics
)

__version__ = "1.0.0"
__author__ = "Matthew Schaefer"
__email__ = "contact@optiontracka.com"

__all__ = [
    # Dynamics models - Three core models
    'GeometricBrownianMotion',
    'HestonModel', 
    'MertonJumpDiffusion',
    'create_dynamics_model',
    
    # Simulation
    'PriceSimulator',
    'simulate_paths',
    'generate_price_data',
    
    # Pricing models
    'BlackScholes',
    'MonteCarloPricer',
    'BinomialTree',
    'TrinomialTree',
    'FiniteDifference',
    'compare_pricing_models',
    
    # Market data
    'MarketDataProvider',
    'OptionChainAnalyzer',
    'fetch_market_data',
    
    # Comparison and analysis
    'ModelMarketComparator',
    'GreeksAnalyzer',
    'VolatilityAnalyzer',
    
    # Utilities
    'calculate_greeks',
    'implied_volatility',
    'option_payoff',
    'black_scholes_price',
    'monte_carlo_price',
    'binomial_tree_price',
    'print_summary_statistics'
] 