"""
Option Pricing Library Demo

This demo shows how to use the option pricing library to:
1. Simulate underlying prices with different dynamics (GBM, Heston, Merton Jump)
2. Price options using various models with proper dynamics integration
3. Compare with market data
4. Analyze Greeks and volatility
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


def demo_underlying_simulation():
    """Demo 1: Simulate underlying prices with different dynamics models"""
    print("🎯 Demo 1: Underlying Price Simulation")
    print("=" * 50)
    
    # Parameters
    S0 = 100
    T = 1.0
    n_steps = 252
    n_paths = 5
    
    # Create different dynamics models
    models = {
        'GBM': create_dynamics_model('GBM', mu=0.05, sigma=0.2),
        'Heston': create_dynamics_model('Heston', mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7),
        'Merton': create_dynamics_model('Merton', mu=0.05, sigma=0.2, lambda_jump=0.1, mu_jump=-0.1, sigma_jump=0.1)
    }
    
    # Simulate and plot paths
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Underlying Price Simulation with Different Dynamics', fontsize=16)
    
    for i, (name, model) in enumerate(models.items()):
        simulator = PriceSimulator(model)
        
        if name == 'Heston':
            S, v, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
            # Plot price paths
            for j in range(min(3, S.shape[0])):
                axes[i].plot(t, S[j], alpha=0.7, linewidth=1)
            axes[i].plot(t, np.mean(S, axis=0), 'r-', linewidth=2, label='Mean Path')
        else:
            S, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
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
    
    # Print statistics
    for name, model in models.items():
        simulator = PriceSimulator(model)
        if name == 'Heston':
            S, v, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        else:
            S, t = simulator.simulate_paths(S0, T, n_steps, n_paths)
        
        stats = simulator.get_statistics(S, t)
        print(f"\n📊 {name} Model Statistics:")
        print(f"  Final Price Range: ${stats['min_price']:.2f} - ${stats['max_price']:.2f}")
        print(f"  Mean Final Price: ${stats['mean_final_price']:.2f}")
        print(f"  Volatility: {stats['volatility'].mean():.2%}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        
        # For Heston, also show variance statistics
        if name == 'Heston':
            print(f"  Final Variance Range: {v[:, -1].min():.6f} - {v[:, -1].max():.6f}")
            print(f"  Mean Final Variance: {v[:, -1].mean():.6f}")


def demo_option_pricing():
    """Demo 2: Price options using different models with dynamics integration"""
    print("\n🎯 Demo 2: Option Pricing Models")
    print("=" * 50)
    
    # Parameters
    S0 = 100
    K = 100
    T = 0.25
    r = 0.05
    sigma = 0.25
    
    print(f"Parameters:")
    print(f"  S0: ${S0}")
    print(f"  K: ${K}")
    print(f"  T: {T} years")
    print(f"  r: {r:.2%}")
    print(f"  σ: {sigma:.2%}")
    
    # Test pricing with different dynamics models
    dynamics_models = {
        'GBM': create_dynamics_model('GBM', mu=r, sigma=sigma),
        'Heston': create_dynamics_model('Heston', mu=r, kappa=2.0, theta=sigma**2, sigma_v=0.3, rho=-0.7),
        'Merton': create_dynamics_model('Merton', mu=r, sigma=sigma, lambda_jump=0.1, mu_jump=-0.1, sigma_jump=0.1)
    }
    
    for model_name, dynamics_model in dynamics_models.items():
        print(f"\n🔍 Testing with {model_name} dynamics:")
        
        try:
            # Compare pricing models with this dynamics model
            results = compare_pricing_models(S0, K, T, r, dynamics_model)
            
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


def demo_market_comparison():
    """Demo 3: Compare model prices with market data"""
    print("\n🎯 Demo 3: Model vs Market Comparison")
    print("=" * 50)
    
    # Try to fetch real market data
    ticker = "AAPL"
    print(f"Fetching market data for {ticker}...")
    
    try:
        market_data = fetch_market_data(ticker, max_expiries=1)
        
        if not market_data['options'].empty:
            print(f"✅ Fetched {len(market_data['options'])} options from market")
            
            # Prepare data for comparison
            options_data = market_data['options'].copy()
            
            # Create time_to_expiry column from available data
            if 'T' in options_data.columns:
                options_data['time_to_expiry'] = options_data['T']
                print(f"  ✅ Using 'T' column for time to expiry")
            elif 'expiry_date' in options_data.columns:
                # Calculate time to expiry from expiry_date
                options_data['expiry_date'] = pd.to_datetime(options_data['expiry_date'])
                today = pd.Timestamp.now()
                options_data['time_to_expiry'] = (options_data['expiry_date'] - today).dt.days / 365
                print(f"  ✅ Calculated time to expiry from expiry_date")
            else:
                # Default to 3 months if no expiry information
                options_data['time_to_expiry'] = 0.25
                print(f"  ⚠️  No expiry information found, using default 3 months")
            
            # Compare prices
            comparator = ModelMarketComparator(market_data['current_price'])
            # Use the volatility from market data for comparison
            sigma = market_data['volatility']
            comparison_data = comparator.compare_prices(options_data, sigma)
            
            # Print summary
            comparator.print_summary(comparison_data)
            
            # Plot comparison
            comparator.plot_comparison(comparison_data)
            
        else:
            print("No market data available, using synthetic data")
            
    except Exception as e:
        print(f"Error fetching market data: {e}")
        print("Continuing with synthetic data demo...")
        
        # Create synthetic market data for demo
        S0 = 100
        strikes = np.linspace(90, 110, 10)
        synthetic_data = []
        
        for K in strikes:
            T = 0.25
            # Create volatility smile
            moneyness = K / S0
            if moneyness < 0.95:  # ITM
                implied_vol = 0.20 + 0.05 * (0.95 - moneyness)
            elif moneyness > 1.05:  # OTM
                implied_vol = 0.20 + 0.05 * (moneyness - 1.05)
            else:  # ATM
                implied_vol = 0.20
            
            # Calculate market price using implied vol
            market_price = BlackScholes(S0, K, T, 0.05, implied_vol).price_call()
            
            synthetic_data.append({
                'strike': K,
                'lastPrice': market_price,
                'option_type': 'call',
                'time_to_expiry': T,
                'current_price': S0
            })
        
        market_df = pd.DataFrame(synthetic_data)
        print(f"✅ Created synthetic market data with {len(market_df)} options")


def demo_greeks_analysis():
    """Demo 4: Greeks analysis with dynamics models"""
    print("\n🎯 Demo 4: Greeks Analysis")
    print("=" * 50)
    
    # Parameters
    S0 = 100
    strikes = np.linspace(80, 120, 20)
    T = 0.25
    r = 0.05
    sigma = 0.25
    
    # Create GBM dynamics model for Greeks calculation
    gbm_model = create_dynamics_model('GBM', mu=r, sigma=sigma)
    
    print(f"📊 Greeks Analysis Summary:")
    print(f"  Number of Options: {len(strikes) * 2}")  # Calls and puts
    print(f"  Strike Range: ${strikes.min():.2f} - ${strikes.max():.2f}")
    
    # Calculate Greeks for a few representative options
    sample_strikes = [90, 100, 110]
    print(f"\n🔍 Sample Greeks (T={T}, r={r:.2%}, σ={sigma:.2%}):")
    
    for K in sample_strikes:
        # Call option
        bs_call = BlackScholes(S0, K, T, r, sigma)
        call_greeks = bs_call.calculate_greeks()
        
        # Put option  
        bs_put = BlackScholes(S0, K, T, r, sigma)
        put_greeks = bs_put.calculate_greeks()
        
        print(f"\n  Strike ${K}:")
        print(f"    Call - Delta: {call_greeks['delta_call']:.4f}, Gamma: {call_greeks['gamma']:.6f}, Vega: {call_greeks['vega']:.2f}")
        print(f"    Put  - Delta: {put_greeks['delta_put']:.4f}, Gamma: {put_greeks['gamma']:.6f}, Vega: {put_greeks['vega']:.2f}")


def demo_volatility_analysis():
    """Demo 5: Volatility analysis with real market data"""
    print("\n🎯 Demo 5: Volatility Analysis with Market Data")
    print("=" * 50)
    
    # Try to fetch real market data
    ticker = "AAPL"  # You can change this to any ticker
    print(f"Fetching market data for {ticker}...")
    
    try:
        market_data = fetch_market_data(ticker, max_expiries=2)
        
        if not market_data['options'].empty:
            print(f"✅ Fetched {len(market_data['options'])} options from market")
            
            # Filter for calls and puts separately to analyze volatility skew
            calls = market_data['options'][market_data['options']['option_type'] == 'call']
            puts = market_data['options'][market_data['options']['option_type'] == 'put']
            
            print(f"  Calls: {len(calls)} options")
            print(f"  Puts: {len(puts)} options")
            print(f"  Current Price: ${market_data['current_price']:.2f}")
            print(f"  Market Volatility: {market_data['volatility']:.2%}")
            
            # Check what columns we actually have
            print(f"  Available columns: {list(market_data['options'].columns)}")
            print(f"  Sample data shape: {market_data['options'].shape}")
            if not market_data['options'].empty:
                print(f"  Sample data:")
                print(market_data['options'][['strike', 'lastPrice', 'option_type', 'expiry_date']].head(3))
                if 'T' in market_data['options'].columns:
                    print(f"  'T' column sample: {market_data['options']['T'].head(3).tolist()}")
            
            # Analyze volatility skew for calls
            if not calls.empty:
                print(f"\n📊 Call Options Volatility Analysis:")
                calls['moneyness'] = calls['strike'] / market_data['current_price']
                
                # Use 'T' column if it exists (created by OptionChainAnalyzer), otherwise calculate from expiry_date
                if 'T' in calls.columns:
                    calls['time_to_expiry'] = calls['T']
                    print(f"    ✅ Using 'T' column for time to expiry")
                elif 'expiry_date' in calls.columns:
                    # Calculate time to expiry from expiry_date
                    calls['expiry_date'] = pd.to_datetime(calls['expiry_date'])
                    today = pd.Timestamp.now()
                    calls['time_to_expiry'] = (calls['expiry_date'] - today).dt.days / 365
                    print(f"    ✅ Calculated time to expiry from expiry_date")
                else:
                    # Default to 3 months if no expiry information
                    calls['time_to_expiry'] = 0.25
                    print("    ⚠️  No expiry information found, using default 3 months")
                
                # Group by expiry and analyze skew
                for expiry in calls['time_to_expiry'].unique():
                    if pd.isna(expiry):
                        continue
                    
                    expiry_calls = calls[calls['time_to_expiry'] == expiry]
                    if len(expiry_calls) < 3:  # Need at least 3 points for meaningful analysis
                        continue
                    
                    print(f"\n  Expiry: {expiry:.2f} years ({expiry*12:.1f} months)")
                    
                    # Sort by strike for proper skew analysis
                    expiry_calls = expiry_calls.sort_values('strike')
                    
                    # Calculate implied volatility for each call
                    implied_vols = []
                    for _, row in expiry_calls.iterrows():
                        try:
                            iv = implied_volatility(
                                market_data['current_price'], row['strike'], row['time_to_expiry'], 
                                0.05, row['lastPrice'], 'call'
                            )
                            implied_vols.append(iv)
                        except:
                            implied_vols.append(np.nan)
                    
                    expiry_calls['implied_volatility'] = implied_vols
                    valid_vols = expiry_calls['implied_volatility'].dropna()
                    
                    if len(valid_vols) > 0:
                        print(f"    Strike Range: ${expiry_calls['strike'].min():.2f} - ${expiry_calls['strike'].max():.2f}")
                        print(f"    Vol Range: {valid_vols.min():.4f} - {valid_vols.max():.4f}")
                        print(f"    Mean Vol: {valid_vols.mean():.4f}")
                        
                        # Plot volatility skew for this expiry
                        plt.figure(figsize=(12, 8))
                        
                        # Plot calls
                        plt.subplot(2, 2, 1)
                        plt.scatter(expiry_calls['strike'], expiry_calls['implied_volatility'], 
                                  alpha=0.7, color='blue', label='Calls')
                        plt.xlabel('Strike Price')
                        plt.ylabel('Implied Volatility')
                        plt.title(f'Call Volatility Skew - {expiry*12:.1f} months')
                        plt.axvline(x=market_data['current_price'], color='red', linestyle='--', 
                                   label=f'Current Price: ${market_data['current_price']:.2f}')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Plot moneyness
                        plt.subplot(2, 2, 2)
                        plt.scatter(expiry_calls['moneyness'], expiry_calls['implied_volatility'], 
                                  alpha=0.7, color='blue', label='Calls')
                        plt.xlabel('Moneyness (Strike/Current Price)')
                        plt.ylabel('Implied Volatility')
                        plt.title(f'Call Volatility vs Moneyness - {expiry*12:.1f} months')
                        plt.axvline(x=1.0, color='red', linestyle='--', label='ATM (Moneyness = 1.0)')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Plot puts if available
                        if not puts.empty:
                            # Also need to handle time_to_expiry for puts
                            if 'T' in puts.columns:
                                puts['time_to_expiry'] = puts['T']
                            elif 'expiry_date' in puts.columns:
                                puts['expiry_date'] = pd.to_datetime(puts['expiry_date'])
                                today = pd.Timestamp.now()
                                puts['time_to_expiry'] = (puts['expiry_date'] - today).dt.days / 365
                            else:
                                puts['time_to_expiry'] = 0.25
                            
                            expiry_puts = puts[puts['time_to_expiry'] == expiry]
                            if len(expiry_puts) >= 3:
                                expiry_puts['moneyness'] = expiry_puts['strike'] / market_data['current_price']
                                expiry_puts = expiry_puts.sort_values('strike')
                                
                                # Calculate implied volatility for puts
                                put_implied_vols = []
                                for _, row in expiry_puts.iterrows():
                                    try:
                                        iv = implied_volatility(
                                            market_data['current_price'], row['strike'], row['time_to_expiry'], 
                                            0.05, row['lastPrice'], 'put'
                                        )
                                        put_implied_vols.append(iv)
                                    except:
                                        put_implied_vols.append(np.nan)
                                
                                expiry_puts['implied_volatility'] = put_implied_vols
                                valid_put_vols = expiry_puts['implied_volatility'].dropna()
                                
                                if len(valid_put_vols) > 0:
                                    # Combined plot
                                    plt.subplot(2, 2, 3)
                                    plt.scatter(expiry_calls['strike'], expiry_calls['implied_volatility'], 
                                              alpha=0.7, color='blue', label='Calls')
                                    plt.scatter(expiry_puts['strike'], expiry_puts['implied_volatility'], 
                                              alpha=0.7, color='red', label='Puts')
                                    plt.xlabel('Strike Price')
                                    plt.ylabel('Implied Volatility')
                                    plt.title(f'Call vs Put Volatility - {expiry*12:.1f} months')
                                    plt.axvline(x=market_data['current_price'], color='black', linestyle='--', 
                                               label=f'Current Price: ${market_data['current_price']:.2f}')
                                    plt.legend()
                                    plt.grid(True, alpha=0.3)
                                    
                                    # Put-call parity analysis
                                    plt.subplot(2, 2, 4)
                                    plt.scatter(expiry_calls['moneyness'], expiry_calls['implied_volatility'], 
                                              alpha=0.7, color='blue', label='Calls')
                                    plt.scatter(expiry_puts['moneyness'], expiry_puts['implied_volatility'], 
                                              alpha=0.7, color='red', label='Puts')
                                    plt.xlabel('Moneyness (Strike/Current Price)')
                                    plt.ylabel('Implied Volatility')
                                    plt.title(f'Put-Call Volatility Parity - {expiry*12:.1f} months')
                                    plt.axvline(x=1.0, color='black', linestyle='--', label='ATM (Moneyness = 1.0)')
                                    plt.legend()
                                    plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Analyze skew patterns
                        atm_calls = expiry_calls[expiry_calls['moneyness'].between(0.98, 1.02)]
                        itm_calls = expiry_calls[expiry_calls['moneyness'] < 0.95]
                        otm_calls = expiry_calls[expiry_calls['moneyness'] > 1.05]
                        
                        print(f"    ATM Volatility (0.98-1.02): {atm_calls['implied_volatility'].mean():.4f}" if not atm_calls.empty else "    ATM Volatility: No data")
                        print(f"    ITM Volatility (<0.95): {itm_calls['implied_volatility'].mean():.4f}" if not itm_calls.empty else "    ITM Volatility: No data")
                        print(f"    OTM Volatility (>1.05): {otm_calls['implied_volatility'].mean():.4f}" if not otm_calls.empty else "    OTM Volatility: No data")
                        
                        # Calculate skew
                        if not atm_calls.empty and not otm_calls.empty:
                            skew = otm_calls['implied_volatility'].mean() - atm_calls['implied_volatility'].mean()
                            print(f"    Volatility Skew (OTM - ATM): {skew:.4f}")
                        
                        break  # Only show first expiry for demo purposes
                    else:
                        print(f"    Could not calculate implied volatilities for this expiry")
            else:
                print("❌ No call options available for analysis")
            
            # Summary statistics
            print(f"\n📊 Summary:")
            print(f"  Total Options: {len(market_data['options'])}")
            print(f"  Current Price: ${market_data['current_price']:.2f}")
            print(f"  Market Volatility: {market_data['volatility']:.2%}")
            
            if 'volatility_smile' in market_data and market_data['volatility_smile']:
                smile = market_data['volatility_smile']
                if 'calls' in smile and smile['calls']['strikes']:
                    print(f"  Volatility Smile Data Available: {len(smile['calls']['strikes'])} call strikes")
            
        else:
            print("❌ No market data available")
            print("Falling back to synthetic data demo...")
            demo_volatility_analysis_synthetic()
            
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        print("Falling back to synthetic data demo...")
        demo_volatility_analysis_synthetic()


def demo_volatility_analysis_synthetic():
    """Fallback synthetic volatility analysis"""
    print("\n📊 Synthetic Volatility Analysis (Fallback)")
    print("=" * 50)
    
    # Create synthetic market data with different implied volatilities
    S0 = 100
    strikes = np.linspace(80, 120, 20)
    synthetic_data = []
    
    for K in strikes:
        T = 0.25
        moneyness = K / S0
        
        # Create volatility smile
        if moneyness < 0.95:  # ITM
            implied_vol = 0.20 + 0.05 * (0.95 - moneyness)
        elif moneyness > 1.05:  # OTM
            implied_vol = 0.20 + 0.05 * (moneyness - 1.05)
        else:  # ATM
            implied_vol = 0.20
        
        # Calculate market price using implied vol
        market_price = BlackScholes(S0, K, T, 0.05, implied_vol).price_call()
        
        synthetic_data.append({
            'strike': K,
            'lastPrice': market_price,
            'option_type': 'call',
            'time_to_expiry': T,
            'current_price': S0
        })
    
    market_df = pd.DataFrame(synthetic_data)
    
    print(f"📊 Volatility Analysis:")
    print(f"  Number of Options: {len(market_df)}")
    print(f"  Strike Range: ${market_df['strike'].min():.2f} - ${market_df['strike'].max():.2f}")
    
    # Calculate implied volatility for each option
    implied_vols = []
    for _, row in market_df.iterrows():
        try:
            iv = implied_volatility(
                row['current_price'], row['strike'], row['time_to_expiry'], 
                0.05, row['lastPrice'], 'call'
            )
            implied_vols.append(iv)
        except:
            implied_vols.append(np.nan)
    
    market_df['implied_volatility'] = implied_vols
    valid_vols = market_df['implied_volatility'].dropna()
    
    if len(valid_vols) > 0:
        print(f"  Implied Vol Range: {valid_vols.min():.4f} - {valid_vols.max():.4f}")
        print(f"  Mean Implied Vol: {valid_vols.mean():.4f}")
        
        # Plot volatility smile
        plt.figure(figsize=(10, 6))
        plt.scatter(market_df['strike'], market_df['implied_volatility'], alpha=0.7)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title('Synthetic Volatility Smile')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("  Could not calculate implied volatilities")


def demo_comprehensive_workflow():
    """Demo 6: Comprehensive workflow with dynamics models"""
    print("\n🎯 Demo 6: Comprehensive Workflow")
    print("=" * 50)
    
    # Step 1: Choose dynamics model
    print("1️⃣ Choosing underlying dynamics model...")
    dynamics_model = create_dynamics_model('Heston', mu=0.05, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    print(f"   Selected: {dynamics_model.get_parameters()['model']} model")
    
    # Step 2: Simulate underlying prices
    print("2️⃣ Simulating underlying price paths...")
    simulator = PriceSimulator(dynamics_model)
    S, v, t = simulator.simulate_paths(100, 1.0, 252, 1)
    print(f"   Simulated {len(t)} time steps, final price: ${S[0, -1]:.2f}")
    
    # Step 3: Price options using different models
    print("3️⃣ Pricing options using different models...")
    K = 100
    T = 0.25
    r = 0.05
    
    # Test different pricing approaches
    pricing_results = {}
    
    # Monte Carlo with Heston dynamics
    try:
        mc = MonteCarloPricer(100, K, T, r, dynamics_model)
        mc_call = mc.price_call()
        pricing_results['Monte Carlo (Heston)'] = mc_call['price']
        print(f"   Monte Carlo Call: ${mc_call['price']:.4f}")
    except Exception as e:
        print(f"   Monte Carlo failed: {e}")
    
    # Black-Scholes (GBM only)
    try:
        gbm_model = create_dynamics_model('GBM', mu=r, sigma=0.3)
        bs = BlackScholes(100, K, T, r, 0.3)
        pricing_results['Black-Scholes (GBM)'] = bs.price_call()
        print(f"   Black-Scholes Call: ${bs.price_call():.4f}")
    except Exception as e:
        print(f"   Black-Scholes failed: {e}")
    
    # Step 4: Calculate Greeks (for GBM models)
    print("4️⃣ Calculating option Greeks...")
    try:
        greeks = bs.calculate_greeks()
        print(f"📊 Greeks (Call Option):")
        print(f"  Delta: {greeks['delta_call']:.4f}")
        print(f"  Gamma: {greeks['gamma']:.4f}")
        print(f"  Vega: {greeks['vega']:.4f}")
        print(f"  Theta: {greeks['theta_call']:.4f}")
        print(f"  Rho: {greeks['rho_call']:.4f}")
    except Exception as e:
        print(f"   Greeks calculation failed: {e}")
    
    # Step 5: Risk analysis
    print("5️⃣ Performing risk analysis...")
    final_prices = S[0, -1]
    if pricing_results:
        best_price = min(pricing_results.values())
        payoff = option_payoff(final_prices, K, 'call')
        print(f"📊 Risk Analysis:")
        print(f"  Final Underlying Price: ${final_prices:.2f}")
        print(f"  Option Payoff: ${payoff:.2f}")
        print(f"  Profit/Loss: ${payoff - best_price:.4f}")
    
    print("\n✅ Comprehensive workflow completed!")


def main():
    """Main demo function"""
    print("🚀 Option Pricing Library Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive option pricing library")
    print("with underlying simulation, option pricing, and market comparison.")
    print()
    
    # Run all demos
    demo_underlying_simulation()
    demo_option_pricing()
    demo_market_comparison()
    demo_greeks_analysis()
    demo_volatility_analysis()
    demo_comprehensive_workflow()
    
    print("\n🎉 All demos completed!")
    print("\n📚 Library Features Demonstrated:")
    print("  ✅ Underlying price simulation with GBM, Heston, and Merton dynamics")
    print("  ✅ Multiple option pricing models with dynamics integration")
    print("  ✅ Market data integration and comparison")
    print("  ✅ Greeks calculation and analysis")
    print("  ✅ Volatility smile analysis")
    print("  ✅ Comprehensive risk management tools")
    print("\n🔧 Ready to use for:")
    print("  • Quantitative research")
    print("  • Risk management")
    print("  • Trading strategy development")
    print("  • Educational purposes")


if __name__ == "__main__":
    main() 