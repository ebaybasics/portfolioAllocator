import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def fetch_historical_data(tickers, period="1y"):
    """Fetch historical price data for a list of tickers"""
    # Get unique tickers across all themes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Downloading price data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Handle different data structures and column availability
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-index columns case (multiple tickers)
        if 'Adj Close' in data.columns.levels[0]:
            price_data = data['Adj Close']
        else:
            # Fall back to Close if Adj Close is unavailable
            print("'Adj Close' not available, using 'Close' prices instead")
            price_data = data['Close']
    else:
        # Single column case (likely single ticker)
        if 'Adj Close' in data.columns:
            price_data = data['Adj Close']
        else:
            print("'Adj Close' not available, using 'Close' prices instead")
            price_data = data['Close']
    
    # Handle single ticker case
    if isinstance(price_data, pd.Series):
        price_data = pd.DataFrame(price_data).rename(columns={price_data.name: tickers[0]})
    
    return price_data

def calculate_theme_volatility(price_data, theme_groups):
    """Calculate volatility for each theme based on average returns"""
    returns = price_data.pct_change().dropna()
    theme_volatility = {}
    
    for theme, tickers in theme_groups.items():
        # Calculate average daily returns for this theme (equal-weighted)
        valid_tickers = [t for t in tickers if t in returns.columns]
        if not valid_tickers:
            print(f"Warning: No valid data for theme {theme}")
            continue
            
        theme_returns = returns[valid_tickers].mean(axis=1)
        
        # Calculate annualized volatility
        theme_volatility[theme] = theme_returns.std() * np.sqrt(252)
    
    return pd.Series(theme_volatility)

def calculate_weights(theme_groups, price_data, conviction_multipliers=None):
    """Calculate weights for themes and tickers using inverse volatility method"""
    # Calculate theme volatility
    theme_volatility = calculate_theme_volatility(price_data, theme_groups)
    
    # Calculate theme weights using inverse volatility
    inverse_vol = 1 / theme_volatility
    
    # Apply conviction multipliers if provided
    if conviction_multipliers:
        for theme in inverse_vol.index:
            if theme in conviction_multipliers:
                inverse_vol[theme] *= conviction_multipliers[theme]
    
    # Normalize to get theme weights
    theme_weights = inverse_vol / inverse_vol.sum()
    
    # Calculate ticker weights by splitting theme weights equally
    ticker_weights = {}
    for theme, tickers in theme_groups.items():
        if theme not in theme_weights:
            continue
            
        theme_weight = theme_weights[theme]
        ticker_weight = theme_weight / len(tickers)
        
        for ticker in tickers:
            ticker_weights[ticker] = ticker_weight
    
    return pd.Series(ticker_weights), theme_weights, theme_volatility

def calculate_allocation(ticker_weights, price_data, portfolio_value):
    """Calculate allocation and shares to buy based on weights"""
    # Get latest prices
    latest_prices = price_data.iloc[-1]
    
    # Create allocation DataFrame
    allocation = pd.DataFrame(index=ticker_weights.index)
    allocation['Weight %'] = (ticker_weights * 100).round(2)
    allocation['Latest Price'] = latest_prices.round(2)
    allocation['Dollar Allocation'] = (ticker_weights * portfolio_value).round(2)
    allocation['Shares to Buy'] = (allocation['Dollar Allocation'] / allocation['Latest Price']).round(0).astype(int)
    
    # Sort by weight descending
    allocation = allocation.sort_values('Weight %', ascending=False)
    
    return allocation

def plot_theme_allocation(theme_groups, theme_weights):
    """Create a pie chart of theme allocations"""
    plt.figure(figsize=(10, 7))
    plt.pie(theme_weights, labels=theme_weights.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Portfolio Allocation by Theme')
    plt.tight_layout()
    plt.show()

def portfolio_allocator(theme_groups, portfolio_value, conviction_multipliers=None, show_plot=False):
    """Main function to allocate portfolio based on themes and risk weighting"""
    # Get all unique tickers
    all_tickers = []
    for tickers in theme_groups.values():
        all_tickers.extend(tickers)
    all_tickers = list(set(all_tickers))
    
    # Fetch historical data
    price_data = fetch_historical_data(all_tickers)
    
    # Calculate weights
    ticker_weights, theme_weights, theme_volatility = calculate_weights(
        theme_groups, price_data, conviction_multipliers
    )
    
    # Calculate allocation
    allocation = calculate_allocation(ticker_weights, price_data, portfolio_value)
    
    # Create theme summary
    theme_summary = pd.DataFrame({
        'Volatility': theme_volatility,
        'Weight %': (theme_weights * 100).round(2)
    })
    
    # Optional: Plot theme allocation
    if show_plot:
        plot_theme_allocation(theme_groups, theme_weights)
    
    return allocation, theme_summary

def main():
    # Example theme groups
    theme_groups = {
        'AI': ['CURI', 'GTLB', 'IDCC', 'NVDA', 'TSM', 'SMCI', 'ANET'],
        'Nuclear / Energy': ['BWXT', 'CEG', 'SMR', 'LEU', 'NEE', 'FSLR', 'NXT'],
        'Defense': ['BAH', 'GD', 'LMT'],
        'Gold': ['FNV', 'WPM', 'VOXR'],
        'Index': ['RSP', 'DIA'],
        'Great-Companies': ['AMZN'],
    }
    
    # Get portfolio value from user
    try:
        portfolio_value = float(input("Enter portfolio value ($): "))
    except ValueError:
        portfolio_value = 23429  # Default value
        print(f"Using default portfolio value: ${portfolio_value:,.2f}")

    # Optional: Ask about conviction multipliers
    use_conviction = input("Do you want to use conviction multipliers? (y/n): ").lower() == 'y'
    conviction_multipliers = None
    
    if use_conviction:
        conviction_multipliers = {}
        for theme in theme_groups.keys():
            try:
                multiplier = float(input(f"Enter conviction multiplier for {theme} (default: 1.0): ") or 1.0)
                conviction_multipliers[theme] = multiplier
            except ValueError:
                conviction_multipliers[theme] = 1.0
                
    # Get allocations        
    allocation, theme_summary = portfolio_allocator(
        theme_groups, portfolio_value, conviction_multipliers, show_plot=True
    )
    
    # Print results
    print("\n--- Theme Risk & Weights ---")
    print(theme_summary)
    
    print("\n--- Portfolio Allocation ---")
    print(allocation)
    
    # Option to export to CSV
    if input("Export allocation to CSV? (y/n): ").lower() == 'y':
        allocation.to_csv('portfolio_allocation.csv')
        print("Allocation exported to 'portfolio_allocation.csv'")

if __name__ == "__main__":
    main()