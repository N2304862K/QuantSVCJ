!pip install git+https://github.com/N2304862K/QuantSVCJ.git
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import svcj_wrapper

def get_formatted_option_chain(ticker_symbol):
    tk = yf.Ticker(ticker_symbol)
    exps = tk.options
    
    chain_data = []
    # Fetch first 2 expirations for demo speed
    for date_str in exps[:2]:
        opt = tk.option_chain(date_str)
        exp_date = datetime.strptime(date_str, '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        T = max(days_to_exp / 365.0, 0.001)
        
        # Calls (Type = 1)
        for _, row in opt.calls.iterrows():
            price = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
            chain_data.append([row['strike'], T, 1, price])
            
        # Puts (Type = -1)
        for _, row in opt.puts.iterrows():
            price = (row['bid'] + row['ask']) / 2 if row['bid'] > 0 else row['lastPrice']
            chain_data.append([row['strike'], T, -1, price])
            
    return np.array(chain_data)

def main():
    assets = ['SPY', 'QQQ', 'IWM', 'GLD']
    print(f"--- Fetching History for {assets} ---")
    
    # 1. Fetch Raw Prices (Time x Assets)
    df = yf.download(assets, period="2y", progress=False)['Close']
    raw_prices = df.values
    dates = df.index[1:] # Returns are T-1 length
    
    print(f"Data Shape: {raw_prices.shape}\n")

    # ======================================================
    # Method 1: Asset-Specific Option Adjusted
    # ======================================================
    target = 'SPY'
    print(f"--- 1. Option Adjusted Calibration ({target}) ---")
    
    target_prices = df[target].values
    s0 = target_prices[-1]
    opt_matrix = get_formatted_option_chain(target)
    
    if len(opt_matrix) > 0:
        res1 = svcj_wrapper.generate_asset_option_adjusted(target_prices, s0, opt_matrix)
        
        print(f"Params: Kappa={res1['kappa']:.4f}, Theta={res1['theta']:.4f}, Rho={res1['rho']:.4f}")
        print(f"Implied Jump Intensity (Lambda): {res1['lambda_j']:.4f}")
        print(f"Current Diffusive Vol: {res1['spot_vol'][-1]:.4f}")
    else:
        print("No option data found, skipping...")

    # ======================================================
    # Method 2: Market Wide Rolling
    # ======================================================
    print(f"\n--- 2. Market Wide Rolling Analysis (Window=126) ---")
    
    # Input: Raw Prices Matrix
    # Output: 3D Array [Asset, Window, Params]
    res2 = svcj_wrapper.analyze_market_rolling(raw_prices, 126)
    
    # Example: Show Theta (Long Run Var) for SPY (Index 0)
    spy_theta_rolling = res2[0, :, 0] 
    print(f"Rolling Windows Processed: {len(spy_theta_rolling)}")
    print(f"Latest Window Theta (SPY): {spy_theta_rolling[-1]:.6f}")

    # ======================================================
    # Method 3: Market Wide Current
    # ======================================================
    print(f"\n--- 3. Market Wide Current Parameters ---")
    
    res3 = svcj_wrapper.analyze_market_current(raw_prices)
    
    # Convert Spot Vol to DataFrame
    vol_df = pd.DataFrame(res3['spot_vol'], index=dates, columns=assets)
    jump_df = pd.DataFrame(res3['jump_prob'], index=dates, columns=assets)
    
    print("Latest Spot Volatility (Annualized):")
    print(vol_df.tail(3))
    print("\nLatest Jump Probability:")
    print(jump_df.tail(3))

    # ======================================================
    # Method 4: Drift Residue Analysis
    # ======================================================
    print(f"\n--- 4. Residue Analysis ({target}) ---")
    
    # Input: Raw Prices 1D
    # Output: Residue 1D (Return - Drift) aligned to T-1
    residues = svcj_wrapper.generate_residue_analysis(target_prices)
    
    res_series = pd.Series(residues, index=dates)
    
    print(f"Last 3 Days Residue (Realized Return - SVCJ Drift):")
    print(res_series.tail(3))
    print(f"Today's Alpha/Surprise: {res_series.iloc[-1]:.6f}")

if __name__ == "__main__":
    main()
