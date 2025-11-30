# SVCJ Factor Engine (UKF-QMLE & OpenMP Enhanced)

A high-performance implementation of the Stochastic Volatility with Correlated Jumps (SVCJ/Bates) model. This engine has been upgraded to use **Unscented Kalman Filters (UKF)** for latent state extraction and **OpenMP** for parallel market-wide screening.

It is designed to extract "Spot Volatility" and "Instantaneous Jump Probabilities" from time-series data while enforcing mathematical stability (Feller conditions).

## Key Features

1.  **Algorithmic Solver (UKF-QMLE):** Replaces heuristic estimation with a rigorous Unscented Kalman Filter to separate continuous volatility from discrete jumps.
2.  **High-Performance Scaling:** Utilizes OpenMP in the C-kernel to analyze 500+ assets in seconds (bypassing the Python GIL).
3.  **Latent State Extraction:** Outputs time-series vectors for $v_t$ (Spot Vol) and $P(J_t)$ (Jump Probability) rather than just static parameters.
4.  **Robustness:** Internal "Zero-Return Denoising" prevents numerical instability on illiquid days or trading halts.
5.  **Option Pricing:** Built-in Carr-Madan Fourier integration for pricing arbitrary option chains (Calls/Puts).

## Installation

This is a C-extension module. You must compile the engine locally before usage.

### Prerequisites
*   Python 3.8+
*   C Compiler (GCC on Linux/Mac, MSVC on Windows)
*   OpenMP libraries (usually included with GCC; on Mac via `brew install libomp`)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install yfinance matplotlib  # Additional requirements for the demo
```

### 2. Compile the Engine
Run the setup script to build the shared object file (`svcj_wrapper.so` or `.pyd`) in place.

```bash
python setup.py build_ext --inplace
```

*Note: If you are on macOS and encounter clang errors regarding OpenMP, ensure you have `libomp` installed and your environment variables are set correctly.*

## Project Structure

*   `svcj_wrapper.pyx`: Cython interface bridging Numpy and C. Handles OpenMP parallel loops.
*   `svcj.c`: The core mathematical kernel (UKF logic, Feller checks, Likelihood calculations).
*   `svcj.h`: Header definitions for Bates model structures.
*   `setup.py`: Build configuration with compiler flags for parallelization.
*   `requirements.txt`: Package dependencies.

## Usage

### 1. Market Screening (Parallel)
The engine accepts a 2D matrix of log returns `(n_assets, time_steps)` and returns latent state matrices.

```python
import svcj_wrapper
import numpy as np

# Load your data (n_assets x time_steps)
# returns_matrix = ... 

# Run parallel extraction
spot_vols, jump_probs = svcj_wrapper.analyze_market_screen(returns_matrix)
```

### 2. Option Pricing
Price mixed chains of Puts and Calls using the internal characteristic function integration.

```python
# Strike prices and types (1=Call, 0=Put)
strikes = np.array([100.0, 105.0, 95.0], dtype=np.float64)
types = np.array([1, 1, 0], dtype=np.int32) 

# Price chain: S0=100, r=0.05, q=0.02, T=1.0
prices = svcj_wrapper.price_option_chain(100.0, 0.05, 0.02, 1.0, strikes, types)
```

## License
MIT


---

### 2. run.py (Demo Script)

This script demonstrates fetching data from Yahoo Finance, preparing the data structure, and running the compiled C-engine.

```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import svcj_wrapper  # This imports the compiled C extension
import time

def main():
    print("--- SVCJ Factor Engine Demo ---")
    
    # 1. Configuration
    tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'NVDA', 'AAPL']
    period = "2y"
    
    print(f"[1] Downloading data for {len(tickers)} assets from Yahoo Finance...")
    data = yf.download(tickers, period=period, progress=False)['Adj Close']
    
    # 2. Pre-process Data
    # Calculate Log Returns: ln(Pt / Pt-1)
    # Fill NaN with 0.0 (The C engine handles 0.0 via internal denoising)
    log_returns = np.log(data / data.shift(1)).fillna(0.0)
    
    # Transpose to shape (N_Assets, Time_Steps) required by the C engine
    # Ensure C-contiguous memory layout for performance
    market_matrix = np.ascontiguousarray(log_returns.values.T, dtype=np.float64)
    
    n_assets, t_steps = market_matrix.shape
    print(f"[2] Data Prepared: {n_assets} Assets over {t_steps} trading days.")
    
    # 3. Run Engine (Parallelized UKF)
    print(f"[3] Running UKF-QMLE Engine (OpenMP Parallelized)...")
    start_time = time.time()
    
    # --- CORE CALL TO C EXTENSION ---
    spot_vols, jump_probs = svcj_wrapper.analyze_market_screen(market_matrix)
    # --------------------------------
    
    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.4f} seconds.")
    print(f"    Processed {n_assets * t_steps} data points.")

    # 4. Visualization of Results
    print("[4] Visualizing results for NVDA (Example)...")
    
    # Find index of NVDA
    asset_idx = tickers.index('NVDA')
    
    dates = log_returns.index
    asset_ret = market_matrix[asset_idx]
    asset_vol = spot_vols[asset_idx]
    asset_jump = jump_probs[asset_idx]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Returns
    ax1.plot(dates, asset_ret, color='grey', alpha=0.6, label='Log Returns')
    ax1.set_title(f"NVDA: Market Returns")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Extracted Spot Volatility (Latent State)
    ax2.plot(dates, asset_vol, color='blue', lw=1.5, label='Extracted Spot Vol (UKF)')
    ax2.set_title("Latent Spot Volatility (Filtered)")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Instantaneous Jump Probability
    ax3.bar(dates, asset_jump, color='red', width=2, label='Jump Probability P(J_t)')
    ax3.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax3.set_title("Instantaneous Jump Probability")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("    Plot generated. Close window to continue.")
    plt.show()

    # 5. Option Pricing Demo
    print("\n[5] Option Pricing Demo (Carr-Madan)")
    S0 = 100.0
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0], dtype=np.float64)
    # 0 = Put, 1 = Call
    types = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    
    print(f"    Pricing chain for S0={S0}...")
    prices = svcj_wrapper.price_option_chain(S0, 0.05, 0.0, 0.5, strikes, types)
    
    df_opts = pd.DataFrame({
        'Strike': strikes,
        'Type': ['Put' if t==0 else 'Call' for t in types],
        'Model_Price': prices
    })
    print(df_opts)

if __name__ == "__main__":
    main()