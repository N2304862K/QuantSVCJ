# QuantSVCJ: High-Performance SVCJ Factor Engine

**QuantSVCJ** is a C-based estimation engine for the **Stochastic Volatility with Correlated Jumps (SVCJ)** model, designed for quantitative finance workflows. 

Unlike standard libraries that rely on slow Python loops or heuristic approximations (Method of Moments), this engine implements **State-Space Filtering (Unscented Kalman Filter)** and **Joint Likelihood Optimization** directly in C. It uses OpenMP to parallelize multi-asset processing, making it suitable for generating volatility and jump features for large universes of assets.

## Core Features

1.  **Mathematical Rigor**:
    *   **Time Series**: Replaces heuristics with an **Unscented Kalman Filter (UKF)** to recover latent volatility ($v_t$) and jump likelihoods.
    *   **Options**: Implements joint calibration using characteristic functions (via complex arithmetic in C).
2.  **Performance**:
    *   **No Python Loops**: All iterative logic (optimization, filtering) is buried in low-level C.
    *   **Parallelized**: Multi-asset feature generation uses OpenMP to saturate CPU cores.
3.  **Desk Ready**: 
    *   Separates data fetching (Yahoo Finance/Bloomberg) from analytics.
    *   Returns clean Pandas structures or raw Numpy arrays.

## Installation

Install directly from GitHub (requires C compiler and OpenMP):

```bash
pip install git+https://github.com/YourUsername/QuantSVCJ.git

## Usage Examples

Data fetching uses `yfinance` purely for demonstration. The engine accepts standard Pandas/Numpy inputs.

### 1. Single Asset Joint Calibration (Snapshot)
Combines historical log returns with today's option chain for precise parameter estimation.

```python
import yfinance as yf
import pandas as pd
import numpy as np
from quantsvcj.api import QuantSVCJ

# --- Data Fetching ---
ticker = "SPY"
# 1. History
hist = yf.download(ticker, period="1y")['Close']
log_rets = np.log(hist / hist.shift(1)).dropna()

# 2. Option Chain (Mocking real chain for demo)
# In production: opt = yf.Ticker("SPY").option_chain('2024-12-20').calls
current_price = hist.iloc[-1]
strikes = np.linspace(current_price*0.9, current_price*1.1, 10)
prices = np.maximum(0, strikes - current_price) + np.random.uniform(0.5, 2.0, 10) # Mock prices
options = pd.DataFrame({
    'strike': strikes,
    'price': prices,
    'T': [0.1] * 10 # 0.1 years to expiry
})

# --- Execution ---
# Input: Series (History), DataFrame (Options), Rate
params = QuantSVCJ.analyze_snapshot(
    returns_series=log_rets,
    option_chain=options,
    risk_free_rate=0.05
)

print(params)
# Output: kappa, theta, sigma_v, rho...
```

### 2. Rolling Window Analysis
Generates a time-series of factors (e.g., Jump Intensity $\lambda_t$) for backtesting.

```python
# Input: Series, Window Size
# Returns: DataFrame indexed by Date
rolling_factors = QuantSVCJ.analyze_rolling(
    returns_series=log_rets,
    window_size=100
)

print(rolling_factors.tail())
# Output:
# Date        kappa    theta    lambda ...
# 2023-11-20  1.52     0.041    0.15
# 2023-11-21  1.55     0.042    0.18
```

### 3. Market-Wide Screen (Multi-Asset)
Screen the entire S&P 500 for volatility/jump regimes in milliseconds using parallel C execution.

```python
# --- Data Fetching ---
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
data = yf.download(tickers, period="2y")['Close']
returns_matrix = np.log(data / data.shift(1)).dropna()

# --- Execution ---
# Input: DataFrame (Index=Date, Cols=Assets)
# Returns: DataFrame (Index=Assets, Cols=Params)
screen_results = QuantSVCJ.analyze_market_screen(returns_matrix)

print(screen_results)
# Output:
#       kappa  theta  sigma_v  lambda ...
# AAPL  2.1    0.05   0.32     0.10
# MSFT  1.8    0.04   0.28     0.05
```