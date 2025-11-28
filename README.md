# QuantSVCJ: Professional Factor Engine

A high-performance C-Core engine for SVCJ calibration. Designed for quantitative desks, it handles raw price data, automates Spot ($S_0$) detection, and stabilizes edge-case volatility estimates.

## Installation
```bash
pip install .
```

## Usage (Live Data)

This engine is designed to take raw **Prices** and **Option Chains**. It handles the return calculations and Spot extraction internally.

```python
import yfinance as yf
import pandas as pd
from quantsvcj.api import QuantSVCJ

# --- 1. Snapshot Analysis (History + Options) ---
# Download Prices (NOT returns)
spy_prices = yf.download("SPY", period="1y", progress=False)['Close']
if isinstance(spy_prices, pd.DataFrame): spy_prices = spy_prices.iloc[:,0]

# Download Option Chain
tk = yf.Ticker("SPY")
# Get first expiry
exp = tk.options[0]
opts = tk.option_chain(exp).calls
# Format needed: strike, price, T
# Calculate T in years
T_years = (pd.Timestamp(exp) - pd.Timestamp.now()).days / 365.0
opt_df = pd.DataFrame({
    'strike': opts['strike'],
    'price': opts['lastPrice'], # or mid
    'T': T_years
}).dropna()

# RUN: Pass raw prices + option DF
# Engine detects S0 = spy_prices.iloc[-1] automatically.
res = QuantSVCJ.analyze_snapshot(spy_prices, opt_df)
print("Snapshot Params:\n", res)

# --- 2. Rolling Analysis ---
# Pass raw prices, engine handles log-ret conversion and rolling window
rolling = QuantSVCJ.analyze_rolling(spy_prices, window=100)
print("\nRolling Factors (Tail):\n", rolling.tail())

# --- 3. Market Screen ---
# Pass raw price matrix
tickers = ["AAPL", "MSFT", "GOOG"]
df_multi = yf.download(tickers, period="6mo", progress=False)['Close']

screen = QuantSVCJ.analyze_market_screen(df_multi)
print("\nMarket Screen:\n", screen)
```

## Technicals
*   **Edge Stabilization**: The UKF uses an adaptive innovation threshold based on running variance to prevent false jump flags on the most recent data point.
*   **Feller Condition**: Enforced via soft penalty in the C-optimizer.
*   **Parallelism**: OpenMP used for Screening and Rolling windows.