# SVCJ Factor Engine (UKF-QMLE & OpenMP Upgrade)

This repository contains a high-performance C-Extension engine for the **Stochastic Volatility with Correlated Jumps (SVCJ)** model (specifically the Bates model variant). 

It has been upgraded to replace heuristic estimation with **Unscented Kalman Filter (UKF) QMLE**, utilizing **OpenMP** for parallel market screening and **Fourier Transforms** for option pricing.

## Key Features

1.  **Algorithmic Core:** Uses **UKF-QMLE** (Unscented Kalman Filter Quasi-Maximum Likelihood Estimation) to rigorously separate continuous volatility from discrete jumps.
2.  **Mathematical Stability:** Enforces the **Feller Condition** ($2\kappa\theta > \sigma_v^2$) and positivity constraints to prevent numerical explosion.
3.  **Latent State Extraction:** Outputs time-series vectors for **Spot Volatility** ($v_t$) and **Instantaneous Jump Probability** ($P(J_t)$).
4.  **High-Performance Scaling:** Implements **OpenMP** parallelization in the C-kernel, allowing market-wide screening (500+ assets) in seconds by releasing the Python GIL.
5.  **Robustness:** Includes internal denoising for 0.00% returns (illiquidity handling).
6.  **Option Support:** Native **Carr-Madan** pricing for arbitrary chains of Calls and Puts.

## Prerequisites

To build this engine, you must have:
*   **C Compiler:** `gcc` (Linux/macOS) or MSVC (Windows).
*   **Python Dev Headers:** Usually included with Python, or `python-dev` on Linux.
*   **OpenMP Support:** 
    *   *Linux:* Standard `libgomp` (usually pre-installed).
    *   *Mac:* `libomp` (install via `brew install libomp`).
    *   *Windows:* Supported natively by MSVC.

## Installation

This package is **not** a standard pip-installable library. It compiles a C extension in place to maximize speed.

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Compile the Engine:**
    Run the setup script to build the shared object file (`svcj_wrapper.so` or `.pyd`) in the current directory.
    ```bash
    python setup.py build_ext --inplace
    ```

    *Success Check:* Ensure a file named `svcj_wrapper...so` (Linux/Mac) or `svcj_wrapper...pyd` (Windows) appears in your folder.

## Usage

Once compiled, you can import `svcj_wrapper` directly in Python.

### 1. High-Performance Market Screening (Parallel)
Analyze hundreds of assets simultaneously using the OpenMP-enabled `analyze_market_screen`.

```python
import numpy as np
import pandas as pd
import svcj_wrapper

# 1. Prepare Data: Matrix of Log Returns (Assets x Time)
# Shape: (500 Assets, 1000 Time Steps)
n_assets = 500
n_days = 1000
market_returns = np.random.normal(0, 0.01, size=(n_assets, n_days)).astype(np.float64)

# 2. Run Analysis
# Returns two matrices: Spot Volatility and Jump Probability
print("Running UKF-QMLE on 500 assets...")
spot_vols, jump_probs = svcj_wrapper.analyze_market_screen(market_returns)

# 3. Process Results
# Example: Get the jump probability time series for Asset #0
asset_0_jumps = jump_probs[0, :]

print(f"Asset 0 Avg Volatility: {np.mean(spot_vols[0, :]):.4f}")
print(f"Asset 0 Max Jump Prob:  {np.max(asset_0_jumps):.4f}")
```

### 2. Option Pricing (Carr-Madan)
Price mixed chains of Calls and Puts efficiently.

```python
import numpy as np
import svcj_wrapper

# Market Params
S0 = 100.0   # Spot Price
r = 0.05     # Risk-free rate
q = 0.0      # Dividend yield
T = 1.0      # Time to maturity (years)

# Option Chain Definition
# Strikes: [95, 100, 105]
# Types:   [0 (Put), 1 (Call), 1 (Call)]
strikes = np.array([95.0, 100.0, 105.0], dtype=np.float64)
types = np.array([0, 1, 1], dtype=np.int32) 

# Calculate Prices
# The engine uses internal Bates model params (calibrated or default)
prices = svcj_wrapper.price_option_chain(S0, r, q, T, strikes, types)

for K, type_flag, price in zip(strikes, types, prices):
    opt_type = "Call" if type_flag == 1 else "Put"
    print(f"{opt_type} Strike {K}: ${price:.4f}")
```

## File Structure

*   `svcj.c`: Core C logic (UKF, Feller Check, Denoising).
*   `svcj.h`: Header definitions for structs and prototypes.
*   `svcj_wrapper.pyx`: Cython interface exposing C logic to Python.
*   `setup.py`: Build script with OpenMP flags.
*   `requirements.txt`: Python dependencies.