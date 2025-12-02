# QuantSVCJ: High-Performance Factor Engine

**QuantSVCJ** is a heavily optimized, C-kernel engine for estimating **Stochastic Volatility with Correlated Jumps (SVCJ)** parameters.

Unlike the original reference repository, this version implements a **Bayesian Maximum A Posteriori (MAP)** estimator within the C layer to solve numerical instability (diffusion collapse), enforces strict memory contiguity for high-speed linear algebra, and handles all data transformations internally.

## 🚀 Key Differences & Improvements

| Feature | Original Implementation | **QuantSVCJ (This Repo)** |
| :--- | :--- | :--- |
| **Input Data** | Required pre-calculated Log Returns | **Raw Prices (Time $\times$ Assets)**. Internal C-level log-return calc. |
| **Optimization** | Simple Coordinate Descent (Prone to local minima) | **Nelder-Mead Simplex** with Bayesian Priors. |
| **Jump Detection** | Static Thresholds (Static output) | **Phenotypic Mixing & Bayesian Filter**. Dynamic $P(J\|y)$ calculation. |
| **Stability** | Prone to "Diffusion Collapse" (Vol $\to$ 0) | **Robust**. Uses MAP estimation to penalize unrealistic variance decay. |
| **Architecture** | Mixed Python/C Logic | **Strict "Matrix In / Matrix Out"**. All math happens in C (No GIL). |
| **Parallelism** | Serial Execution | **OpenMP Parallelized**. Scales across all CPU cores for rolling windows. |
| **Option Pricing** | Inconsistent Black-Scholes | **Merton Jump Diffusion**. Consistent with time-series calibration. |

---

## 🛠 Installation

The engine requires a C compiler (GCC/MSVC) and Python development headers.

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Compile the C-Kernel:**
    ```bash
    python setup.py build_ext --inplace
    ```
    *This generates the `svcj_wrapper` shared object file optimized for your hardware.*

---

## 🧠 Core Methodology: The "Diffusion Collapse" Fix

A common failure mode in SVCJ estimation is **Diffusion Collapse**, where the optimizer sets the diffusive volatility to near-zero, attributing *all* market movement to jumps.

**QuantSVCJ solves this via:**
1.  **Phenotypic Mixing:** When calculating the probability of a jump, the filter uses a "Robust Variance" floor ($0.25 \times \theta$) rather than instantaneous spot volatility. This prevents the "Small Variance Trap."
2.  **Bayesian MAP Estimation:** The C-kernel includes log-normal priors on Long-Run Variance ($\theta$) and Jump Size ($\sigma_j$). This penalizes the optimizer if it attempts to fit chemically impossible parameters.

---

## 💻 Usage

The engine is designed for **Zero Data Manipulation** in Python. You pass raw price matrices (NumPy arrays), and the engine handles log-return calculation, memory layout sanitization, and parallel processing.

### 1. Market-Wide Current Snapshot
Generate Spot Volatility and Jump Probabilities for a basket of assets for the entire history.

```python
import svcj_wrapper

# Input: Raw Prices Matrix (Rows=Time, Cols=Assets)
# The engine automatically handles Log Return calculation
results = svcj_wrapper.analyze_market_current(raw_prices_matrix)

# Returns aligned Transposed Matrices (Time x Assets)
spot_vols = results['spot_vol']  # Annualized Diffusive Volatility
jump_probs = results['jump_prob'] # Instantaneous Probability of Jump
params = results['params']        # Fitted SVCJ Parameters
```

### 2. Historic Rolling Analysis (OpenMP)
Perform rolling window calibration across thousands of days and assets in parallel.

```python
# Input: Raw Prices Matrix, Window Size (e.g., 126 days)
# Output: 3D Tensor [Asset, Window, Parameter_Index]
rolling_res = svcj_wrapper.analyze_market_rolling(raw_prices_matrix, window=126)

# Access Theta (Long Run Var) for Asset 0 across all windows
theta_series = rolling_res[0, :, 0]
```

### 3. Asset-Specific Option Calibration
Calibrate parameters using both Historic Time-Series and Current Option Chain (Risk-Neutral adjustment).

```python
# Input: 1D Array of Prices, Current Spot, Option Chain Matrix
# Option Chain Cols: [Strike, Expiry (Years), Type (1=Call, -1=Put), MarketPrice]
calib = svcj_wrapper.generate_asset_option_adjusted(
    spy_prices, 
    current_spot, 
    option_chain_matrix
)

print(f"Calibrated Lambda: {calib['lambda_j']}")
print(f"Model Option Prices: {calib['model_prices']}")
```

### 4. Alpha/Residue Generation (Forward Targeting)
Extract the "Surprise" component (Realized Return - Model Expected Drift).

```python
# Input: 1D Array of Prices
# Output: 1D Array of Residues (aligned to T-1)
# Residue[t] = Realized_Return[t] - SVCJ_Drift[t]
residues = svcj_wrapper.generate_residue_analysis(asset_prices)

print(f"Today's Alpha: {residues[-1]}")
```

---

## 📂 File Structure

*   `svcj.c`: The computational core. Contains the UKF, Nelder-Mead Optimizer, and Bayesian Priors.
*   `svcj.h`: Struct definitions and function prototypes.
*   `svcj_wrapper.pyx`: Cython interface. Handles memory sanitization, OpenMP loop release, and Python type conversion.
*   `setup.py`: Build script with OpenMP flags detected by OS.
*   `run.py`: Demo script utilizing live data from `yfinance`.
