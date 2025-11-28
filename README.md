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

---

## Installation

Clone the repository and install in editable mode (requires a C compiler like `gcc` or MSVC).

```bash
git clone https://github.com/YourRepo/QuantSVCJ.git
cd QuantSVCJ
pip install .