# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_returns(double* returns, int n) nogil
    void optimize_svcj(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices) nogil

# --- 1. Helper: Process Raw Prices to Log Returns (2D) ---
# Input: (Time, Assets) Raw Prices
# Output: (Assets, Time-1) Log Returns [C-Contiguous for C-Core]
cdef np.ndarray[double, ndim=2, mode='c'] _process_prices_2d(object prices_input):
    cdef np.ndarray[double, ndim=2] prices = np.asarray(prices_input, dtype=np.float64)
    
    # Validation: Ensure Time > Assets (heuristic for YF format)
    if prices.shape[0] < prices.shape[1]:
        # If user passed (Assets, Time), transpose first
        prices = prices.T

    # Log Returns: ln(P_t) - ln(P_{t-1})
    # numpy handles this efficiently
    cdef np.ndarray[double, ndim=2] log_prices = np.log(prices)
    cdef np.ndarray[double, ndim=2] returns = np.diff(log_prices, axis=0)
    
    # Transpose to (Assets, Time) and enforce C-contiguity for the Loop
    return np.ascontiguousarray(returns.T)

# --- 2. Helper: Process Raw Prices to Log Returns (1D) ---
cdef np.ndarray[double, ndim=1, mode='c'] _process_prices_1d(object prices_input):
    cdef np.ndarray[double, ndim=1] prices = np.asarray(prices_input, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] log_prices = np.log(prices)
    cdef np.ndarray[double, ndim=1] returns = np.diff(log_prices)
    return np.ascontiguousarray(returns)


# --- Method 1: Asset-Specific Option Adjusted ---
def generate_asset_option_adjusted(double[:] prices, double s0, object option_chain):
    """
    prices: 1D array of raw prices (Time). Last element is Current Price.
    """
    # 1. Convert Prices to Returns
    cdef np.ndarray[double, ndim=1, mode='c'] c_ret = _process_prices_1d(prices)
    cdef int n = c_ret.shape[0]
    
    # 2. Handle Option Chain
    cdef np.ndarray arr = np.array(option_chain, copy=False)
    cdef int n_opts = arr.shape[0]
    cdef np.ndarray[double, ndim=1, mode='c'] ks = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode='c'] ts = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
    cdef np.ndarray[int, ndim=1, mode='c'] types = np.ascontiguousarray(arr[:, 2].astype(np.int32), dtype=np.int32)
    
    # 3. Alloc Outputs
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    cdef np.ndarray[double, ndim=1] model_prices = np.zeros(n_opts)
    
    cdef SVCJParams p
    clean_returns(&c_ret[0], n)
    optimize_svcj(&c_ret[0], n, &p, &spot_vol[0], &jump_prob[0])
    
    # Price using Current Spot Vol (Last element)
    price_option_chain(s0, &ks[0], &ts[0], &types[0], n_opts, &p, spot_vol[n-1], &model_prices[0])
    
    return {
        "kappa": p.kappa, "theta": p.theta, "sigma_v": p.sigma_v, "rho": p.rho,
        "lambda_j": p.lambda_j, "mu_j": p.mu_j, "sigma_j": p.sigma_j, "mu": p.mu,
        "spot_vol": spot_vol, "jump_prob": jump_prob, "model_prices": model_prices
    }

# --- Method 2: Market Wide Rolling ---
def analyze_market_rolling(object market_prices, int window):
    # Convert Matrix of Prices -> Matrix of Returns (Assets, Time)
    cdef np.ndarray[double, ndim=2, mode='c'] data = _process_prices_2d(market_prices)
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_windows = n_days - window
    
    if n_windows < 1: return None
    
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 5))
    cdef SVCJParams p
    cdef int i, w
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                optimize_svcj(&data[i, w], window, &p, NULL, NULL)
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.kappa
                results[i, w, 2] = p.sigma_v
                results[i, w, 3] = p.rho
                results[i, w, 4] = p.lambda_j

    return results

# --- Method 3: Market Wide Current ---
def analyze_market_current(object market_prices):
    # Convert Matrix of Prices -> Matrix of Returns (Assets, Time)
    cdef np.ndarray[double, ndim=2, mode='c'] data = _process_prices_2d(market_prices)
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_assets, n_days))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_assets, n_days))
    cdef np.ndarray[double, ndim=2] out_params = np.zeros((n_assets, 8))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            clean_returns(&data[i, 0], n_days)
            optimize_svcj(&data[i, 0], n_days, &p, &out_spot[i, 0], &out_jump[i, 0])
            
            out_params[i, 0] = p.kappa
            out_params[i, 1] = p.theta
            out_params[i, 2] = p.sigma_v
            out_params[i, 3] = p.rho
            out_params[i, 4] = p.lambda_j
            out_params[i, 5] = p.mu_j
            out_params[i, 6] = p.sigma_j
            out_params[i, 7] = p.mu

    # Returns (Time, Assets) to align with YFinance conventions
    return {
        "spot_vol": out_spot.T,
        "jump_prob": out_jump.T,
        "params": out_params
    }

# --- Method 4: Asset-Specific Drift-Realized Residue ---
def generate_residue_analysis(double[:] prices):
    """
    Input: Raw Prices (Time).
    Output: Residue Vector (Length = Time - 1).
    Residue[t] = RealizedReturn[t] - ModelDrift
    Last element is the residue for Today.
    """
    cdef np.ndarray[double, ndim=1, mode='c'] c_ret = _process_prices_1d(prices)
    cdef int n = c_ret.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n)
    cdef SVCJParams p
    
    clean_returns(&c_ret[0], n)
    optimize_svcj(&c_ret[0], n, &p, NULL, NULL)
    
    cdef double daily_drift = p.mu * (1.0/252.0)
    
    cdef int t
    for t in range(n):
        residues[t] = c_ret[t] - daily_drift
        
    return residues