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

# --- Helper: Memory Layout Enforcer ---
# Fixes the "transposition" and "contiguity" issues transparently
cdef np.ndarray[double, ndim=2, mode='c'] _sanitize_input(object input_matrix):
    cdef np.ndarray[double, ndim=2] arr = np.asarray(input_matrix, dtype=np.float64)
    if arr.shape[0] > arr.shape[1]: 
        # Assume Time x Assets -> Transpose to Assets x Time
        return np.ascontiguousarray(arr.T)
    return np.ascontiguousarray(arr)

# --- 1. Asset-Specific Option Adjusted Generation ---
def generate_asset_option_adjusted(double[:] returns, double s0, double[:, :] option_chain):
    """
    returns: 1D array of log returns
    option_chain: Matrix [Strike, Expiry, Type, MarketPrice]
    """
    cdef int n = returns.shape[0]
    cdef int n_opts = option_chain.shape[0]
    cdef SVCJParams p
    
    # 1. Cast inputs to C-contiguous NumPy arrays (Fixes TypeError)
    cdef np.ndarray[double, ndim=1, mode='c'] c_returns = np.ascontiguousarray(returns)
    cdef np.ndarray[double, ndim=1, mode='c'] strikes = np.array(option_chain[:, 0], dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode='c'] expiries = np.array(option_chain[:, 1], dtype=np.float64)
    cdef np.ndarray[int, ndim=1, mode='c'] types = np.array(option_chain[:, 2], dtype=np.int32)
    
    # 2. Output Buffers
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    cdef np.ndarray[double, ndim=1] model_prices = np.zeros(n_opts)
    
    # 3. Clean & Optimize
    clean_returns(&c_returns[0], n)
    optimize_svcj(&c_returns[0], n, &p, &spot_vol[0], &jump_prob[0])
    
    # 4. Price using last fitted spot vol
    price_option_chain(s0, &strikes[0], &expiries[0], &types[0], n_opts, &p, spot_vol[n-1], &model_prices[0])
    
    return {
        "params": {
            "mu": p.mu, "kappa": p.kappa, "theta": p.theta, 
            "sigma_v": p.sigma_v, "rho": p.rho, "lambda_j": p.lambda_j
        },
        "spot_vol": spot_vol,
        "jump_prob": jump_prob,
        "model_prices": model_prices
    }

# --- 2. Market Wide Historic Rolling ---
def analyze_market_rolling(object market_returns, int window):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize_input(market_returns)
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_windows = n_days - window
    
    if n_windows < 1:
        raise ValueError("Window size larger than data length")

    # [Asset, Window, 4 Metrics]
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 4)) 
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                # Optimization on sliding window
                optimize_svcj(&data[i, w], window, &p, NULL, NULL)
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.lambda_j
                results[i, w, 2] = p.kappa
                results[i, w, 3] = p.sigma_v
                
    return results

# --- 3. Market Wide Current SVCJ Params ---
def analyze_market_current(object market_returns):
    cdef np.ndarray[double, ndim=2, mode='c'] data = _sanitize_input(market_returns)
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

    # Return as Transposed (Time, Asset) to match typical pandas DataFrame alignment
    return {
        "spot_vol": out_spot.T, 
        "jump_prob": out_jump.T, 
        "params": out_params
    }

# --- 4. Asset-Specific Drift-Realized Residue ---
def generate_residue_analysis(double[:] returns, int forward_window):
    cdef int n = returns.shape[0]
    cdef np.ndarray[double, ndim=1, mode='c'] c_returns = np.ascontiguousarray(returns)
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef SVCJParams p
    
    optimize_svcj(&c_returns[0], n, &p, &spot_vol[0], NULL)
    
    cdef int t
    for t in range(n - forward_window):
        # Forward Realized Return - Expected Drift (based on fit)
        # Residue implies 'Surprise' component
        residues[t] = c_returns[t + forward_window] - (p.mu * (1.0/252.0));
        
    return residues