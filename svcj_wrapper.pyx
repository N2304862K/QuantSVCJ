# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# Link to C
cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_returns(double* returns, int n) nogil
    void optimize_svcj(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices) nogil

# --- 1. Asset-Specific Option Adjusted Generation ---
def generate_asset_option_adjusted(object returns_in, double s0, object option_chain_in):
    # 1. Safe Input Casting
    cdef np.ndarray[double, ndim=1, mode='c'] returns = np.ascontiguousarray(returns_in, dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode='c'] chain = np.ascontiguousarray(option_chain_in, dtype=np.float64)
    
    cdef int n = returns.shape[0]
    cdef int n_opts = chain.shape[0]
    cdef SVCJParams p
    
    # 2. Allocate Outputs
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] model_prices = np.zeros(n_opts, dtype=np.float64)
    
    # 3. Safe Extraction of Columns (Fixing the TypeError)
    # forcing copy/contiguous makes it a safe separate array
    cdef np.ndarray[double, ndim=1, mode='c'] strikes = np.ascontiguousarray(chain[:, 0])
    cdef np.ndarray[double, ndim=1, mode='c'] expiries = np.ascontiguousarray(chain[:, 1])
    # Cast types to int
    cdef np.ndarray[int, ndim=1, mode='c'] types = np.ascontiguousarray(chain[:, 2], dtype=np.int32)

    # 4. Execution
    clean_returns(&returns[0], n)
    optimize_svcj(&returns[0], n, &p, &spot_vol[0], &jump_prob[0])
    
    # Price using the latest spot volatility
    price_option_chain(s0, &strikes[0], &expiries[0], &types[0], n_opts, &p, spot_vol[n-1], &model_prices[0])
    
    return {
        "params": {"kappa": p.kappa, "theta": p.theta, "lambda_j": p.lambda_j, "rho": p.rho},
        "spot_vol": spot_vol,
        "jump_prob": jump_prob,
        "model_prices": model_prices
    }

# --- 2. Market Wide Historic Rolling ---
def analyze_market_rolling(object market_returns_in, int window):
    # Heuristic Transpose: Ensure (Assets, Time)
    cdef np.ndarray[double, ndim=2] arr_in = np.ascontiguousarray(market_returns_in, dtype=np.float64)
    if arr_in.shape[0] > arr_in.shape[1]: 
        arr_in = np.ascontiguousarray(arr_in.T) # Flip to (Assets, Time)
        
    cdef np.ndarray[double, ndim=2, mode='c'] data = arr_in
    cdef int n_assets = data.shape[0]
    cdef int n_days = data.shape[1]
    cdef int n_windows = n_days - window
    
    if n_windows < 1:
        raise ValueError("Window size larger than data length.")

    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 2)) # storing theta, lambda
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            for w in range(n_windows):
                # Optimization on the window slice
                optimize_svcj(&data[i, w], window, &p, NULL, NULL)
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.lambda_j
                
    return results

# --- 3. Market Wide Current SVCJ Params ---
def analyze_market_current(object market_returns_in):
    # Heuristic Transpose
    cdef np.ndarray[double, ndim=2] arr_in = np.ascontiguousarray(market_returns_in, dtype=np.float64)
    if arr_in.shape[0] > arr_in.shape[1]: 
        arr_in = np.ascontiguousarray(arr_in.T)
        
    cdef np.ndarray[double, ndim=2, mode='c'] data = arr_in
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
            
    return {
        "spot_vol": out_spot.T,  # Return as (Time, Assets) to match input
        "jump_prob": out_jump.T,
        "params": out_params
    }

# --- 4. Residue Analysis ---
def generate_residue_analysis(object returns_in, int forward_window):
    cdef np.ndarray[double, ndim=1, mode='c'] returns = np.ascontiguousarray(returns_in, dtype=np.float64)
    cdef int n = returns.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef SVCJParams p
    
    clean_returns(&returns[0], n)
    optimize_svcj(&returns[0], n, &p, &spot_vol[0], NULL)
    
    cdef int t
    for t in range(n - forward_window):
        residues[t] = returns[t + forward_window] - (p.mu - 0.5 * spot_vol[t]*spot_vol[t]);
        
    return residues