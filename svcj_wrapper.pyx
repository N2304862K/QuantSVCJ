# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_returns(double* returns, int n, int stride) nogil
    void constrain_params(SVCJParams* params) nogil
    double run_ukf_likelihood(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void calibrate_to_history(double* returns, int n, int stride, SVCJParams* params) nogil
    void calibrate_to_options(double s0, double* strikes, double* expiries, int* types, double* mkt_prices, int n_opts, SVCJParams* params) nogil

# --- Helper to init safe seed params (Daily Scale) ---
cdef void init_params(SVCJParams* p) nogil:
    p.mu = 0.0004         # ~10% annual drift
    p.kappa = 0.01        # Mean reversion (slow daily)
    p.theta = 0.00015     # ~20% Annual Vol -> 0.00015 Daily Var
    p.sigma_v = 0.001     # Vol of Vol
    p.rho = -0.5          # Leverage effect
    p.lambda_j = 0.005    # Rare jumps
    p.mu_j = -0.01        # Downward jump bias
    p.sigma_j = 0.02

# --- 1. Asset-Specific Option Adjusted ---
def generate_asset_option_adjusted(double[:] returns, double s0, double[:, :] option_chain):
    cdef int n = returns.shape[0]
    cdef int n_opts = option_chain.shape[0]
    cdef SVCJParams params
    
    # 1. Init
    init_params(&params)
    clean_returns(&returns[0], n, 1)
    
    # 2. Calibrate Diffusive Params to History
    calibrate_to_history(&returns[0], n, 1, &params)
    
    # 3. Extract Option Data Columns (Contiguous buffers needed)
    cdef double[:] strikes = option_chain[:, 0].copy()
    cdef double[:] expiries = option_chain[:, 1].copy()
    cdef int[:] types = option_chain[:, 2].astype(np.int32).copy() # Cast to int
    cdef double[:] prices = option_chain[:, 3].copy()
    
    # 4. Calibrate Jump Params to Options
    calibrate_to_options(s0, &strikes[0], &expiries[0], &types[0], &prices[0], n_opts, &params)
    
    # 5. Generate Final Time Series
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    run_ukf_likelihood(&returns[0], n, 1, &params, &spot_vol[0], &jump_prob[0])
    
    return {
        "kappa": params.kappa, "theta": params.theta, "rho": params.rho,
        "sigma_v": params.sigma_v, "lambda_j": params.lambda_j,
        "spot_vol": spot_vol, "jump_prob": jump_prob
    }

# --- 2. Market Wide Current (Optimized) ---
def analyze_market_current(double[:, :] market_returns):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    cdef int stride = n_assets 
    
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_days, n_assets))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_days, n_assets))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            init_params(&p)
            clean_returns(&market_returns[0, i], n_days, stride)
            
            # OPTIMIZE per asset
            calibrate_to_history(&market_returns[0, i], n_days, stride, &p)
            
            # Generate outputs
            run_ukf_likelihood(&market_returns[0, i], n_days, stride, &p, &out_spot[0, i], &out_jump[0, i])
            
    return {"spot_vol": out_spot, "jump_prob": out_jump}

# --- 3. Market Wide Rolling (Optimized) ---
def analyze_market_rolling(double[:, :] market_returns, int window):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    cdef int n_windows = n_days - window
    cdef int stride = n_assets 
    
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 3)) 
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                init_params(&p)
                # Optimize ONLY on the window slice
                calibrate_to_history(&market_returns[w, i], window, stride, &p)
                
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.rho
    
    return results

# --- 4. Residue Analysis ---
def generate_residue_analysis(double[:] returns, int forward_window):
    cdef int n = returns.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef SVCJParams p
    
    init_params(&p)
    clean_returns(&returns[0], n, 1)
    calibrate_to_history(&returns[0], n, 1, &p)
    
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    run_ukf_likelihood(&returns[0], n, 1, &p, &spot_vol[0], NULL)
    
    cdef int t
    for t in range(n - forward_window):
        # Expected return over window approx
        double drift = (p.mu - 0.5 * sq(spot_vol[t])) * forward_window;
        residues[t] = returns[t + forward_window] - drift;
        
    return residues