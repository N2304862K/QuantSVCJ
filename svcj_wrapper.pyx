# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# --- C Interface ---
cdef extern from "svcj_kernel.h":
    ctypedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double lambda_j
        double mu_j
        double sigma_j
        double v0

    ctypedef struct FilterState:
        double spot_vol
        double jump_prob
        double drift_residue

    ctypedef struct OptionContract:
        double strike
        double price
        double T
        int is_call

    void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params) nogil
    void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states) nogil
    void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params) nogil

# --- Python Interface ---

# 1. Asset-Specific Option Adjusted Analysis
def fit_asset_options(np.ndarray[double, ndim=1] log_returns, 
                      np.ndarray[double, ndim=2] option_chain, 
                      double spot_price):
    """
    Input: Log returns (history), Option Chain [Strike, Price, T, IsCall(0/1)]
    Output: Dict of Risk Params, Drift, Spot Vol Series, Jump Prob Series
    """
    cdef int n_steps = log_returns.shape[0]
    cdef int n_opts = option_chain.shape[0]
    cdef SVCJParams params
    cdef OptionContract* c_opts = <OptionContract*> malloc(n_opts * sizeof(OptionContract))
    cdef FilterState* states = <FilterState*> malloc(n_steps * sizeof(FilterState))
    
    # Pack Options
    for i in range(n_opts):
        c_opts[i].strike = option_chain[i, 0]
        c_opts[i].price = option_chain[i, 1]
        c_opts[i].T = option_chain[i, 2]
        c_opts[i].is_call = <int>option_chain[i, 3]

    # 1. Calibrate Risk Neutral Params to Options
    calibrate_to_options(c_opts, n_opts, spot_price, &params)
    
    # 2. Run UKF on history using these params to get Spot Vol / Jump Prob
    run_ukf_filter(&log_returns[0], n_steps, &params, states)

    # Unpack
    spot_vols = np.zeros(n_steps)
    jump_probs = np.zeros(n_steps)
    
    for i in range(n_steps):
        spot_vols[i] = states[i].spot_vol
        jump_probs[i] = states[i].jump_prob

    free(c_opts)
    free(states)

    return {
        "params": {"kappa": params.kappa, "theta": params.theta, "rho": params.rho},
        "spot_vol": spot_vols,
        "jump_prob": jump_probs
    }

# 2. Market Wide Historic Rolling (OpenMP Parallel)
def generate_historic_rolling(np.ndarray[double, ndim=2] market_returns, int window):
    """
    Input: Matrix (Assets x Time), Window Size
    Output: Matrix (Assets x Time x Params) - Simplified to just Spot Vol for demo
    """
    cdef int n_assets = market_returns.shape[0]
    cdef int n_time = market_returns.shape[1]
    cdef int i, t
    
    # Output: Spot Vol Matrix
    cdef np.ndarray[double, ndim=2] result = np.zeros((n_assets, n_time))
    
    cdef SVCJParams p
    cdef FilterState* states
    
    # Release GIL for parallel processing across assets
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            states = <FilterState*> malloc(n_time * sizeof(FilterState))
            
            # Simple rolling optimization logic would go here
            # For speed, we fit once on whole history and filter
            optimize_params_history(&market_returns[i, 0], n_time, &p)
            run_ukf_filter(&market_returns[i, 0], n_time, &p, states)
            
            for t in range(n_time):
                result[i, t] = states[t].spot_vol
            
            free(states)
            
    return result

# 3. Market Wide Current Snapshot (Spot Vol & Jump Prob)
def generate_current_screen(np.ndarray[double, ndim=2] market_returns):
    """
    Input: Matrix (Assets x History)
    Output: Matrix (Assets x 2) -> [Latest Spot Vol, Latest Jump Prob]
    """
    cdef int n_assets = market_returns.shape[0]
    cdef int n_time = market_returns.shape[1]
    cdef int i
    cdef np.ndarray[double, ndim=2] output = np.zeros((n_assets, 2))
    
    cdef SVCJParams p
    cdef FilterState* states
    
    with nogil:
        for i in prange(n_assets):
            states = <FilterState*> malloc(n_time * sizeof(FilterState))
            optimize_params_history(&market_returns[i, 0], n_time, &p)
            run_ukf_filter(&market_returns[i, 0], n_time, &p, states)
            
            output[i, 0] = states[n_time-1].spot_vol
            output[i, 1] = states[n_time-1].jump_prob
            
            free(states)
            
    return output

# 4. Asset-Specific Forward Targeting (Drift Residue)
def calculate_drift_residue(np.ndarray[double, ndim=1] log_returns):
    """
    Input: Log returns
    Output: Vector of realized residues (Return - Expected Drift)
    """
    cdef int n_steps = log_returns.shape[0]
    cdef SVCJParams p
    cdef FilterState* states = <FilterState*> malloc(n_steps * sizeof(FilterState))
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n_steps)
    
    optimize_params_history(&log_returns[0], n_steps, &p)
    run_ukf_filter(&log_returns[0], n_steps, &p, states)
    
    for i in range(n_steps):
        residues[i] = states[i].drift_residue
        
    free(states)
    return residues