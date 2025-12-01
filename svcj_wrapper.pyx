# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "svcj_kernel.h":
    ctypedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
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

    void optimize_params_history(double* returns, int n_steps, SVCJParams* out) nogil
    void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out) nogil
    void calibrate_to_options(OptionContract* opts, int n, double spot, SVCJParams* out) nogil

# --- Python API ---

def fit_asset_options(np.ndarray[double, ndim=1] log_returns, 
                      np.ndarray[double, ndim=2] option_chain, 
                      double spot_price):
    cdef int n_steps = log_returns.shape[0]
    cdef int n_opts = option_chain.shape[0]
    cdef SVCJParams p
    memset(&p, 0, sizeof(SVCJParams))
    
    cdef OptionContract* c_opts = <OptionContract*> malloc(n_opts * sizeof(OptionContract))
    cdef FilterState* states = <FilterState*> malloc(n_steps * sizeof(FilterState))
    
    for i in range(n_opts):
        c_opts[i].strike = option_chain[i, 0]
        c_opts[i].price = option_chain[i, 1]
        c_opts[i].T = option_chain[i, 2]
        c_opts[i].is_call = <int>option_chain[i, 3]

    # 1. Fit History (Daily Terms)
    optimize_params_history(&log_returns[0], n_steps, &p)
    
    # 2. Adjust using Options (Converts Annual Implied to Daily Internal)
    calibrate_to_options(c_opts, n_opts, spot_price, &p)
    
    # 3. Filter
    run_ukf_filter(&log_returns[0], n_steps, &p, states)
    
    spot_vols = np.zeros(n_steps)
    jump_probs = np.zeros(n_steps)
    
    for i in range(n_steps):
        spot_vols[i] = states[i].spot_vol
        jump_probs[i] = states[i].jump_prob

    free(c_opts)
    free(states)

    return {
        "params": {"kappa": p.kappa, "theta_daily": p.theta, "rho": p.rho},
        "spot_vol_daily": spot_vols,
        "jump_prob": jump_probs
    }

def generate_current_screen(np.ndarray[double, ndim=2] market_returns):
    cdef int n_assets = market_returns.shape[0]
    cdef int n_time = market_returns.shape[1]
    cdef int i
    cdef np.ndarray[double, ndim=2] output = np.zeros((n_assets, 2))
    cdef SVCJParams p
    cdef FilterState* states
    
    with nogil:
        for i in prange(n_assets):
            memset(&p, 0, sizeof(SVCJParams))
            states = <FilterState*> malloc(n_time * sizeof(FilterState))
            optimize_params_history(&market_returns[i, 0], n_time, &p)
            run_ukf_filter(&market_returns[i, 0], n_time, &p, states)
            output[i, 0] = states[n_time-1].spot_vol
            output[i, 1] = states[n_time-1].jump_prob
            free(states)
            
    return output

def calculate_drift_residue(np.ndarray[double, ndim=1] log_returns):
    cdef int n_steps = log_returns.shape[0]
    cdef SVCJParams p
    memset(&p, 0, sizeof(SVCJParams))
    cdef FilterState* states = <FilterState*> malloc(n_steps * sizeof(FilterState))
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n_steps)
    optimize_params_history(&log_returns[0], n_steps, &p)
    run_ukf_filter(&log_returns[0], n_steps, &p, states)
    for i in range(n_steps):
        residues[i] = states[i].drift_residue
    free(states)
    return residues