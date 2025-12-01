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
        double T_years
        int is_call
        int valid

    void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns) nogil
    void optimize_params_history(double* returns, int n_steps, SVCJParams* out) nogil
    void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out) nogil
    void calibrate_to_options(OptionContract* opts, int n, double spot, SVCJParams* out) nogil

# --- Python API ---

def analyze_asset_raw(np.ndarray[double, ndim=1] raw_prices, 
                      np.ndarray[double, ndim=2] raw_option_chain):
    cdef int n_prices = raw_prices.shape[0]
    cdef int n_steps = n_prices - 1
    cdef int n_opts = raw_option_chain.shape[0]
    cdef double spot_price = raw_prices[n_prices-1]

    cdef double* c_returns = <double*> malloc(n_steps * sizeof(double))
    cdef OptionContract* c_opts = <OptionContract*> malloc(n_opts * sizeof(OptionContract))
    cdef FilterState* states = <FilterState*> malloc(n_steps * sizeof(FilterState))
    cdef SVCJParams p
    memset(&p, 0, sizeof(SVCJParams))

    for i in range(n_opts):
        c_opts[i].strike = raw_option_chain[i, 0]
        c_opts[i].price = raw_option_chain[i, 1]
        c_opts[i].T_years = raw_option_chain[i, 2]
        c_opts[i].is_call = <int>raw_option_chain[i, 3]

    with nogil:
        # Pipeline: Raw Prices -> Returns -> Optimize -> Option Calib -> Filter
        calculate_returns_from_prices(&raw_prices[0], n_prices, c_returns)
        optimize_params_history(c_returns, n_steps, &p)
        calibrate_to_options(c_opts, n_opts, spot_price, &p)
        run_ukf_filter(c_returns, n_steps, &p, states)

    spot_vols = np.zeros(n_steps)
    jump_probs = np.zeros(n_steps)
    drift_residues = np.zeros(n_steps)
    
    for i in range(n_steps):
        spot_vols[i] = states[i].spot_vol
        jump_probs[i] = states[i].jump_prob
        drift_residues[i] = states[i].drift_residue

    free(c_returns)
    free(c_opts)
    free(states)

    # Dictionary keys match the struct and user expectation
    return {
        "params": {"kappa": p.kappa, "theta": p.theta, "rho": p.rho},
        "daily_volatility": spot_vols,
        "jump_probability": jump_probs,
        "drift_residue": drift_residues 
    }

def screen_market_raw(np.ndarray[double, ndim=2] market_prices):
    cdef int n_assets = market_prices.shape[0]
    cdef int n_prices = market_prices.shape[1]
    cdef int n_steps = n_prices - 1
    cdef int i
    
    cdef np.ndarray[double, ndim=2] output = np.zeros((n_assets, 2))
    
    cdef double* c_returns
    cdef FilterState* states
    cdef SVCJParams p

    with nogil:
        for i in prange(n_assets):
            c_returns = <double*> malloc(n_steps * sizeof(double))
            states = <FilterState*> malloc(n_steps * sizeof(FilterState))
            memset(&p, 0, sizeof(SVCJParams))
            
            calculate_returns_from_prices(&market_prices[i, 0], n_prices, c_returns)
            optimize_params_history(c_returns, n_steps, &p)
            run_ukf_filter(c_returns, n_steps, &p, states)
            
            output[i, 0] = states[n_steps-1].spot_vol
            output[i, 1] = states[n_steps-1].jump_prob
            
            free(c_returns)
            free(states)
            
    return output