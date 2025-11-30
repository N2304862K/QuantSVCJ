# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free

# --- C Declarations ---
cdef extern from "svcj.h":
    ctypedef struct ModelParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double lambda_j
        double mu_j
        double sigma_j
        double v0

    ctypedef struct FilterOutput:
        double spot_vol
        double jump_prob
        double log_likelihood

    void run_ukf_qmle(double* log_returns, int T, ModelParams* params, FilterOutput* out_states) nogil
    double carr_madan_price(double S0, double K, double T, double r, double q, ModelParams* p, int is_call) nogil

# --- Python Wrapper Class ---

def analyze_market_screen(np.ndarray[double, ndim=2] market_returns):
    """
    High-Performance Scaling: Analyzes multiple assets in parallel using OpenMP.
    market_returns: shape (n_assets, time_steps)
    """
    cdef int n_assets = market_returns.shape[0]
    cdef int T = market_returns.shape[1]
    
    # Output arrays
    cdef np.ndarray[double, ndim=2] spot_vols = np.zeros((n_assets, T), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] jump_probs = np.zeros((n_assets, T), dtype=np.float64)
    
    # Temporary pointers for OpenMP
    cdef double[:, :] ret_view = market_returns
    cdef double[:, :] vol_view = spot_vols
    cdef double[:, :] jump_view = jump_probs
    
    cdef int i, t
    cdef ModelParams p
    cdef FilterOutput* out_buffer
    
    # Generic Default Parameters for screening
    p.kappa = 2.0
    p.theta = 0.04
    p.sigma_v = 0.3
    p.rho = -0.7
    p.lambda_j = 0.5
    p.mu_j = -0.05
    p.sigma_j = 0.1
    p.v0 = 0.04

    # Parallel Loop (Releases GIL)
    with nogil:
        for i in prange(n_assets, schedule='static'):
            # Allocate thread-local buffer
            out_buffer = <FilterOutput*> malloc(T * sizeof(FilterOutput))
            
            # Copy row to contiguous buffer if needed, or pass pointer directly
            # Here we assume C-contiguous input, passing address of first element
            run_ukf_qmle(&ret_view[i, 0], T, &p, out_buffer)
            
            # Unpack results back to views
            for t in range(T):
                vol_view[i, t] = out_buffer[t].spot_vol
                jump_view[i, t] = out_buffer[t].jump_prob
            
            free(out_buffer)

    return spot_vols, jump_probs

def price_option_chain(double S0, double r, double q, double T, 
                       np.ndarray[double, ndim=1] strikes, 
                       np.ndarray[int, ndim=1] types):
    """
    Comprehensive Option Support: Prices mixed Calls (1) and Puts (0) 
    using Bates/SVCJ logic.
    """
    cdef int n = strikes.shape[0]
    cdef np.ndarray[double, ndim=1] prices = np.zeros(n, dtype=np.float64)
    cdef ModelParams p
    
    # Load calibrated params (Hardcoded for demo, usually passed in)
    p.kappa = 1.5; p.theta = 0.04; p.sigma_v = 0.4; p.rho = -0.6;
    p.lambda_j = 0.3; p.mu_j = -0.09; p.sigma_j = 0.15; p.v0 = 0.04;

    cdef int i
    for i in range(n):
        prices[i] = carr_madan_price(S0, strikes[i], T, r, q, &p, types[i])
        
    return prices