# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cdef extern from "svcj_kernel.h":
    ctypedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double v0

    ctypedef struct FilterResult:
        double spot_vol
        double jump_prob
        double drift_residue

    ctypedef struct RawOption:
        double strike
        double price
        double T_days
        int is_call

    void full_svcj_pipeline(double* raw_prices, int n_price_steps, RawOption* opts, int n_opts, SVCJParams* p, FilterResult* res) nogil
    void batch_process_matrix(double* mat, int assets, int time, double* v, double* j) nogil

# --- PUBLIC INTERFACE ---

def analyze_asset_raw(np.ndarray[double, ndim=1] prices_array, 
                      np.ndarray[double, ndim=2] option_array):
    """
    INPUTS:
      prices_array: 1D array of raw close prices (Time Series)
      option_array: 2D array [Strike, Price, DaysToExpiry, IsCall(1/0)]
    """
    cdef int n_prices = prices_array.shape[0]
    cdef int n_opts = option_array.shape[0]
    
    # Allocations
    cdef SVCJParams params
    cdef FilterResult* results = <FilterResult*> malloc((n_prices - 1) * sizeof(FilterResult))
    cdef RawOption* c_opts = <RawOption*> malloc(n_opts * sizeof(RawOption))
    
    # Marshal Options
    for i in range(n_opts):
        c_opts[i].strike = option_array[i, 0]
        c_opts[i].price = option_array[i, 1]
        c_opts[i].T_days = option_array[i, 2]
        c_opts[i].is_call = <int>option_array[i, 3]

    # CALL KERNEL (GIL release for speed if needed, though mostly serial here)
    full_svcj_pipeline(&prices_array[0], n_prices, c_opts, n_opts, &params, results)
    
    # Unpack Results
    # Since returns are N-1, output arrays are N-1
    spot_vols = np.zeros(n_prices - 1)
    jump_probs = np.zeros(n_prices - 1)
    residues = np.zeros(n_prices - 1)

    for i in range(n_prices - 1):
        spot_vols[i] = results[i].spot_vol
        jump_probs[i] = results[i].jump_prob
        residues[i] = results[i].drift_residue
        
    free(c_opts)
    free(results)
    
    return {
        "params": {"kappa": params.kappa, "theta_daily": params.theta, "rho": params.rho},
        "spot_vol": spot_vols,
        "jump_prob": jump_probs,
        "residues": residues
    }

def screen_market_raw(np.ndarray[double, ndim=2] price_matrix):
    """
    INPUT:
      price_matrix: Raw Prices [Assets x Time]
    OUTPUT:
      [Assets x 2] matrix of (Last Spot Vol, Last Jump Prob)
    """
    cdef int n_assets = price_matrix.shape[0]
    cdef int n_time = price_matrix.shape[1]
    
    cdef np.ndarray[double, ndim=1] out_vols = np.zeros(n_assets)
    cdef np.ndarray[double, ndim=1] out_jumps = np.zeros(n_assets)
    
    # OpenMP Parallel Batch Processing
    with nogil:
        batch_process_matrix(&price_matrix[0,0], n_assets, n_time, &out_vols[0], &out_jumps[0])
        
    return np.column_stack((out_vols, out_jumps))