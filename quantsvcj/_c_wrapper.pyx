# distutils: sources = src/svcj_core.c
# distutils: include_dirs = src/
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj_core.h":
    typedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double lambda
        double mu_j
        double sigma_j

    SVCJParams optimize_svcj(double* returns, int n_ret, double dt, 
                             double* strikes, double* prices, double* T, int n_opts, 
                             double S0, double r, int mode) nogil

def fit_multi_asset_history(double[:, ::1] returns_matrix, double dt=1.0/252.0):
    """
    Input: (N_assets, T_time) numpy array of log returns.
    Output: (N_assets, 7) array of SVCJ parameters.
    """
    cdef int n_assets = returns_matrix.shape[0]
    cdef int n_obs = returns_matrix.shape[1]
    
    # Output array
    cdef double[:, ::1] results = np.zeros((n_assets, 7), dtype=np.float64)
    
    cdef int i
    cdef SVCJParams p
    
    # OpenMP Parallel Loop -> No Python GIL
    # This is where the speed comes from.
    for i in prange(n_assets, nogil=True):
        # Pass pointer to the specific row
        p = optimize_svcj(&returns_matrix[i, 0], n_obs, dt, 
                          NULL, NULL, NULL, 0, 0, 0, 1) # Mode 1: History
        
        results[i, 0] = p.kappa
        results[i, 1] = p.theta
        results[i, 2] = p.sigma_v
        results[i, 3] = p.rho
        results[i, 4] = p.lambda
        results[i, 5] = p.mu_j
        results[i, 6] = p.sigma_j
        
    return np.asarray(results)

def fit_single_asset_joint(double[::1] returns, 
                           double[::1] strikes, 
                           double[::1] prices, 
                           double[::1] T, 
                           double S0, double r, double dt=1.0/252.0):
    """
    Fits single asset using both history and option chain.
    """
    cdef int n_ret = returns.shape[0]
    cdef int n_opts = strikes.shape[0]
    
    cdef SVCJParams p
    
    with nogil:
        p = optimize_svcj(&returns[0], n_ret, dt,
                          &strikes[0], &prices[0], &T[0], n_opts,
                          S0, r, 3) # Mode 3: Joint
                          
    return {
        "kappa": p.kappa, "theta": p.theta, "sigma_v": p.sigma_v,
        "rho": p.rho, "lambda": p.lambda, "mu_j": p.mu_j, "sigma_j": p.sigma_j
    }