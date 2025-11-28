# distutils: sources = src/svcj_engine.c
# distutils: include_dirs = src/
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj_engine.h":
    typedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double lambda
        double mu_j
        double sigma_j

    SVCJParams optimize_core(double* returns, int n_ret, double dt,
                             double* strikes, double* prices, double* T_exp, int n_opts,
                             double S0, double r, int mode) nogil

# --- 1. Single Asset Joint Calibration ---
def fit_joint(double[:] returns, double[:] strikes, double[:] prices, double[:] T, double S0, double r):
    cdef SVCJParams p
    cdef int nr = returns.shape[0]
    cdef int no = strikes.shape[0]
    
    with nogil:
        p = optimize_core(&returns[0], nr, 1.0/252.0, 
                          &strikes[0], &prices[0], &T[0], no, 
                          S0, r, 1) # Mode 1: Joint

    return np.array([p.kappa, p.theta, p.sigma_v, p.rho, p.lambda, p.mu_j, p.sigma_j])

# --- 2. Rolling History (Single Asset) ---
def fit_rolling(double[:] returns, int window):
    cdef int T = returns.shape[0]
    cdef int n_windows = T - window + 1
    cdef double[:, :] res = np.zeros((n_windows, 7), dtype=np.float64)
    
    cdef int i
    cdef SVCJParams p
    cdef double dt = 1.0/252.0

    # No Python Loop here. pure C loop over windows.
    # Can be parallelized if needed, but sequential is usually fine for one asset
    for i in range(n_windows):
        # We hold the GIL briefly here unless we fully release, 
        # but optimize_core is fast. For strict speed:
        with nogil:
            p = optimize_core(&returns[i], window, dt, 
                              NULL, NULL, NULL, 0, 
                              0, 0, 0) # Mode 0: History Only
            
            res[i, 0] = p.kappa
            res[i, 1] = p.theta
            res[i, 2] = p.sigma_v
            res[i, 3] = p.rho
            res[i, 4] = p.lambda
            res[i, 5] = p.mu_j
            res[i, 6] = p.sigma_j
            
    return np.asarray(res)

# --- 3. Multi-Asset Screen (Parallel) ---
def fit_screen(double[:, :] returns_matrix):
    cdef int n_assets = returns_matrix.shape[0]
    cdef int n_obs = returns_matrix.shape[1]
    cdef double[:, :] res = np.zeros((n_assets, 7), dtype=np.float64)
    cdef SVCJParams p
    cdef int i
    
    # OpenMP Parallel Loop
    for i in prange(n_assets, nogil=True):
        p = optimize_core(&returns_matrix[i, 0], n_obs, 1.0/252.0,
                          NULL, NULL, NULL, 0,
                          0, 0, 0)
        
        res[i, 0] = p.kappa
        res[i, 1] = p.theta
        res[i, 2] = p.sigma_v
        res[i, 3] = p.rho
        res[i, 4] = p.lambda
        res[i, 5] = p.mu_j
        res[i, 6] = p.sigma_j
        
    return np.asarray(res)