# distutils: sources = svcj.c
# distutils: include_dirs = .
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    typedef struct SVCJParams:
        double kappa
        double theta
        double sigma_v
        double rho
        double lambda
        double mu_j
        double sigma_j

    typedef struct SVCJResult:
        SVCJParams p
        double spot_vol
        double jump_prob
        double error

    SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                             double* strikes, double* prices, double* T_exp, int n_opts,
                             double S0, double r, int mode) nogil

# --- Helper to pack result into array ---
cdef void pack_result(double[:] out, int idx, SVCJResult res):
    out[0] = res.p.kappa
    out[1] = res.p.theta
    out[2] = res.p.sigma_v
    out[3] = res.p.rho
    out[4] = res.p.lambda
    out[5] = res.p.mu_j
    out[6] = res.p.sigma_j
    out[7] = res.spot_vol
    out[8] = res.jump_prob

# --- 1. Snapshot (Joint) ---
def c_analyze_snapshot(double[:] returns, double[:] strikes, double[:] prices, double[:] T, double S0, double r):
    cdef SVCJResult res
    cdef int n_ret = returns.shape[0]
    cdef int n_opt = strikes.shape[0]
    
    with nogil:
        res = optimize_svcj(&returns[0], n_ret, 1.0/252.0,
                            &strikes[0], &prices[0], &T[0], n_opt,
                            S0, r, 1) # Mode 1: Joint
                            
    return {
        "kappa": res.p.kappa, "theta": res.p.theta, "sigma_v": res.p.sigma_v,
        "rho": res.p.rho, "lambda": res.p.lambda, "mu_j": res.p.mu_j, "sigma_j": res.p.sigma_j,
        "spot_vol": res.spot_vol, "jump_prob": res.jump_prob
    }

# --- 2. Rolling History ---
def c_analyze_rolling(double[:] returns, int window):
    cdef int total_len = returns.shape[0]
    cdef int n_out = total_len - window + 1
    if n_out <= 0: return np.zeros((0, 9))
    
    cdef double[:, ::1] out = np.zeros((n_out, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    cdef double dt = 1.0/252.0
    
    # Parallelize Rolling Window (Heavy Compute)
    for i in prange(n_out, nogil=True):
        res = optimize_svcj(&returns[i], window, dt,
                            NULL, NULL, NULL, 0,
                            0, 0, 0) # Mode 0: History
        
        out[i, 0] = res.p.kappa
        out[i, 1] = res.p.theta
        out[i, 2] = res.p.sigma_v
        out[i, 3] = res.p.rho
        out[i, 4] = res.p.lambda
        out[i, 5] = res.p.mu_j
        out[i, 6] = res.p.sigma_j
        out[i, 7] = res.spot_vol
        out[i, 8] = res.jump_prob
        
    return np.asarray(out)

# --- 3. Market Screen ---
def c_analyze_screen(double[:, ::1] returns_matrix):
    cdef int n_assets = returns_matrix.shape[0]
    cdef int n_obs = returns_matrix.shape[1]
    
    cdef double[:, ::1] out = np.zeros((n_assets, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    
    for i in prange(n_assets, nogil=True):
        res = optimize_svcj(&returns_matrix[i, 0], n_obs, 1.0/252.0,
                            NULL, NULL, NULL, 0,
                            0, 0, 0) # Mode 0
        
        out[i, 0] = res.p.kappa
        out[i, 1] = res.p.theta
        out[i, 2] = res.p.sigma_v
        out[i, 3] = res.p.rho
        out[i, 4] = res.p.lambda
        out[i, 5] = res.p.mu_j
        out[i, 6] = res.p.sigma_j
        out[i, 7] = res.spot_vol
        out[i, 8] = res.jump_prob
        
    return np.asarray(out)