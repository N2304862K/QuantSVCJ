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
                             double* strikes, double* prices, double* T_exp, int* types, int n_opts,
                             double S0, double r, int mode) nogil

# --- 1. Snapshot (Joint) ---
def c_joint_fit(double[:] returns, double[:] strikes, double[:] prices, double[:] T, int[:] types, double S0, double r):
    cdef int nr = returns.shape[0]
    cdef int no = strikes.shape[0]
    cdef SVCJResult res
    
    with nogil:
        res = optimize_svcj(&returns[0], nr, 1.0/252.0, 
                            &strikes[0], &prices[0], &T[0], &types[0], no, 
                            S0, r, 1) # Mode 1
                            
    return {
        "kappa": res.p.kappa, "theta": res.p.theta, "sigma_v": res.p.sigma_v, 
        "rho": res.p.rho, "lambda": res.p.lambda, "mu_j": res.p.mu_j, 
        "sigma_j": res.p.sigma_j, "spot_vol": res.spot_vol, "jump_prob": res.jump_prob
    }

# --- 2. Single Rolling ---
def c_rolling_fit(double[:] returns, int window):
    cdef int T = returns.shape[0]
    cdef int n_out = T - window + 1
    if n_out < 1: return np.zeros((0, 9))
    cdef double[:, ::1] out = np.zeros((n_out, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    
    # Parallel Rolling
    for i in prange(n_out, nogil=True):
        res = optimize_svcj(&returns[i], window, 1.0/252.0, 
                            NULL, NULL, NULL, NULL, 0, 
                            0, 0, 0)
        out[i, 0] = res.p.kappa; out[i, 1] = res.p.theta; out[i, 2] = res.p.sigma_v;
        out[i, 3] = res.p.rho;   out[i, 4] = res.p.lambda; out[i, 5] = res.p.mu_j;
        out[i, 6] = res.p.sigma_j; out[i, 7] = res.spot_vol; out[i, 8] = res.jump_prob;
        
    return np.asarray(out)

# --- 3. Market Screen (Static) ---
def c_market_screen(double[:, ::1] returns_matrix):
    cdef int n_assets = returns_matrix.shape[0]
    cdef int n_obs = returns_matrix.shape[1]
    cdef double[:, ::1] out = np.zeros((n_assets, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    
    for i in prange(n_assets, nogil=True):
        res = optimize_svcj(&returns_matrix[i, 0], n_obs, 1.0/252.0,
                            NULL, NULL, NULL, NULL, 0,
                            0, 0, 0)
        out[i, 0] = res.p.kappa; out[i, 1] = res.p.theta; out[i, 2] = res.p.sigma_v;
        out[i, 3] = res.p.rho;   out[i, 4] = res.p.lambda; out[i, 5] = res.p.mu_j;
        out[i, 6] = res.p.sigma_j; out[i, 7] = res.spot_vol; out[i, 8] = res.jump_prob;
        
    return np.asarray(out)

# --- 4. Multi-Asset Rolling ---
# Returns flat matrix: [Time, Asset*9]
def c_multi_rolling(double[:, ::1] returns_matrix, int window):
    cdef int n_assets = returns_matrix.shape[0]
    cdef int T_total = returns_matrix.shape[1]
    cdef int n_out = T_total - window + 1
    if n_out < 1: return np.zeros((0, 0))
    
    # Output: [Windows, Assets * 9]
    cdef double[:, ::1] out = np.zeros((n_out, n_assets * 9), dtype=np.float64)
    
    cdef int a, w, base_idx
    cdef SVCJResult res
    cdef double dt = 1.0/252.0
    
    # Parallelize by Asset (Outer Loop)
    for a in prange(n_assets, nogil=True):
        base_idx = a * 9
        # Sequential Rolling for each asset
        for w in range(n_out):
            res = optimize_svcj(&returns_matrix[a, w], window, dt,
                                NULL, NULL, NULL, NULL, 0,
                                0, 0, 0)
            
            out[w, base_idx + 0] = res.p.kappa
            out[w, base_idx + 1] = res.p.theta
            out[w, base_idx + 2] = res.p.sigma_v
            out[w, base_idx + 3] = res.p.rho
            out[w, base_idx + 4] = res.p.lambda
            out[w, base_idx + 5] = res.p.mu_j
            out[w, base_idx + 6] = res.p.sigma_j
            out[w, base_idx + 7] = res.spot_vol
            out[w, base_idx + 8] = res.jump_prob

    return np.asarray(out)