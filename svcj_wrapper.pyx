# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    void clean_returns(double* returns, int n) nogil
    void check_constraints(SVCJParams* p) nogil
    double run_filter(double* returns, int n, SVCJParams* p, double* out_spot, double* out_jump) nogil
    void fit_history(double* returns, int n, SVCJParams* p) nogil
    void calibrate_options(double s0, double* strikes, double* expiries, int* types, double* prices, int n_opts, SVCJParams* p) nogil

cdef SVCJParams defaults() nogil:
    cdef SVCJParams p
    p.mu=0.0; p.kappa=2.0; p.theta=0.04; p.sigma_v=0.3; p.rho=-0.7;
    p.lambda_j=0.1; p.mu_j=-0.05; p.sigma_j=0.1;
    return p

# --- 1. Asset Specific Option Adjusted ---
def generate_asset_option_adjusted(double[:] returns_in, double s0, double[:, :] opts):
    cdef double[::1] ret = np.ascontiguousarray(returns_in) # Force Contiguous
    cdef double[:, ::1] opt_c = np.ascontiguousarray(opts)
    
    cdef int n = ret.shape[0]
    cdef int n_opts = opt_c.shape[0]
    cdef SVCJParams p = defaults()
    
    cdef np.ndarray[double] spot = np.zeros(n)
    cdef np.ndarray[double] jump = np.zeros(n)
    
    # Extract option columns
    cdef np.ndarray[double] K = np.ascontiguousarray(opt_c[:, 0])
    cdef np.ndarray[double] T = np.ascontiguousarray(opt_c[:, 1])
    cdef np.ndarray[int] Typ = np.ascontiguousarray(opt_c[:, 2]).astype(np.int32)
    cdef np.ndarray[double] P = np.ascontiguousarray(opt_c[:, 3])
    
    with nogil:
        clean_returns(&ret[0], n)
        fit_history(&ret[0], n, &p)
        calibrate_options(s0, &K[0], &T[0], <int*>&Typ[0], &P[0], n_opts, &p)
        run_filter(&ret[0], n, &p, &spot[0], &jump[0])
        
    return {"kappa": p.kappa, "theta": p.theta, "rho": p.rho, "spot_vol": spot, "jump_prob": jump}

# --- 2. Market Current (Optimized Memory) ---
def analyze_market_current(double[:, :] market_in):
    # TRANSPOSE AND FORCE CONTIGUOUS: Shape (Assets, Days)
    # This ensures mkt[i] is a contiguous block of memory for Asset i
    cdef double[:, ::1] mkt = np.ascontiguousarray(market_in.T)
    
    cdef int n_assets = mkt.shape[0]
    cdef int n_days = mkt.shape[1]
    
    cdef np.ndarray[double, ndim=2] spot = np.zeros((n_assets, n_days))
    cdef np.ndarray[double, ndim=2] jump = np.zeros((n_assets, n_days))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            p = defaults()
            clean_returns(&mkt[i, 0], n_days)
            fit_history(&mkt[i, 0], n_days, &p)
            run_filter(&mkt[i, 0], n_days, &p, &spot[i, 0], &jump[i, 0])
            
    # Transpose output back to (Days, Assets) to match input format
    return {"spot_vol": spot.T, "jump_prob": jump.T}

# --- 3. Market Rolling (Parallelized) ---
def analyze_market_rolling(double[:, :] market_in, int win):
    # Transpose for Contiguity: (Assets, Days)
    cdef double[:, ::1] mkt = np.ascontiguousarray(market_in.T)
    cdef int n_assets = mkt.shape[0]
    cdef int n_days = mkt.shape[1]
    cdef int n_wins = n_days - win
    
    cdef np.ndarray[double, ndim=3] res = np.zeros((n_assets, n_wins, 3))
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            # For each window of this asset
            for w in range(n_wins):
                p = defaults()
                # Pointer arithmetic is safe here because mkt[i] is contiguous
                # &mkt[i, w] points to start of window w for asset i
                fit_history(&mkt[i, w], win, &p)
                
                res[i, w, 0] = p.theta
                res[i, w, 1] = p.rho
                res[i, w, 2] = p.kappa
                
    return res

# --- 4. Residue Analysis ---
def generate_residue_analysis(double[:] returns_in, int fwd):
    cdef double[::1] ret = np.ascontiguousarray(returns_in)
    cdef int n = ret.shape[0]
    cdef SVCJParams p = defaults()
    cdef np.ndarray[double] spot = np.zeros(n)
    cdef np.ndarray[double] res = np.zeros(n - fwd)
    
    with nogil:
        clean_returns(&ret[0], n)
        fit_history(&ret[0], n, &p)
        run_filter(&ret[0], n, &p, &spot[0], NULL)
        
        for t in range(n - fwd):
            double drift = (p.mu - 0.5*spot[t]*spot[t])*fwd*DT;
            res[t] = ret[t+fwd] - drift;
            
    return res