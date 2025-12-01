# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_returns(double* returns, int n, int stride) nogil
    void check_feller_and_fix(SVCJParams* params) nogil
    double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices) nogil

# --- 1. Asset-Specific Option Adjusted ---
def generate_asset_option_adjusted(double[:] returns, double s0, double[:, :] option_chain):
    cdef int n = returns.shape[0]
    cdef int n_opts = option_chain.shape[0]
    cdef SVCJParams params
    
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    
    # Daily scale defaults
    params.kappa = 0.008; params.theta = 0.00016; params.sigma_v = 0.01; params.rho = -0.7;
    params.lambda_j = 0.002; params.mu_j = -0.01; params.sigma_j = 0.02; params.mu = 0.0004;
    
    # Stride 1 for 1D
    clean_returns(&returns[0], n, 1)
    check_feller_and_fix(&params)
    run_ukf_qmle(&returns[0], n, 1, &params, &spot_vol[0], &jump_prob[0])
    
    return {
        "kappa": params.kappa, "theta": params.theta, "rho": params.rho,
        "spot_vol": spot_vol, "jump_prob": jump_prob
    }

# --- 2. Market Wide Historic Rolling ---
def analyze_market_rolling(double[:, :] market_returns, int window):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    cdef int n_windows = n_days - window
    cdef int stride = n_assets 
    
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 3)) 
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                p.kappa = 0.01; p.theta = 0.00015; p.sigma_v = 0.01; 
                p.rho = -0.5; p.lambda_j = 0.001; p.mu_j = 0.0; p.sigma_j = 0.01; p.mu = 0.0;
                
                check_feller_and_fix(&p)
                # Pass NULL for outputs as we only want internal param state if we were optimizing
                run_ukf_qmle(&market_returns[w, i], window, stride, &p, NULL, NULL)
                
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.rho
    
    return results

# --- 3. Market Wide Current (FIXED) ---
def analyze_market_current(double[:, :] market_returns):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    cdef int stride = n_assets 
    
    # These matrices are (n_days, n_assets)
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_days, n_assets))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_days, n_assets))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            p.kappa = 0.01; p.theta = 0.00016; p.sigma_v = 0.01; p.rho = -0.7; 
            p.lambda_j = 0.002; p.mu_j = -0.05; p.sigma_j = 0.05; p.mu = 0.0002;
            
            clean_returns(&market_returns[0, i], n_days, stride)
            check_feller_and_fix(&p)
            
            # Pass pointers to the specific column (asset) in output matrices
            # stride handles the vertical writing down the time series
            run_ukf_qmle(&market_returns[0, i], n_days, stride, &p, &out_spot[0, i], &out_jump[0, i])
            
    return {"spot_vol": out_spot, "jump_prob": out_jump}

# --- 4. Asset-Specific Drift-Realized Residue ---
def generate_residue_analysis(double[:] returns, int forward_window):
    cdef int n = returns.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef SVCJParams p
    
    p.kappa = 0.01; p.theta = 0.00016; p.sigma_v = 0.01; p.rho = -0.7;
    p.lambda_j = 0.002; p.mu_j = -0.05; p.sigma_j = 0.05; p.mu = 0.0005;
    
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    
    run_ukf_qmle(&returns[0], n, 1, &p, &spot_vol[0], NULL)
    
    cdef int t
    cdef double realized, expected
    for t in range(n - forward_window):
        realized = returns[t + forward_window] 
        expected = (p.mu - 0.5 * spot_vol[t]*spot_vol[t]) * forward_window;
        residues[t] = realized - expected;
        
    return residues