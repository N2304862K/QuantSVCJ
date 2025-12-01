# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

cdef extern from "svcj.h":
    ctypedef struct SVCJParams:
        double mu, kappa, theta, sigma_v, rho, lambda_j, mu_j, sigma_j
    
    void clean_and_copy(double* src, double* dest, int n) nogil
    void check_stability(SVCJParams* params) nogil
    double run_ukf_qmle(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices) nogil

# Helper to ensure contiguous C-double array
cdef double[:] ensure_contiguous(object data):
    return np.ascontiguousarray(data, dtype=np.double)

# --- 1. Asset-Specific Option Adjusted ---
def generate_asset_option_adjusted(double[:] returns, double s0, double[:, :] option_chain):
    # Ensure input is safe
    cdef double[:] safe_returns = np.ascontiguousarray(returns, dtype=np.double)
    cdef int n = safe_returns.shape[0]
    
    cdef SVCJParams params
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    
    # Defaults (Annualized)
    params.kappa = 2.0
    params.theta = 0.04   # 20% Vol
    params.sigma_v = 0.3
    params.rho = -0.7
    params.lambda_j = 0.5
    params.mu_j = -0.05
    params.sigma_j = 0.05
    params.mu = 0.08

    # We use a temp buffer for cleaning to protect original data
    cdef double[:] clean_buffer = np.zeros(n, dtype=np.double)
    clean_and_copy(&safe_returns[0], &clean_buffer[0], n)
    
    check_stability(&params)
    run_ukf_qmle(&clean_buffer[0], n, &params, &spot_vol[0], &jump_prob[0])
    
    return {
        "kappa": params.kappa, "theta": params.theta, "rho": params.rho,
        "spot_vol": spot_vol, "jump_prob": jump_prob
    }

# --- 2. Market Wide Current (FIXED & PROTECTED) ---
def analyze_market_current(double[:, :] market_returns):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    
    # 1. Create Outputs
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_days, n_assets))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_days, n_assets))
    
    # 2. Python-side loop to prepare buffers (Cannot allocate python objs in nogil)
    # Actually, we can just process contiguous columns if we transpose, 
    # but to satisfy "Matrix In", we iterate.
    
    # IMPORTANT: To avoid the stride issue, we Copy the column to a temp buffer inside the thread
    # But malloc in Cython nogil is tedious.
    # BETTER STRATEGY: Transpose input to (Assets, Time) so each row is contiguous.
    # This is "Copy the thing" logic.
    cdef double[:, :] market_T = np.ascontiguousarray(market_returns.T, dtype=np.double)
    
    cdef int i
    cdef SVCJParams p
    
    # Parallel Loop over Assets (Rows of Transposed Matrix)
    with nogil:
        for i in prange(n_assets):
            p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7; 
            p.lambda_j = 0.5; p.mu_j = -0.05; p.sigma_j = 0.05; p.mu = 0.08;
            
            check_stability(&p)
            
            # Since market_T is (Assets, Time), market_T[i, :] is a contiguous block of 'n_days'
            # We pass address of market_T[i, 0]
            # No cleaning buffer needed if we accept in-place cleaning on the copy
            # But let's be safe and clean in-place on the Transposed copy (which is discarded later)
            
            clean_and_copy(&market_T[i, 0], &market_T[i, 0], n_days)
            
            # Run UKF
            # We write to Transposed Output buffers too? No, we can write to out_spot[days, i]
            # Writing strided output is fine if inputs are contiguous.
            # Actually, simpler to return Transposed output and transpose back in Python.
            # But 'out_spot' is passed by pointer. Let's make temp output buffers.
            # Given complexity, we just allocate a small stack buffer if n_days small? No.
            # We will write directly to out_spot but we need to handle the index manually or use specific pointer.
            # Correction: We can't easily write columns to a row-major matrix in C without strides.
            # BUT: We solved the input stride by Transposing.
            # Let's create Transposed Outputs.
    
    # Re-declare outputs as Transposed for parallel efficiency
    cdef np.ndarray[double, ndim=2] out_spot_T = np.zeros((n_assets, n_days))
    cdef np.ndarray[double, ndim=2] out_jump_T = np.zeros((n_assets, n_days))

    with nogil:
        for i in prange(n_assets):
            p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7; 
            p.lambda_j = 0.5; p.mu_j = -0.05; p.sigma_j = 0.05; p.mu = 0.08;
            
            check_stability(&p)
            clean_and_copy(&market_T[i, 0], &market_T[i, 0], n_days)
            
            run_ukf_qmle(&market_T[i, 0], n_days, &p, &out_spot_T[i, 0], &out_jump_T[i, 0])

    # Return Transposed back to original shape (Time, Assets)
    return {"spot_vol": out_spot_T.T, "jump_prob": out_jump_T.T}

# --- 3. Rolling Analysis ---
def analyze_market_rolling(double[:, :] market_returns, int window):
    cdef int n_days = market_returns.shape[0]
    cdef int n_assets = market_returns.shape[1]
    cdef int n_windows = n_days - window
    
    # Transpose for contiguous access
    cdef double[:, :] market_T = np.ascontiguousarray(market_returns.T, dtype=np.double)
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 3)) 
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            # Pre-clean the whole asset history once
            clean_and_copy(&market_T[i, 0], &market_T[i, 0], n_days)
            
            for w in range(n_windows):
                p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; 
                p.rho = -0.5; p.lambda_j = 0.1; p.mu_j = 0.0; p.sigma_j = 0.05; p.mu = 0.0;
                
                check_stability(&p)
                # Input: Start at window w
                run_ukf_qmle(&market_T[i, w], window, &p, NULL, NULL)
                
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.rho
    
    return results

# --- 4. Residue Analysis ---
def generate_residue_analysis(double[:] returns, int forward_window):
    cdef double[:] safe_ret = np.ascontiguousarray(returns, dtype=np.double)
    cdef int n = safe_ret.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef SVCJParams p
    
    p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7;
    p.lambda_j = 0.1; p.mu_j = -0.05; p.sigma_j = 0.05; p.mu = 0.08;
    
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    
    run_ukf_qmle(&safe_ret[0], n, &p, &spot_vol[0], NULL)
    
    cdef int t
    cdef double realized, expected
    # DT is already handled in params, but here we project returns
    # Expected Return (Daily) = (mu - 0.5*v) * dt
    cdef double dt = 1.0/252.0
    
    for t in range(n - forward_window):
        realized = safe_ret[t + forward_window] 
        # Simple projection: Vol is Annualized, so convert to daily var
        expected = (p.mu - 0.5 * spot_vol[t]*spot_vol[t]) * dt * forward_window;
        residues[t] = realized - expected;
        
    return residues