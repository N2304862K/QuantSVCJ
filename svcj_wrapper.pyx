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
    double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob) nogil
    void optimize_svcj(double* returns, int n, int stride, SVCJParams* params) nogil

def generate_asset_option_adjusted(double[:] returns, double s0, double[:, :] option_chain):
    cdef int n = returns.shape[0]
    cdef SVCJParams params
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    cdef np.ndarray[double, ndim=1] jump_prob = np.zeros(n)
    
    # Init Defaults
    params.kappa = 2.0; params.theta = 0.04; params.sigma_v = 0.3; params.rho = -0.7;
    params.lambda_j = 0.5; params.mu_j = -0.05; params.sigma_j = 0.1; params.mu = 0.0;
    
    # 1D Array -> Stride is 1
    clean_returns(&returns[0], n, 1)
    optimize_svcj(&returns[0], n, 1, &params) # Fit to History first
    run_ukf_qmle(&returns[0], n, 1, &params, &spot_vol[0], &jump_prob[0])
    
    return {
        "kappa": params.kappa, "theta": params.theta, "rho": params.rho,
        "spot_vol": spot_vol, "jump_prob": jump_prob
    }

def analyze_market_rolling(double[:, :] market_returns, int window):
    cdef int n_assets = market_returns.shape[1]
    cdef int n_days = market_returns.shape[0]
    cdef int n_windows = n_days - window
    
    # Stride for matrix columns = n_assets (Since it's Row-Major/C-Contiguous)
    cdef int stride = n_assets 
    
    cdef np.ndarray[double, ndim=3] results = np.zeros((n_assets, n_windows, 3)) 
    
    cdef int i, w
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets, schedule='dynamic'):
            for w in range(n_windows):
                p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; 
                p.rho = -0.7; p.lambda_j = 0.1; p.mu_j = 0.0; p.sigma_j = 0.05; p.mu = 0.0;
                
                # Pointer to start of window for asset i
                # Row w, Column i -> &market_returns[w, i]
                clean_returns(&market_returns[w, i], window, stride)
                optimize_svcj(&market_returns[w, i], window, stride, &p)
                
                results[i, w, 0] = p.theta
                results[i, w, 1] = p.rho
    return results

def analyze_market_current(double[:, :] market_returns):
    cdef int n_assets = market_returns.shape[1]
    cdef int n_days = market_returns.shape[0]
    cdef int stride = n_assets # Row-Major Stride
    
    cdef np.ndarray[double, ndim=2] out_spot = np.zeros((n_days, n_assets))
    cdef np.ndarray[double, ndim=2] out_jump = np.zeros((n_days, n_assets))
    
    cdef int i
    cdef SVCJParams p
    
    with nogil:
        for i in prange(n_assets):
            p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7; 
            p.lambda_j = 0.1; p.mu_j = -0.1; p.sigma_j = 0.1; p.mu = 0.0;
            
            clean_returns(&market_returns[0, i], n_days, stride)
            optimize_svcj(&market_returns[0, i], n_days, stride, &p)
            
            # Note: out_spot is also (n_days, n_assets), so we need stride there too?
            # run_ukf_qmle expects contiguous output arrays.
            # CRITICAL: We cannot pass &out_spot[0,i] as a contiguous output buffer because it is strided.
            # WE MUST use a temp buffer in C or transpose. 
            # Solution: We run NULL output in optimize, but we need output here.
            # Implementation specific: We will let run_ukf_qmle write to a temp buffer then copy back
            # However, simpler for this wrapper: The C-kernel assumes out_* is contiguous 1D.
            # We can't easily fix that in C without more complex indexing.
            # Hack for speed: Just run it, but we can't write directly to out_spot strided.
            # We will alloc a small stack buffer since N is usually small (~252)? 
            # No, N is 2 years.
            
            # Since we are in nogil/OpenMP, we can't alloc numpy arrays.
            # We will use malloc (stdlib)
            pass 
            
    # RE-IMPLEMENTATION TO HANDLE OUTPUT STRIDE SAFELY
    # Python Loop for filling output is safer for memory, but slower.
    # To keep speed, we do:
    cdef double *temp_spot
    cdef double *temp_jump
    
    with nogil:
        for i in prange(n_assets):
            p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7; 
            p.lambda_j = 0.1; p.mu_j = -0.1; p.sigma_j = 0.1; p.mu = 0.0;
            
            # 1. Optimize
            optimize_svcj(&market_returns[0, i], n_days, stride, &p)
            
            # 2. Alloc Temp Buffer
            temp_spot = <double *> stdlib.malloc(n_days * sizeof(double))
            temp_jump = <double *> stdlib.malloc(n_days * sizeof(double))
            
            if temp_spot != NULL and temp_jump != NULL:
                # Run UKF to fill temp buffers (stride passed for input, output is contiguous temp)
                run_ukf_qmle(&market_returns[0, i], n_days, stride, &p, temp_spot, temp_jump)
                
                # Copy back to strided output matrix
                for t in range(n_days):
                    out_spot[t, i] = temp_spot[t]
                    out_jump[t, i] = temp_jump[t]
                
                stdlib.free(temp_spot)
                stdlib.free(temp_jump)

    return {"spot_vol": out_spot, "jump_prob": out_jump}

def generate_residue_analysis(double[:] returns, int forward_window):
    cdef int n = returns.shape[0]
    cdef np.ndarray[double, ndim=1] residues = np.zeros(n - forward_window)
    cdef SVCJParams p
    p.kappa = 2.0; p.theta = 0.04; p.sigma_v = 0.3; p.rho = -0.7;
    p.lambda_j = 0.1; p.mu_j = -0.1; p.sigma_j = 0.1; p.mu = 0.0;
    
    cdef np.ndarray[double, ndim=1] spot_vol = np.zeros(n)
    
    clean_returns(&returns[0], n, 1)
    optimize_svcj(&returns[0], n, 1, &p)
    run_ukf_qmle(&returns[0], n, 1, &p, &spot_vol[0], NULL)
    
    cdef int t
    for t in range(n - forward_window):
        residues[t] = returns[t + forward_window] - (p.mu - 0.5 * spot_vol[t]*spot_vol[t]);
        
    return residues

# Helper for malloc
from libc cimport stdlib