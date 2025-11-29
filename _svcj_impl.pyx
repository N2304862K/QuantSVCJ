# distutils: sources = svcj.c
# distutils: include_dirs = .
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    typedef struct SVCJParams:
        double kappa; double theta; double sigma_v; double rho;
        double lambda_; "lambda" # Escape keyword
        double mu_j; double sigma_j;
    
    typedef struct SVCJResult:
        SVCJParams p
        double spot_vol
        double jump_prob
        double error

    SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                             double* strikes, double* prices, double* T_exp, int n_opts,
                             double S0, double r, int mode) nogil

# --- Python-Visible Functions ---

def run_joint_calibration(double[::1] returns, double[::1] strikes, double[::1] prices, double[::1] T, double S0, double r):
    cdef int n_ret = returns.shape[0]
    cdef int n_opts = strikes.shape[0]
    cdef SVCJResult res
    
    with nogil:
        res = optimize_svcj(&returns[0], n_ret, 1.0/252.0, 
                            &strikes[0], &prices[0], &T[0], n_opts,
                            S0, r, 1)
                            
    return _pack_dict(res)

def run_rolling_analysis(double[::1] returns, int window):
    cdef int total = returns.shape[0]
    cdef int n_out = total - window + 1
    if n_out < 1: return np.zeros((0, 9))
    
    cdef double[:, ::1] out = np.zeros((n_out, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    cdef double dt = 1.0/252.0
    
    for i in prange(n_out, nogil=True):
        res = optimize_svcj(&returns[i], window, dt, 
                            NULL, NULL, NULL, 0, 
                            0, 0, 0)
        _pack_row(out, i, res)
        
    return np.asarray(out)

def run_market_screen(double[:, ::1] ret_matrix):
    cdef int n_assets = ret_matrix.shape[0]
    cdef int n_obs = ret_matrix.shape[1]
    cdef double[:, ::1] out = np.zeros((n_assets, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    
    for i in prange(n_assets, nogil=True):
        res = optimize_svcj(&ret_matrix[i, 0], n_obs, 1.0/252.0,
                            NULL, NULL, NULL, 0,
                            0, 0, 0)
        _pack_row(out, i, res)
        
    return np.asarray(out)

# --- Internal Helpers ---
cdef void _pack_row(double[:, ::1] out, int i, SVCJResult res) nogil:
    out[i,0]=res.p.kappa; out[i,1]=res.p.theta; out[i,2]=res.p.sigma_v;
    out[i,3]=res.p.rho;   out[i,4]=res.p.lambda_; out[i,5]=res.p.mu_j;
    out[i,6]=res.p.sigma_j; out[i,7]=res.spot_vol; out[i,8]=res.jump_prob;

def _pack_dict(SVCJResult res):
    return {
        "kappa": res.p.kappa, "theta": res.p.theta, "sigma_v": res.p.sigma_v,
        "rho": res.p.rho, "lambda": res.p.lambda_, "mu_j": res.p.mu_j,
        "sigma_j": res.p.sigma_j, "spot_vol": res.spot_vol, "jump_prob": res.jump_prob
    }