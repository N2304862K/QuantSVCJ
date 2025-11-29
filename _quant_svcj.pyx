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

def c_joint_fit(double[:] returns, double[:] strikes, double[:] prices, double[:] T, int[:] types, double S0, double r):
    cdef int nr = returns.shape[0]
    cdef int no = strikes.shape[0]
    cdef SVCJResult res
    with nogil:
        # Pass types pointer
        res = optimize_svcj(&returns[0], nr, 1.0/252.0, &strikes[0], &prices[0], &T[0], &types[0], no, S0, r, 1)
    return {
        "kappa": res.p.kappa, "theta": res.p.theta, "sigma_v": res.p.sigma_v, 
        "rho": res.p.rho, "lambda": res.p.lambda, "mu_j": res.p.mu_j, 
        "sigma_j": res.p.sigma_j, "spot_vol": res.spot_vol, "jump_prob": res.jump_prob
    }

def c_rolling_fit(double[:] returns, int window):
    cdef int T = returns.shape[0]
    cdef int n_out = T - window + 1
    if n_out < 1: return np.zeros((0, 9))
    cdef double[:, ::1] out = np.zeros((n_out, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    for i in prange(n_out, nogil=True):
        res = optimize_svcj(&returns[i], window, 1.0/252.0, NULL, NULL, NULL, NULL, 0, 0, 0, 0)
        out[i,0]=res.p.kappa; out[i,1]=res.p.theta; out[i,2]=res.p.sigma_v;
        out[i,3]=res.p.rho; out[i,4]=res.p.lambda; out[i,5]=res.p.mu_j;
        out[i,6]=res.p.sigma_j; out[i,7]=res.spot_vol; out[i,8]=res.jump_prob;
    return np.asarray(out)

def c_market_screen(double[:, ::1] matrix):
    cdef int n_a = matrix.shape[0]
    cdef int n_t = matrix.shape[1]
    cdef double[:, ::1] out = np.zeros((n_a, 9), dtype=np.float64)
    cdef int i
    cdef SVCJResult res
    for i in prange(n_a, nogil=True):
        res = optimize_svcj(&matrix[i, 0], n_t, 1.0/252.0, NULL, NULL, NULL, NULL, 0, 0, 0, 0)
        out[i,0]=res.p.kappa; out[i,1]=res.p.theta; out[i,2]=res.p.sigma_v;
        out[i,3]=res.p.rho; out[i,4]=res.p.lambda; out[i,5]=res.p.mu_j;
        out[i,6]=res.p.sigma_j; out[i,7]=res.spot_vol; out[i,8]=res.jump_prob;
    return np.asarray(out)

def c_multi_rolling_fit(double[:, ::1] matrix, int window):
    cdef int n_a = matrix.shape[0]
    cdef int n_t = matrix.shape[1]
    cdef int n_win = n_t - window + 1
    if n_win < 1: return np.zeros((0, 0, 0))
    
    cdef double[:, :, ::1] out = np.zeros((n_a, n_win, 9), dtype=np.float64)
    cdef int i, t
    cdef SVCJResult res
    for i in prange(n_a, nogil=True):
        for t in range(n_win):
            res = optimize_svcj(&matrix[i, t], window, 1.0/252.0, NULL, NULL, NULL, NULL, 0, 0, 0, 0)
            out[i, t, 0]=res.p.kappa; out[i, t, 1]=res.p.theta; out[i, t, 2]=res.p.sigma_v;
            out[i, t, 3]=res.p.rho; out[i, t, 4]=res.p.lambda; out[i, t, 5]=res.p.mu_j;
            out[i, t, 6]=res.p.sigma_j; out[i, t, 7]=res.spot_vol; out[i, t, 8]=res.jump_prob;
    return np.asarray(out)