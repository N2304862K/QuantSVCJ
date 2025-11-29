# distutils: sources = svcj.c
# distutils: include_dirs = .
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange

cdef extern from "svcj.h":
    typedef struct SVCJParams:
        double kappa; double theta; double sigma_v; double rho;
        double lambda_; "lambda"
        double mu_j; double sigma_j;
    typedef struct SVCJResult:
        SVCJParams p; double spot_vol; double jump_prob; double error
    SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                             double* strikes, double* prices, double* T_exp, int n_opts,
                             double S0, double r, int mode) nogil

def run_joint(double[::1] returns, double[::1] ks, double[::1] ps, double[::1] ts, double S0, double r):
    cdef SVCJResult res
    with nogil:
        res = optimize_svcj(&returns[0], returns.shape[0], 1.0/252.0, &ks[0], &ps[0], &ts[0], ks.shape[0], S0, r, 1)
    return _pack(res)

def run_rolling(double[::1] returns, int win):
    cdef int n = returns.shape[0] - win + 1
    if n < 1: return np.zeros((0,9))
    cdef double[:, ::1] out = np.zeros((n, 9), dtype=np.float64)
    cdef int i
    for i in prange(n, nogil=True):
        _pack_row(out, i, optimize_svcj(&returns[i], win, 1.0/252.0, NULL, NULL, NULL, 0, 0, 0, 0))
    return np.asarray(out)

def run_screen(double[:, ::1] mat):
    cdef int n = mat.shape[0]
    cdef double[:, ::1] out = np.zeros((n, 9), dtype=np.float64)
    cdef int i
    for i in prange(n, nogil=True):
        _pack_row(out, i, optimize_svcj(&mat[i, 0], mat.shape[1], 1.0/252.0, NULL, NULL, NULL, 0, 0, 0, 0))
    return np.asarray(out)

cdef void _pack_row(double[:, ::1] out, int i, SVCJResult res) nogil:
    out[i,0]=res.p.kappa; out[i,1]=res.p.theta; out[i,2]=res.p.sigma_v; out[i,3]=res.p.rho;
    out[i,4]=res.p.lambda_; out[i,5]=res.p.mu_j; out[i,6]=res.p.sigma_j;
    out[i,7]=res.spot_vol; out[i,8]=res.jump_prob;

def _pack(SVCJResult res):
    return {"kappa": res.p.kappa, "theta": res.p.theta, "sigma_v": res.p.sigma_v, "rho": res.p.rho,
            "lambda": res.p.lambda_, "mu_j": res.p.mu_j, "sigma_j": res.p.sigma_j,
            "spot_vol": res.spot_vol, "jump_prob": res.jump_prob}