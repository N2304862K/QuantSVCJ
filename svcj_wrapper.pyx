# distutils: language = c
# distutils: sources = svcjmath.c
import numpy as np
import pandas as pd
cimport numpy as np

cdef extern from "svcjmath.h":
    int svcj_matrix_fit(const double*, int, int, int, int, double*)
    void svcj_snapshot_fit(const double*, int, const double*, const double*, const double*, int, double, double, double*)
    int NUM_PARAMS # 10

# --- 1. Full Matrix Rolling Fit (Time Series Only) ---
def generate_svcj_factor_matrix(object log_returns_df, int window_size, int step_size):
    # Prepare Input
    cdef np.ndarray[np.float64_t, ndim=2] returns_matrix = log_returns_df.values.astype(np.float64)
    asset_names = log_returns_df.columns.tolist()
    
    cdef int total_T = returns_matrix.shape[0]
    cdef int total_A = returns_matrix.shape[1]
    cdef int max_rolls = (total_T - window_size) // step_size + 1
    
    if max_rolls <= 0: return pd.DataFrame()
    
    # Allocate Buffer: [Rolls * Assets * Params]
    cdef np.ndarray[np.float64_t, ndim=1] output_buffer = np.zeros(max_rolls * total_A * NUM_PARAMS, dtype=np.float64)
    
    # Run C Core
    cdef int actual_rolls = svcj_matrix_fit(
        &returns_matrix[0, 0], 
        total_T, total_A, window_size, step_size, &output_buffer[0]
    )
    
    # Reshape: (Rolls, Assets, Params)
    cdef np.ndarray[np.float64_t, ndim=3] tensor = \
        output_buffer[:actual_rolls * total_A * NUM_PARAMS].reshape(actual_rolls, total_A, NUM_PARAMS)
    
    # Flatten to 2D for DataFrame: (Rolls, Assets * Params)
    cdef np.ndarray[np.float64_t, ndim=2] final_matrix = tensor.reshape(actual_rolls, -1)
    
    # Columns
    param_names = ['mu', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda', 'mu_J', 'sigma_J', 'spot_vol', 'jump_prob']
    cols = [f'{asset}_{p}' for asset in asset_names for p in param_names]
    
    # Index
    idx_points = np.arange(window_size - 1, window_size - 1 + actual_rolls * step_size, step_size)
    idx = log_returns_df.index[idx_points]
    
    return pd.DataFrame(final_matrix, index=idx, columns=cols)

# --- 2. Snapshot Fit (History + Option Chain) ---
def analyze_snapshot(object price_series, object option_chain, double risk_free_rate=0.04):
    # Data Prep
    vals = price_series.values
    # Detect if Prices or Returns. If > 5.0, assume Prices.
    if vals[-1] > 5.0:
        S0 = vals[-1]
        log_rets = np.log(vals[1:] / vals[:-1])
    else:
        # If returns passed, try to infer S0 from ATM option, else fail soft
        if option_chain is not None and not option_chain.empty:
            S0 = option_chain['strike'].mean()
        else:
            S0 = 100.0 
        log_rets = vals

    cdef np.ndarray[np.float64_t, ndim=1] c_rets = np.ascontiguousarray(log_rets, dtype=np.float64)
    
    # Options Prep
    if option_chain is not None and not option_chain.empty:
        opts = option_chain[['strike', 'price', 'T']].dropna()
        ks = np.ascontiguousarray(opts['strike'].values, dtype=np.float64)
        ps = np.ascontiguousarray(opts['price'].values, dtype=np.float64)
        ts = np.ascontiguousarray(opts['T'].values, dtype=np.float64)
        n_opts = len(ks)
    else:
        ks = np.zeros(0, dtype=np.float64)
        ps = np.zeros(0, dtype=np.float64)
        ts = np.zeros(0, dtype=np.float64)
        n_opts = 0
        
    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros(10, dtype=np.float64)
    
    # Run C
    svcj_snapshot_fit(
        &c_rets[0] if len(c_rets)>0 else NULL, len(c_rets),
        &ks[0] if n_opts>0 else NULL, 
        &ps[0] if n_opts>0 else NULL, 
        &ts[0] if n_opts>0 else NULL, n_opts,
        S0, risk_free_rate,
        &out[0]
    )
    
    cols = ['mu', 'kappa', 'theta', 'sigma_v', 'rho', 'lambda', 'mu_J', 'sigma_J', 'spot_vol', 'jump_prob']
    return pd.Series(out, index=cols, name="SVCJ_Snapshot")