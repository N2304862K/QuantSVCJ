import numpy as np
import pandas as pd
import warnings

# Import the compiled C-Extension
try:
    import _svcj_impl
except ImportError:
    try:
        from . import _svcj_impl
    except ImportError:
        raise ImportError("QuantSVCJ C-Extension not found. Install via 'pip install .'")

COLUMNS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    
    @staticmethod
    def fit_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Joint Calibration (History + Options).
        Input: 
           price_history: pd.Series (Prices)
           option_chain: pd.DataFrame (columns: strike, price, T)
        """
        # 1. Auto-Detect Prices/Returns
        vals = price_history.values.astype(np.float64)
        if vals[-1] > 5.0: 
            S0 = vals[-1]
            # Log Returns: ln(P_t / P_{t-1})
            log_rets = np.log(vals[1:] / vals[:-1])
        else:
            # Fallback if user passes returns
            S0 = option_chain['strike'].mean()
            log_rets = vals

        # 2. Parse Options
        opts = option_chain[['strike', 'price', 'T']].dropna()
        ks = np.ascontiguousarray(opts['strike'].values, dtype=np.float64)
        ps = np.ascontiguousarray(opts['price'].values, dtype=np.float64)
        ts = np.ascontiguousarray(opts['T'].values, dtype=np.float64)
        
        # 3. Execute C Engine
        res = _svcj_impl.run_joint(
            np.ascontiguousarray(log_rets, dtype=np.float64),
            ks, ps, ts,
            float(S0), float(risk_free_rate)
        )
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def fit_rolling(price_history, window=252):
        """
        Rolling Window Analysis.
        Input: pd.Series (Prices)
        """
        vals = price_history.values.astype(np.float64)
        
        # Determine Returns & Index
        if vals[0] > 5.0:
            log_rets = np.log(vals[1:] / vals[:-1])
            out_idx = price_history.index[1:][window-1:]
        else:
            log_rets = vals
            out_idx = price_history.index[window-1:]
            
        mat = _svcj_impl.run_rolling(
            np.ascontiguousarray(log_rets, dtype=np.float64), 
            int(window)
        )
        return pd.DataFrame(mat, index=out_idx, columns=COLUMNS)

    @staticmethod
    def fit_screen(price_matrix):
        """
        Multi-Asset Screen.
        Input: pd.DataFrame (Index=Date, Cols=Assets, Values=Prices)
        """
        # Convert Prices Matrix to Returns Matrix
        # axis=0 diff
        rets = np.log(price_matrix / price_matrix.shift(1)).dropna()
        
        # Transpose to (Assets, Time) for C-memory layout
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        res = _svcj_impl.run_screen(mat)
        return pd.DataFrame(res, index=rets.columns, columns=COLUMNS)