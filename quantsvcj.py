import numpy as np
import pandas as pd
import warnings

# Try import compiled extension
try:
    import _svcj_impl
except ImportError:
    # If installed via pip, it might be directly available
    try:
        from . import _svcj_impl
    except ImportError:
        raise ImportError("QuantSVCJ compiled extension not found. Please install via pip.")

COLUMNS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    
    @staticmethod
    def fit_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Snapshot Fit: History + Options.
        """
        # 1. Prices to Returns
        vals = price_history.values
        if vals[-1] > 5.0: # Check if prices
            S0 = vals[-1]
            log_rets = np.log(vals[1:] / vals[:-1])
        else: # Fallback if returns
            S0 = option_chain['strike'].mean()
            log_rets = vals

        # 2. Options
        opts = option_chain[['strike', 'price', 'T']].dropna()
        
        # 3. Call C
        res = _svcj_impl.run_joint(
            np.ascontiguousarray(log_rets, dtype=np.float64),
            np.ascontiguousarray(opts['strike'].values, dtype=np.float64),
            np.ascontiguousarray(opts['price'].values, dtype=np.float64),
            np.ascontiguousarray(opts['T'].values, dtype=np.float64),
            float(S0), float(risk_free_rate)
        )
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def fit_rolling(price_history, window=252):
        """
        Rolling Window Fit.
        """
        vals = price_history.values
        if vals[0] > 5.0:
            log_rets = np.log(vals[1:] / vals[:-1])
            idx = price_history.index[1:][window-1:]
        else:
            log_rets = vals
            idx = price_history.index[window-1:]
            
        mat = _svcj_impl.run_rolling(np.ascontiguousarray(log_rets, dtype=np.float64), window)
        return pd.DataFrame(mat, index=idx, columns=COLUMNS)

    @staticmethod
    def fit_screen(price_matrix):
        """
        Market Screen (Multi-Asset).
        """
        rets = np.log(price_matrix / price_matrix.shift(1)).dropna()
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        res = _svcj_impl.run_screen(mat)
        return pd.DataFrame(res, index=rets.columns, columns=COLUMNS)