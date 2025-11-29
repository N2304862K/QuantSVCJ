import numpy as np
import pandas as pd
import warnings

# Try import compiled extension
try:
    from . import _svcj_impl
except ImportError:
    try:
        import _svcj_impl
    except ImportError:
        raise ImportError("QuantSVCJ C-Extension not compiled. Run 'pip install .'")

COLUMNS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    """
    Main Entry Point.
    Accepts Raw Prices (Series/Matrix) and Option Chains.
    """

    @staticmethod
    def fit_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Input: 
          price_history: pd.Series (Prices)
          option_chain: pd.DataFrame (columns: strike, price, T)
        """
        # 1. Auto-Detect S0 and Log Returns
        vals = price_history.values
        if vals[-1] > 5.0: # Prices
            S0 = vals[-1]
            log_rets = np.log(vals[1:] / vals[:-1])
        else: # Already returns
            log_rets = vals
            # Infer S0 from Option Chain ATM
            S0 = option_chain['strike'].mean()

        rets_c = np.ascontiguousarray(log_rets, dtype=np.float64)
        
        # 2. Prepare Options
        # Auto-clean
        opts = option_chain[['strike', 'price', 'T']].dropna()
        ks = np.ascontiguousarray(opts['strike'].values, dtype=np.float64)
        ps = np.ascontiguousarray(opts['price'].values, dtype=np.float64)
        ts = np.ascontiguousarray(opts['T'].values, dtype=np.float64)
        
        # 3. Run
        res = _svcj_impl.run_joint_calibration(rets_c, ks, ps, ts, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Fit")

    @staticmethod
    def fit_rolling(price_history, window=252):
        """
        Rolling window analysis.
        Input: pd.Series (Prices)
        """
        vals = price_history.values
        # Convert to returns if prices
        if vals[0] > 5.0:
            log_rets = np.log(vals[1:] / vals[:-1])
            dates = price_history.index[1:][window-1:]
        else:
            log_rets = vals
            dates = price_history.index[window-1:]
            
        rets_c = np.ascontiguousarray(log_rets, dtype=np.float64)
        
        mat = _svcj_impl.run_rolling_analysis(rets_c, window)
        return pd.DataFrame(mat, index=dates, columns=COLUMNS)

    @staticmethod
    def fit_screen(price_matrix):
        """
        Multi-Asset Parallel Fit.
        Input: pd.DataFrame (Index=Date, Cols=Assets) - PRICES
        """
        # Calculate returns matrix
        ret_df = np.log(price_matrix / price_matrix.shift(1)).dropna()
        assets = ret_df.columns
        
        # Transpose to (Assets, Time) for C-access
        # Ensure contiguous memory block
        mat = np.ascontiguousarray(ret_df.T.values, dtype=np.float64)
        
        res = _svcj_impl.run_market_screen(mat)
        return pd.DataFrame(res, index=assets, columns=COLUMNS)