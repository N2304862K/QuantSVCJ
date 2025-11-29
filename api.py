import numpy as np
import pandas as pd
from ._quant_svcj import c_analyze_snapshot, c_analyze_rolling, c_analyze_screen

COLS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    
    @staticmethod
    def analyze_snapshot(returns, option_chain, risk_free_rate=0.04):
        """
        Joint Calibration: History + Options.
        Inputs:
          returns: Series/Array of log returns.
          option_chain: DataFrame ['strike', 'price', 'T'].
          risk_free_rate: float.
        """
        # Auto-config: Convert inputs
        ret_arr = np.ascontiguousarray(returns.values if hasattr(returns, 'values') else returns, dtype=np.float64)
        
        # Spot Price estimate (Last close implicitly used in ret generation, 
        # but for options we need S0 relative to strikes).
        # We assume strikes are raw. We estimate S0 from ATM or user provided context.
        # *Strict adherence*: User said no pre-selection. 
        # We assume the user provides valid option chain for the asset.
        # We estimate S0 as the average of strikes (roughly ATM).
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        s0_est = np.mean(ks) 

        res = c_analyze_snapshot(ret_arr, ks, ps, ts, s0_est, risk_free_rate)
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(returns, window=None):
        """
        Rolling History Analysis.
        Inputs:
          returns: Series with Datetime Index.
          window: int (default=252).
        """
        if window is None: window = 252
        
        # Validate
        if not isinstance(returns, pd.Series):
             raise ValueError("Rolling requires Pandas Series with Date Index")
             
        vals = np.ascontiguousarray(returns.values, dtype=np.float64)
        dates = returns.index[window-1:]
        
        res_mat = c_analyze_rolling(vals, window)
        
        return pd.DataFrame(res_mat, index=dates, columns=COLS)

    @staticmethod
    def analyze_market_screen(returns_matrix):
        """
        Multi-Asset Screen.
        Inputs:
           returns_matrix: DataFrame (Index=Date, Cols=Assets)
        """
        assets = returns_matrix.columns
        # Transpose to (Assets, Time) C-contiguous
        mat = np.ascontiguousarray(returns_matrix.T.values, dtype=np.float64)
        
        res_mat = c_analyze_screen(mat)
        
        return pd.DataFrame(res_mat, index=assets, columns=COLS)