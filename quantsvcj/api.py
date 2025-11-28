import numpy as np
import pandas as pd
from ._c_wrapper import fit_multi_asset_history, fit_single_asset_joint

class QuantSVCJ:
    @staticmethod
    def generate_features(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Input: DataFrame of Log Returns (Columns = Assets, Rows = Time)
        Output: DataFrame of SVCJ Parameters (Index = Assets, Cols = Params)
        """
        # Ensure C-contiguous float64 matrix
        mat = returns_df.T.values.astype(np.float64)
        
        # Call C Engine
        raw_params = fit_multi_asset_history(mat)
        
        cols = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j"]
        return pd.DataFrame(raw_params, index=returns_df.columns, columns=cols)

    @staticmethod
    def calibrate_single(returns_series: pd.Series, 
                         option_chain: pd.DataFrame, 
                         spot_price: float, 
                         risk_free_rate: float = 0.04):
        """
        Input:
            returns_series: Historical log returns
            option_chain: DataFrame with columns ['strike', 'price', 'T']
            spot_price: Current S0
        """
        # Prepare buffers
        ret_buf = returns_series.values.astype(np.float64)
        
        k_buf = option_chain['strike'].values.astype(np.float64)
        p_buf = option_chain['price'].values.astype(np.float64)
        t_buf = option_chain['T'].values.astype(np.float64)
        
        # Call C Engine (Joint Optimization)
        params = fit_single_asset_joint(ret_buf, k_buf, p_buf, t_buf, 
                                        spot_price, risk_free_rate)
        return params