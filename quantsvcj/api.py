import numpy as np
import pandas as pd
from ._wrapper import fit_joint, fit_rolling, fit_screen

PARAM_COLS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j"]

class QuantSVCJ:
    
    @staticmethod
    def analyze_snapshot(returns_series, option_chain, risk_free_rate=0.04):
        """
        Mode 1: Single Asset Joint Calibration (History + Options)
        
        Args:
            returns_series (pd.Series/np.array): Log returns history.
            option_chain (pd.DataFrame): Must contain ['strike', 'price', 'T'].
            risk_free_rate (float): Annualized risk-free rate.
        
        Returns:
            pd.Series: Calibrated parameters.
        """
        # Data Prep
        rets = np.ascontiguousarray(returns_series.values if isinstance(returns_series, pd.Series) else returns_series, dtype=np.float64)
        
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        
        # Spot price is implicit in the options or provided. 
        # NOTE: Standard approach assumes S0 is current price at end of returns.
        # Ideally passed, but for this signature we assume S0 is roughly closest to ATM strike or user handles normalization.
        # To be robust, let's assume the user normalized options or we take the last price from external context.
        # *Correction based on prompt constraints*: Inputs are only Matrix/Options/Rf. 
        # We will assume S0=100 (normalized) or derive from options if possible. 
        # BETTER: Let's assume options are raw and S0 is roughly the mean of strikes if not provided.
        # However, precise S0 is needed. We will estimate S0 from the option chain (ATM).
        S0_est = ks[np.argmin(np.abs(ks - ks.mean()))] 

        params = fit_joint(rets, ks, ps, ts, S0_est, risk_free_rate)
        return pd.Series(params, index=PARAM_COLS, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(returns_series, window_size=252):
        """
        Mode 2: Rolling History Fit
        
        Args:
            returns_series (pd.Series): Time-indexed log returns.
            window_size (int): Lookback window.
            
        Returns:
            pd.DataFrame: Parameters indexed by Date (alignment with end of window).
        """
        if not isinstance(returns_series, pd.Series):
            raise ValueError("Rolling fit requires a pandas Series with Datetime Index.")
            
        vals = np.ascontiguousarray(returns_series.values, dtype=np.float64)
        dates = returns_series.index[window_size-1:]
        
        raw_res = fit_rolling(vals, window_size)
        
        return pd.DataFrame(raw_res, index=dates, columns=PARAM_COLS)

    @staticmethod
    def analyze_market_screen(returns_matrix):
        """
        Mode 3: Multi-Asset Feature Generation (Current Day)
        
        Args:
            returns_matrix (pd.DataFrame): Index=Time, Columns=Assets.
            
        Returns:
            pd.DataFrame: Index=Assets, Columns=Params.
        """
        # Transpose to (N_Assets, T_Time) for C iteration
        assets = returns_matrix.columns
        mat = np.ascontiguousarray(returns_matrix.T.values, dtype=np.float64)
        
        raw_res = fit_screen(mat)
        
        return pd.DataFrame(raw_res, index=assets, columns=PARAM_COLS)