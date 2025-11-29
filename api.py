import numpy as np
import pandas as pd
from ._quant_svcj import c_joint_fit, c_rolling_fit, c_market_screen

COL_NAMES = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _to_log_returns(prices):
        """Internal: Safe conversion to log returns."""
        if isinstance(prices, pd.Series) or isinstance(prices, pd.DataFrame):
            # Check if already returns (small values ~0)
            if prices.iloc[0:5].abs().max() < 0.5: 
                # Likely returns, pass through
                return prices
            else:
                # Likely prices, compute log returns
                return np.log(prices / prices.shift(1)).dropna()
        return prices

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Snapshot Analysis (History + Option Joint Fit).
        
        Args:
            price_history (pd.Series): Historical Close prices (or log returns).
            option_chain (pd.DataFrame): Columns ['strike', 'price', 'T'].
            risk_free_rate (float): Annualized rate.
        """
        # 1. Handle Price History -> S0 and Returns
        if isinstance(price_history, pd.Series):
            S0 = price_history.iloc[-1] # Optimal S0: Last traded price
            # If user passed returns, S0 will be small, which is wrong. 
            # We assume user passes Prices as requested.
            # If values < 1.0, user might have passed returns. 
            # In that case, we need to infer S0 from options (less accurate).
            if S0 < 2.0 and option_chain['strike'].mean() > 10.0:
                 # User passed returns, infer S0 from ATM strike
                 S0 = option_chain['strike'].mean()
            
            rets = QuantSVCJ._to_log_returns(price_history)
            ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        else:
            raise ValueError("price_history must be a Pandas Series")

        # 2. Handle Options
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        
        # 3. Execute
        res = c_joint_fit(ret_arr, ks, ps, ts, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(price_history, window=252):
        """
        Rolling Analysis.
        Args:
            price_history (pd.Series): Historical Prices.
            window (int): Lookback period.
        """
        rets = QuantSVCJ._to_log_returns(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        raw = c_rolling_fit(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=COL_NAMES)

    @staticmethod
    def analyze_market_screen(price_matrix):
        """
        Multi-Asset Screen.
        Args:
            price_matrix (pd.DataFrame): Cols=Assets, Index=Date. Prices.
        """
        # Calculate Returns
        rets = np.log(price_matrix / price_matrix.shift(1)).dropna()
        assets = rets.columns
        
        # Transpose for C (N_assets, T_time)
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw = c_market_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=COL_NAMES)