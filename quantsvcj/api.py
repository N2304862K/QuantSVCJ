import numpy as np
import pandas as pd
from ._quant_svcj import c_joint_fit, c_rolling_fit, c_market_screen

COL_NAMES = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _prepare_returns(prices):
        """Converts prices to log returns safely."""
        if isinstance(prices, pd.Series) or isinstance(prices, pd.DataFrame):
            # Heuristic: if max value > 10, it's definitely Price, not Returns
            if prices.max().max() > 2.0: 
                return np.log(prices / prices.shift(1)).dropna()
            return prices # Assume already returns
        return prices

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Inputs:
            price_history: pd.Series (Prices)
            option_chain: pd.DataFrame ['strike', 'price', 'T']
            risk_free_rate: float
        """
        # 1. Infer S0 from last price
        if hasattr(price_history, 'iloc'):
            S0 = price_history.iloc[-1]
        else:
            S0 = price_history[-1]

        # 2. Convert to Returns
        rets = QuantSVCJ._prepare_returns(price_history)
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)

        # 3. Prepare Options
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        
        # 4. Run C-Engine
        res = c_joint_fit(ret_arr, ks, ps, ts, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(price_history, window=252):
        """
        Inputs: price_history (Series), window (int)
        """
        rets = QuantSVCJ._prepare_returns(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        raw = c_rolling_fit(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=COL_NAMES)

    @staticmethod
    def analyze_market_screen(price_matrix):
        """
        Inputs: price_matrix (DataFrame: Index=Date, Cols=Assets)
        """
        rets = QuantSVCJ._prepare_returns(price_matrix)
        assets = rets.columns
        # Transpose for C layout (Assets, Time)
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw = c_market_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=COL_NAMES)