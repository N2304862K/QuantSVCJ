import numpy as np
import pandas as pd
from ._quant_svcj import c_joint_fit, c_rolling_fit, c_market_screen, c_multi_rolling

COL_NAMES = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _to_log_returns(prices):
        """Internal helper."""
        if isinstance(prices, (pd.Series, pd.DataFrame)):
            # Heuristic: if values > 5, assume prices. If < 0.5, assume returns.
            if prices.iloc[0:10].abs().max() > 1.0:
                return np.log(prices / prices.shift(1)).dropna()
            return prices # Already returns
        return prices

    # --- Method 1: Snapshot ---
    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Single Asset Joint Fit.
        option_chain must have: 'strike', 'price', 'T', 'type' ('call'/'put').
        """
        if isinstance(price_history, pd.Series):
             S0 = price_history.iloc[-1]
             rets = QuantSVCJ._to_log_returns(price_history)
             ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        else: raise ValueError("price_history must be Series")

        # Process Options
        df = option_chain.copy()
        # Map type to int: Call=1, Put=0
        if 'type' not in df.columns: df['type'] = 'call' # Default
        type_flag = df['type'].astype(str).str.lower().apply(lambda x: 1 if 'c' in x else 0).values.astype(np.int32)
        
        ks = df['strike'].values.astype(np.float64)
        ps = df['price'].values.astype(np.float64)
        ts = df['T'].values.astype(np.float64)
        
        res = c_joint_fit(ret_arr, ks, ps, ts, type_flag, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    # --- Method 2: Single Rolling ---
    @staticmethod
    def analyze_rolling(price_history, window=252):
        rets = QuantSVCJ._to_log_returns(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        raw = c_rolling_fit(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=COL_NAMES)

    # --- Method 3: Market Screen (Static) ---
    @staticmethod
    def analyze_market_screen(price_matrix):
        rets = np.log(price_matrix / price_matrix.shift(1)).dropna()
        assets = rets.columns
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw = c_market_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=COL_NAMES)

    # --- Method 4: Multi-Asset Rolling ---
    @staticmethod
    def analyze_multi_rolling(price_matrix, window=252):
        """
        Output: DataFrame. Index=Date. Columns= Asset-Param (Flattened).
        """
        rets = np.log(price_matrix / price_matrix.shift(1)).dropna()
        dates = rets.index[window-1:]
        assets = rets.columns
        
        # C expects (Assets, Time)
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        # Raw: [Windows, Assets*9]
        raw = c_multi_rolling(mat, window)
        
        # Construct Column Names: [AAPL-kappa, AAPL-theta, ..., MSFT-kappa...]
        flat_cols = []
        for a in assets:
            for c in COL_NAMES:
                flat_cols.append(f"{a}-{c}")
                
        return pd.DataFrame(raw, index=dates, columns=flat_cols)