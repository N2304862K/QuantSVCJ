import numpy as np
import pandas as pd

try:
    import _quant_svcj
except ImportError:
    raise ImportError("QuantSVCJ C-Extension not built.")

COL_NAMES = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _prepare_rets(data):
        if hasattr(data, 'values') and np.nanmax(data.values) > 5.0:
            if isinstance(data, pd.DataFrame):
                return np.log(data / data.shift(1)).dropna(), True
            return np.log(data / data.shift(1)).dropna(), True
        return data, False

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Method 1: Joint Fit (History + Full Option Surface)
        option_chain: DataFrame with ['strike', 'price', 'T', 'type'] (type='call'/'put')
        """
        rets, was_price = QuantSVCJ._prepare_rets(price_history)
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        if was_price and hasattr(price_history, 'iloc'):
            S0 = price_history.iloc[-1]
        else:
            S0 = option_chain['strike'].mean()

        # Convert Types to Int (Call=1, Put=0)
        # Assumes column 'type' exists. If not, assumes all Calls.
        if 'type' in option_chain.columns:
            types = np.where(option_chain['type'].str.lower().str.contains('call'), 1, 0).astype(np.int32)
        else:
            types = np.ones(len(option_chain), dtype=np.int32)

        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        types_c = np.ascontiguousarray(types, dtype=np.int32)
        
        res = _quant_svcj.c_joint_fit(ret_arr, ks, ps, ts, types_c, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(price_history, window=252):
        """Method 2: Single Rolling"""
        rets, _ = QuantSVCJ._prepare_rets(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        raw = _quant_svcj.c_rolling_fit(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=COL_NAMES)

    @staticmethod
    def analyze_market_screen(price_matrix):
        """Method 3: Multi-Asset Screen"""
        rets, _ = QuantSVCJ._prepare_rets(price_matrix)
        assets = rets.columns
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        raw = _quant_svcj.c_market_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=COL_NAMES)

    @staticmethod
    def analyze_multi_rolling(price_matrix, window=252):
        """Method 4: Multi-Asset Rolling Feature Generation"""
        rets, _ = QuantSVCJ._prepare_rets(price_matrix)
        assets = rets.columns
        dates = rets.index[window-1:]
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw_3d = _quant_svcj.c_multi_rolling_fit(mat, window)
        
        # Flatten Tensor: [Time, Asset*Param]
        reshaped = raw_3d.transpose(1, 0, 2)
        n_time, n_assets, n_params = reshaped.shape
        flat_data = reshaped.reshape(n_time, n_assets * n_params)
        
        flat_cols = [f"{asset}_{param}" for asset in assets for param in COL_NAMES]
        return pd.DataFrame(flat_data, index=dates, columns=flat_cols)