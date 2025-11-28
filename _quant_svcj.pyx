import numpy as np
import pandas as pd

# Import the C-Extension we just built
try:
    import _quant_svcj
except ImportError:
    raise ImportError("QuantSVCJ C-Extension not linked. Did installation fail?")

COL_NAMES = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _prepare_rets(data):
        """Converts Prices to Log Returns automatically."""
        # Check for DataFrame/Series
        if hasattr(data, 'values'):
            # Heuristic: if values > 5.0, it's Price, else Returns
            if np.nanmax(data.values) > 5.0:
                # Log returns: ln(Pt / Pt-1)
                if isinstance(data, pd.DataFrame):
                    return np.log(data / data.shift(1)).dropna(), True
                return np.log(data / data.shift(1)).dropna(), True
            return data, False
        return data, False

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Joint Fit (History + Options).
        price_history: pd.Series (Prices or Returns)
        option_chain: pd.DataFrame ['strike', 'price', 'T']
        """
        rets, was_price = QuantSVCJ._prepare_rets(price_history)
        
        # Ensure C-Contiguous array
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        # Estimate S0 (Spot)
        if was_price and hasattr(price_history, 'iloc'):
            S0 = price_history.iloc[-1]
        else:
            S0 = option_chain['strike'].mean()

        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        
        res = _quant_svcj.c_joint_fit(ret_arr, ks, ps, ts, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling(price_history, window=252):
        rets, _ = QuantSVCJ._prepare_rets(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        raw = _quant_svcj.c_rolling_fit(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=COL_NAMES)

    @staticmethod
    def analyze_market_screen(price_matrix):
        # Expecting DataFrame: Index=Date, Cols=Assets
        rets, _ = QuantSVCJ._prepare_rets(price_matrix)
        assets = rets.columns
        
        # Transpose to (Assets, Time) for C iteration
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw = _quant_svcj.c_market_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=COL_NAMES)