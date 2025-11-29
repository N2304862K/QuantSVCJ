import numpy as np
import pandas as pd
try:
    import _quant_svcj
except ImportError:
    _quant_svcj = None

PARAMS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _check():
        if _quant_svcj is None: raise ImportError("C-Extension not built. Run 'pip install .'")
    
    @staticmethod
    def _get_rets(data):
        if hasattr(data, 'values'):
            if np.nanmax(data.values) > 5.0: return np.log(data / data.shift(1)).dropna(), True
            return data, False
        return data, False

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        QuantSVCJ._check()
        rets, was_price = QuantSVCJ._get_rets(price_history)
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        S0 = price_history.iloc[-1] if (was_price and hasattr(price_history, 'iloc')) else option_chain['strike'].mean()
        
        # Parse Options
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        # Parse Type (Call=1, Put=0)
        if 'type' in option_chain.columns:
            types = option_chain['type'].apply(lambda x: 1 if str(x).lower().startswith('c') else 0).values.astype(np.int32)
        else:
            types = np.ones(len(option_chain), dtype=np.int32)

        res = _quant_svcj.c_joint_fit(ret_arr, ks, ps, ts, types, float(S0), float(risk_free_rate))
        return pd.Series(res, index=PARAMS, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling_single(price_history, window=252):
        QuantSVCJ._check()
        rets, _ = QuantSVCJ._get_rets(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        raw = _quant_svcj.c_rolling_single(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=PARAMS)

    @staticmethod
    def analyze_market_screen(price_matrix):
        QuantSVCJ._check()
        rets, _ = QuantSVCJ._get_rets(price_matrix)
        assets = rets.columns
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        raw = _quant_svcj.c_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=PARAMS)

    @staticmethod
    def analyze_rolling_multi(price_matrix, window=252):
        QuantSVCJ._check()
        rets, _ = QuantSVCJ._get_rets(price_matrix)
        assets = rets.columns
        dates = rets.index[window-1:]
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        # Get 3D Tensor [Asset, Window, Param]
        raw_3d = _quant_svcj.c_rolling_multi(mat, window)
        
        # Flatten to 2D DataFrame: [Date, (Asset_Param...)]
        final_dfs = []
        for i, asset in enumerate(assets):
            cols = [f"{asset}_{p}" for p in PARAMS]
            final_dfs.append(pd.DataFrame(raw_3d[i], index=dates, columns=cols))
            
        return pd.concat(final_dfs, axis=1)