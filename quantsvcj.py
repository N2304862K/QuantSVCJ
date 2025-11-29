import numpy as np
import pandas as pd
try:
    import _quant_svcj
except ImportError:
    raise ImportError("C-Extension missing. Run 'pip install .'")

PARAMS = ["kappa", "theta", "sigma_v", "rho", "lambda", "mu_j", "sigma_j", "spot_vol", "jump_prob"]

class QuantSVCJ:
    @staticmethod
    def _get_rets(data):
        """Auto-detect Price vs Returns."""
        if hasattr(data, 'values'):
            if np.nanmax(data.values) > 5.0: # Is Price
                return np.log(data / data.shift(1)).dropna(), True
            return data, False
        return data, False

    @staticmethod
    def analyze_snapshot(price_history, option_chain, risk_free_rate=0.04):
        """
        Method 1: Joint Fit.
        option_chain: DataFrame with ['strike', 'price', 'T', 'type'].
                      'type': 'call' or 'put'.
        """
        rets, was_price = QuantSVCJ._get_rets(price_history)
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        # S0 Detection
        S0 = price_history.iloc[-1] if (was_price and hasattr(price_history, 'iloc')) else option_chain['strike'].mean()

        # Prepare Option Arrays
        ks = option_chain['strike'].values.astype(np.float64)
        ps = option_chain['price'].values.astype(np.float64)
        ts = option_chain['T'].values.astype(np.float64)
        
        # Handle Types (Call=1, Put=0)
        # Check if 'type' col exists, else assume Call
        if 'type' in option_chain.columns:
            types = option_chain['type'].apply(lambda x: 1 if str(x).lower().startswith('c') else 0).values.astype(np.int32)
        else:
            types = np.ones(len(option_chain), dtype=np.int32)

        res = _quant_svcj.c_joint_fit(ret_arr, ks, ps, ts, types, float(S0), float(risk_free_rate))
        return pd.Series(res, name="SVCJ_Snapshot")

    @staticmethod
    def analyze_rolling_single(price_history, window=252):
        """Method 2: Rolling Single Asset."""
        rets, _ = QuantSVCJ._get_rets(price_history)
        dates = rets.index[window-1:]
        ret_arr = np.ascontiguousarray(rets.values, dtype=np.float64)
        
        raw = _quant_svcj.c_rolling_single(ret_arr, window)
        return pd.DataFrame(raw, index=dates, columns=PARAMS)

    @staticmethod
    def analyze_market_screen(price_matrix):
        """Method 3: Multi-Asset Screen (Latest)."""
        rets, _ = QuantSVCJ._get_rets(price_matrix)
        assets = rets.columns
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        raw = _quant_svcj.c_screen(mat)
        return pd.DataFrame(raw, index=assets, columns=PARAMS)

    @staticmethod
    def analyze_rolling_multi(price_matrix, window=252):
        """
        Method 4: Rolling Multi-Asset.
        Returns: DataFrame indexed by Date, Columns = {Asset}_{Param}.
        """
        rets, _ = QuantSVCJ._get_rets(price_matrix)
        assets = rets.columns
        dates = rets.index[window-1:]
        
        # (N_Assets, T_Time)
        mat = np.ascontiguousarray(rets.T.values, dtype=np.float64)
        
        # Call C (Returns 3D array: [Asset, Window, Param])
        # Shape: (n_assets, n_windows, 9)
        raw_3d = _quant_svcj.c_rolling_multi(mat, window)
        
        # Flatten to DataFrame: Date Index, Columns = AAPL_kappa, AAPL_theta...
        final_df = pd.DataFrame(index=dates)
        
        for i, asset in enumerate(assets):
            # Extract (Windows, 9) for this asset
            asset_data = raw_3d[i]
            # Create Col Names
            cols = [f"{asset}_{p}" for p in PARAMS]
            # Assign
            temp_df = pd.DataFrame(asset_data, index=dates, columns=cols)
            final_df = pd.concat([final_df, temp_df], axis=1)
            
        return final_df