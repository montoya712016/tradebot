from __future__ import annotations

import logging
import time
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# ML Imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from realtime.bot.settings import LiveSettings
from modules.prepare_features.features import calculate_features_for_window
from modules.prepare_features.feature_studio import FeatureStudio

log = logging.getLogger("realtime.components.strategy")

class StrategyEngine:
    """
    Manages ML models, Feature Engineering, and Scoring.
    """
    def __init__(self, settings: LiveSettings, feature_flags: dict):
        self.settings = settings
        self.feature_flags = feature_flags
        
        self.models: Dict[str, Any] = {} # entry, exit, etc
        self.meta: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.last_score_wallclock: Dict[str, float] = {}
        
        # GPU Predictor state
        self.predictor = None
        self.gpu_enabled = False

    def load_models(self):
        """Loads models from disk."""
        path = self.settings.models_dir
        if not os.path.exists(path):
            log.error("[strat] models_dir not found: %s", path)
            return

        found_any = False
        # Entry Model
        entry_path = os.path.join(path, "xgb_entry_model.json")
        if os.path.exists(entry_path) and xgb:
            bst = xgb.Booster()
            bst.load_model(entry_path)
            self.models["entry"] = bst
            found_any = True
            log.info("[strat] Loaded entry model: %s", entry_path)
        
        # Meta
        meta_path = os.path.join(path, "meta.pkl")
        if os.path.exists(meta_path):
            try:
                self.meta = joblib.load(meta_path)
                self.feature_cols = self.meta.get("feature_cols", [])
                log.info("[strat] Loaded meta.pkl (%d cols)", len(self.feature_cols))
            except Exception as e:
                log.error("[strat] Failed to load meta.pkl: %s", e)

        # GPU logic (simplified)
        if self.settings.use_gpu_score and xgb and self.models.get("entry"):
             self._enable_gpu_predictor(self.models["entry"], self.feature_cols)

    def _enable_gpu_predictor(self, booster, feat_cols, enable_gpu=True):
        """Optimizes XGBoost for GPU inference if available."""
        if not enable_gpu:
            return
        try:
            # Try to set gpu_hist (only works if compiled with GPU support)
            try:
                booster.set_param({"device": "cuda", "tree_method": "hist"})
            except Exception:
                pass
            self.gpu_enabled = True
            log.info("[strat] GPU inference enabled (experimental)")
        except Exception as e:
            log.warning("[strat] Failed to enable GPU: %s", e)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates features using FeatureStudio/existing logic."""
        if df.empty:
            return df
        
        # Ensure 'dates' column exists if needed by feature logic, or use index
        # Current logic usually uses existing columns
        
        # Use centralized feature calculation
        # Adapting to match original: prepare_features.features.calculate_features_for_window
        # Or FeatureStudio if available.
        
        # Shim to ensure compatibility
        try:
            return calculate_features_for_window(df, self.feature_cols)
        except Exception as e:
            log.error("[strat] Feature calculation failed: %s", e)
            return pd.DataFrame()

    def score_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Runs the model for a single symbol dataframe.
        Returns a decision dict or None.
        """
        if df is None or len(df) < 50: # Minimum size check
            return None

        # 1. Build Features
        feat_df = self.build_features(df)
        if feat_df.empty or self.feature_cols and len(feat_df.columns) < len(self.feature_cols):
            return None
        
        # Get last row (most recent)
        latest = feat_df.iloc[[-1]] # Keep as DataFrame
        
        # 2. Predict Entry
        model = self.models.get("entry")
        if not model or not xgb:
            return None
            
        # Ensure columns match
        if self.feature_cols:
            # Add missing as 0
            for c in self.feature_cols:
                if c not in latest.columns:
                    latest[c] = 0.0
            # Order correctly
            latest = latest[self.feature_cols]
        
        dmatrix = xgb.DMatrix(latest)
        pred_entry = model.predict(dmatrix) # Returns numpy array
        
        p_entry = float(pred_entry[0])
        
        decision = {
            "symbol": symbol,
            "ts_ms": int(latest.index[-1].value // 1_000_000) if hasattr(latest.index[-1], 'value') else 0,
            "p_entry": p_entry,
            "price": float(df["close"].iloc[-1]),
            "buy": False,
            "sell": False
        }
        
        # 3. Apply Thresholds
        threshold = float(self.settings.entry_threshold or 0.5)
        if p_entry >= threshold:
            decision["buy"] = True
            
        # Log (trace)
        log.info("[score][trace] %s pe=%.4f thresh=%.3f buy=%s", symbol, p_entry, threshold, decision["buy"])
        
        self.last_score_wallclock[symbol] = time.time()
        return decision
