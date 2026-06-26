"""
models_foundation.py
====================
``FoundationModelForecaster`` — a guarded wrapper around time-series foundation
models (Amazon Chronos / Chronos-Bolt, Google TimesFM, IBM Tiny Time Mixers /
Granite-TS) for use as *benchmark entrants* in the monthly lineup.

Honest risk assessment (do not skip):
  * M6 (Makridakis et al. 2024) found ~98% of entrants could not beat a random
    forecast at the monthly horizon.
  * Rahimikia (2025) finds off-the-shelf TSFMs "perform poorly zero-shot and
    fine-tuned" on financial returns; only from-scratch domain pretraining helps.
  * Public commodity series likely contaminate TSFM pretraining (arXiv 2510.13654),
    so any apparent edge must be confirmed on data *after* the model's training
    cutoff.
  * Therefore the *only* finance-adjacent place these have shown value is the
    **volatility** target — not the price-level return.

This class is built so the bet is *available and auditable* but never breaks a
run: if the backend package is not installed (the default offline state) it
degrades to the random walk (zeros) and labels itself accordingly.  Wire it on a
machine with the backend installed, point it at the volatility target, and
validate on post-cutoff hold-out data before trusting any result.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models import BaseForecaster

logger = logging.getLogger(__name__)

# backend key -> (importable module name, human label)
_BACKENDS = {
    "chronos": ("chronos", "Chronos"),
    "timesfm": ("timesfm", "TimesFM"),
    "tinytimemixer": ("tsfm_public", "TinyTimeMixer"),
}


def _is_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:  # pragma: no cover - defensive
        return False


class FoundationModelForecaster(BaseForecaster):
    """Zero-shot time-series foundation model wrapped to the BaseForecaster API.

    Parameters
    ----------
    backend:
        ``"chronos"`` | ``"timesfm"`` | ``"tinytimemixer"`` | ``"auto"`` (first
        installed backend).  When the chosen backend is unavailable the model
        degrades to the random walk and reports ``available_ == False``.
    context_length:
        Number of trailing target observations fed to the model as context.
    """

    def __init__(self, backend: str = "auto", context_length: int = 64) -> None:
        self.backend = backend
        self.context_length = context_length
        self._y: Optional[np.ndarray] = None
        self._resolved: Optional[str] = None
        self.available_ = False

    def _resolve_backend(self) -> Optional[str]:
        if self.backend == "auto":
            for key, (mod, _label) in _BACKENDS.items():
                if _is_available(mod):
                    return key
            return None
        mod = _BACKENDS.get(self.backend, (None, None))[0]
        return self.backend if (mod and _is_available(mod)) else None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FoundationModelForecaster":
        self._y = np.asarray(y, dtype=float)
        self._resolved = self._resolve_backend()
        self.available_ = self._resolved is not None
        if not self.available_:
            logger.warning(
                "FoundationModelForecaster: backend %r unavailable — falling back "
                "to the random walk. Install the package and validate on "
                "post-training-cutoff data before use.", self.backend)
        return self

    def _zero_shot(self, n: int) -> np.ndarray:
        """Best-effort zero-shot mean forecast, broadcast over the test block.

        Only exercised when a backend is installed; kept minimal because the
        offline path is the RW fallback. Any failure degrades to zeros.
        """
        ctx = self._y[-self.context_length:]
        if self._resolved == "chronos":
            import torch  # noqa: F401
            from chronos import ChronosPipeline
            pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-small")
            q = pipe.predict(torch.tensor(ctx, dtype=torch.float32).unsqueeze(0),
                             prediction_length=1)
            return np.full(n, float(np.asarray(q).reshape(-1)[0]))
        if self._resolved == "timesfm":
            import timesfm
            tfm = timesfm.TimesFm()
            fc, _ = tfm.forecast([ctx], freq=[0])
            return np.full(n, float(np.asarray(fc).reshape(-1)[0]))
        if self._resolved == "tinytimemixer":
            # Granite-TS / TTM exposes a HF pipeline; left to the host install.
            raise NotImplementedError("Wire the TTM pipeline on the host machine.")
        raise RuntimeError("no resolved backend")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.available_ or self._y is None:
            return np.zeros(len(X))            # honest RW fallback (TSFM unavailable)
        try:
            return self._zero_shot(len(X))
        except Exception as exc:
            logger.warning("FoundationModel %s failed (%s) — RW fallback",
                           self._resolved, exc)
            return np.zeros(len(X))

    @property
    def name(self) -> str:
        if self._resolved:
            return f"TSFM[{_BACKENDS[self._resolved][1]}]"
        return f"TSFM[{self.backend}->RW]"
