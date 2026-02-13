from __future__ import annotations

import json
import os
from dataclasses import dataclass

@dataclass
class BiasState:
    # bias means: (forecast - observed) in feet
    # corrected_forecast = forecast - bias
    rolling_bias_ft: float = 0.0
    n: int = 0

def load_bias(path: str) -> BiasState:
    if not os.path.exists(path):
        return BiasState()
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return BiasState(
            rolling_bias_ft=float(j.get("rolling_bias_ft", 0.0)),
            n=int(j.get("n", 0)),
        )
    except Exception:
        # If file is corrupt or empty, start fresh
        return BiasState()

def save_bias(path: str, state: BiasState) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"rolling_bias_ft": state.rolling_bias_ft, "n": state.n}, f, indent=2)

def update_bias(state: BiasState, new_error_ft: float, max_n: int = 60) -> BiasState:
    """
    Updates the rolling bias.
    new_error_ft = (Forecast - Observed)
    """
    n = min(state.n + 1, max_n)
    if state.n < max_n:
        # standard average while building up sample size
        rolling = (state.rolling_bias_ft * state.n + new_error_ft) / n
    else:
        # exponential moving average-ish update for stable size
        alpha = 1.0 / max_n
        rolling = (1 - alpha) * state.rolling_bias_ft + alpha * new_error_ft
    return BiasState(rolling_bias_ft=float(rolling), n=int(n))
