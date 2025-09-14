# opponent_style_tagger.py
from __future__ import annotations
from typing import Dict, Iterable, Optional, Set
import numpy as np
import pandas as pd

# Defaults are intentionally simple & tunable
DEFAULT_THRESH = dict(
    ppda_high=12.0,          # higher => less pressing
    ppda_low=7.0,            # lower  => high pressing
    press_rate_high=0.33,    # >= 33% of opp-possession frames pressed
    avg_press_high=55.0,     # Herold/Adrienko composite (0..100+)
    width_high=38.0,         # meters
    width_low=28.0,
    depth_high=40.0,
    depth_low=30.0,
    possession_low=44.0,     # %
    possession_high=56.0,    # %
)

def _get(m: Dict, key: str, default=np.nan) -> float:
    v = m.get(key, default)
    try: return float(v)
    except: return np.nan

def _bool(b) -> bool:
    try: return bool(b)
    except: return False

def tag_styles(
    data_graph=None,
    metrics: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Set[str]:
    """
    Heuristic opponent style tags from:
      - passing graph (optional, if you supply it)
      - per-window/per-match metrics you already compute (xT, PPDA, press_rate, width, depth, possession, ...)
    Return a *set* of tags; kept tiny & fast.
    """
    T = {**DEFAULT_THRESH, **(thresholds or {})}
    M = metrics or {}

    # Pull the common fields your pipeline already has (name them from *opponent* pov)
    ppda        = _get(M, "ppda")
    ppda_press  = _get(M, "ppda_press")
    press_rate  = _get(M, "press_rate")
    avg_press   = _get(M, "avg_press")
    width       = _get(M, "width")
    depth       = _get(M, "depth")
    possession  = _get(M, "possession_pct")
    xt          = _get(M, "xt")

    tags: Set[str] = set()

    # Pressing identity
    if np.isfinite(ppda) and ppda <= T["ppda_low"]:
        tags.add("high_pressing")
    if np.isfinite(ppda) and ppda >= T["ppda_high"]:
        tags.add("low_pressing")
    if np.isfinite(press_rate) and press_rate >= T["press_rate_high"]:
        tags.add("press_triggers_often")
    if np.isfinite(avg_press) and avg_press >= T["avg_press_high"]:
        tags.add("physically_intense_press")

    # Block height (proxy via PPDA + press rate) — coarse but useful
    if "high_pressing" in tags and "press_triggers_often" in tags:
        tags.add("high_block")
    elif "low_pressing" in tags:
        tags.add("low_block")
    else:
        tags.add("mid_block")

    # Shape / occupation
    if np.isfinite(width) and width >= T["width_high"]:
        tags.add("wide_overloads")
    if np.isfinite(width) and width <= T["width_low"] and np.isfinite(depth) and depth <= T["depth_low"]:
        tags.add("compact_block")
    if np.isfinite(depth) and depth >= T["depth_high"]:
        tags.add("vertical_stretch")

    # Possession & transition flavor
    if np.isfinite(possession) and possession <= T["possession_low"]:
        tags.add("counter_attacking")
    if np.isfinite(possession) and possession >= T["possession_high"]:
        tags.add("possession_play")

    # Threat footprint (optional nudge)
    if np.isfinite(xt) and xt >= 0.25:
        tags.add("halfspace_threat")

    # Optional: derive wing bias from passing graph if present
    # Expect node attributes 'y' (meters) and edge weights 'w'
    try:
        if data_graph is not None and hasattr(data_graph, "edge_index") and hasattr(data_graph, "edge_attr"):
            # Very small, robust estimator of left/right usage
            ei = data_graph.edge_index.numpy()
            w  = np.asarray(data_graph.edge_attr).reshape(-1)
            y  = np.asarray(getattr(data_graph, "ypos", None))  # if you store node y-pos
            if y is not None and len(y) > 0:
                src, dst = ei[0], ei[1]
                use_left  = w[(y[src] < 0) | (y[dst] < 0)].sum() if w.size else 0.0
                use_right = w[(y[src] > 0) | (y[dst] > 0)].sum() if w.size else 0.0
                if use_left > 1.2 * (use_right + 1e-6):  tags.add("left_bias")
                if use_right > 1.2 * (use_left + 1e-6):  tags.add("right_bias")
    except Exception:
        pass

    # (Tiny) mutual exclusivity clean-up
    if "high_block" in tags and "low_block" in tags:
        tags.discard("low_block")
    if "left_bias" in tags and "right_bias" in tags:
        # keep none if symmetric
        tags.discard("left_bias"); tags.discard("right_bias")

    return tags
