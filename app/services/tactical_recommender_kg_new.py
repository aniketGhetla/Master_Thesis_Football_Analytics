# app/services/tactical_recommender_kg_new.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from scipy.special import softmax
except Exception:
    def softmax(x, axis=-1):
        x = np.asarray(x, dtype=float)
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

# ---- temp KG adapter import ----
try:
    # temp-aware prior (from kg_adapter_temp.py)
    from app.services.kg_adapter_temp import kg_prior_with_evidence_temp as _kg_prior_fn
except Exception:
    _kg_prior_fn = None

from app.services.opponent_style_tagger_new import tag_styles  # keep your tagger

TEMP_FVF_TABLE = "formation_vs_formation_new_temp"   # <— use temp table


def _softmax(x, t=1.0):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    x = x / max(t, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in d.items()}
    s = float(sum(max(0.0, v) for v in vals.values()))
    if s <= 1e-12:
        u = 1.0 / len(vals)
        return {k: u for k in vals}
    return {k: max(0.0, v) / s for k, v in vals.items()}


def _robust_minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    p5 = np.nanpercentile(s, 5) if np.isfinite(np.nanpercentile(s, 5)) else np.nanmin(s)
    p95 = np.nanpercentile(s, 95) if np.isfinite(np.nanpercentile(s, 95)) else np.nanmax(s)
    lo = 0.0 if not np.isfinite(p5) else p5
    hi = (lo + 1.0) if (not np.isfinite(p95) or p95 == lo) else p95
    return ((s - lo) / (hi - lo)).clip(0, 1)


def _matchup_prior_from_fvf(
    eng,
    provider: str,
    our_side: str,
    opp_form: str,
    opp_phase: str,
    candidate_forms: List[str],
    tags: Optional[List[str]] = None,
    strategy: Optional[str] = None,
) -> Tuple[Dict[str, float], List[dict]]:
    q = text(f"""
        SELECT *
        FROM {TEMP_FVF_TABLE}
        WHERE provider = :p
          AND {{opp_form_col}} = :f
          AND {{opp_phase_col}} = :ph
    """.format(
        opp_form_col = "away_form"  if our_side=="home" else "home_form",
        opp_phase_col = "phase_away" if our_side=="home" else "phase_home",
    ))
    df = pd.read_sql(q, eng, params={"p": provider, "f": opp_form, "ph": opp_phase})
    if df.empty:
        return {}, [{"source":"FVF","note":"no rows for context"}]

    our_form_col = "home_form" if our_side=="home" else "away_form"
    keep_cols = [
        our_form_col, "pairs",
        f"{our_side}_strength", f"{our_side}_xt", f"{our_side}_ppda", f"{our_side}_ppda_press",
        f"{our_side}_press_rate", f"{our_side}_avg_press",
        f"{our_side}_width", f"{our_side}_depth"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    g = (df[keep_cols]
         .groupby(our_form_col, dropna=False)
         .agg("mean")
         .reset_index()
         .rename(columns={our_form_col: "form"}))

    g = g[g["form"].isin(candidate_forms)]
    if g.empty:
        return {}, [{"source":"FVF","note":"no overlap with candidate forms"}]

    z = pd.DataFrame({"form": g["form"].values})
    z["pairs_w"]   = _robust_minmax(g["pairs"]).fillna(0.0)
    z["strength"]  = g[f"{our_side}_strength"].astype(float).clip(0,1).fillna(0.0)
    z["xt"]        = _robust_minmax(g[f"{our_side}_xt"])
    z["ppda_inv"]  = 1.0 - _robust_minmax(g[f"{our_side}_ppda"])
    z["ppdap_inv"] = 1.0 - _robust_minmax(g[f"{our_side}_ppda_press"])
    z["press_r"]   = _robust_minmax(g[f"{our_side}_press_rate"])
    z["avg_pr"]    = _robust_minmax(g[f"{our_side}_avg_press"])
    z["width"]     = _robust_minmax(g[f"{our_side}_width"])
    z["depth"]     = _robust_minmax(g[f"{our_side}_depth"])

    if strategy == "attack":
        w = {"strength":0.35, "xt":0.35, "ppda_inv":0.10, "ppdap_inv":0.05, "press_r":0.05, "avg_pr":0.05}
    elif strategy == "defense":
        w = {"strength":0.25, "xt":0.10, "ppda_inv":0.30, "ppdap_inv":0.15, "press_r":0.15, "avg_pr":0.05}
    else:
        w = {"strength":0.50, "xt":0.20, "ppda_inv":0.15, "ppdap_inv":0.05, "press_r":0.05, "avg_pr":0.05}

    tags = set(tags or [])
    if "compact_block" in tags:
        w["width"] = w.get("width", 0.0) + 0.05
        w["depth"] = w.get("depth", 0.0) + 0.05
    if "wide_overloads" in tags or "switches_of_play" in tags:
        w["ppda_inv"] = w.get("ppda_inv", 0.0) + 0.03
        w["press_r"]  = w.get("press_r",  0.0) + 0.02

    comp = (
        w["strength"]*z["strength"] + w["xt"]*z["xt"]
        + w["ppda_inv"]*z["ppda_inv"] + w["ppdap_inv"]*z["ppdap_inv"]
        + w["press_r"]*z["press_r"] + w["avg_pr"]*z["avg_pr"]
        + w.get("width",0.0)*z["width"] + w.get("depth",0.0)*z["depth"]
    )

    prior = (comp * (0.3 + 0.7*z["pairs_w"])).to_numpy()
    prior_map = _norm({f: float(s) for f, s in zip(z["form"], prior)})

    ev = [{
        "source":"FVF",
        "note":"metric-aware prior",
        "weights": w,
        "forms": {f: float(s) for f, s in zip(z["form"], comp.to_numpy())},
        "pairs_w": {f: float(p) for f, p in zip(z["form"], z["pairs_w"].to_numpy())},
        "tags": list(tags)
    }]
    return prior_map, ev


class TacticalRecommender:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    @staticmethod
    def _coach_note(opponent_form, opponent_phase, rec_form, tags, flavor=None):
        # ... keep your TAG_TIPS dictionary & logic exactly as in your file ...
        # (omitted here for brevity – paste your full block unchanged)
        return (
            f"Against a {opponent_form} in {opponent_phase}, deploy a {rec_form}. "
            f"Attack: ... Defense: ..."
        )

    def recommend(
        self,
        provider: str,
        our_side: str,
        opponent_form: str,
        opponent_phase: str,
        gnn_logits,
        id2form: Dict[int, str],
        data_graph,
        alpha: float = 0.55,
        beta: float = 0.45,
        topk: int = 3,
        strategy: Optional[str] = None,
        opp_metrics: Optional[Dict[str, float]] = None,
        our_metrics: Optional[Dict[str, float]] = None,
    ):
        tags_opp = tag_styles(data_graph, metrics=opp_metrics)
        tags_us  = tag_styles(None, metrics=our_metrics) if our_metrics else set()
        tags = tags_opp

        logits = np.asarray(gnn_logits)
        if logits.ndim == 2:
            logits = logits[0]
        num_classes = logits.shape[0]
        gnn_probs = softmax(logits.reshape(1, -1), axis=-1)[0]
        candidate_forms = [id2form[i] for i in range(num_classes)]
        gnn_map = {id2form[i]: float(gnn_probs[i]) for i in range(num_classes)}
        gnn_ev  = [{"source": "GNN", "formation": id2form[i], "prob": float(gnn_probs[i])}
                   for i in range(num_classes)]

        fvf_prior_map, fvf_ev = _matchup_prior_from_fvf(
            self.engine, provider, our_side, opponent_form, opponent_phase,
            candidate_forms=candidate_forms, tags=tags_opp, strategy=strategy
        )

        prior: Dict[str, float] = {}
        prior_ev: List[dict] = []
        if _kg_prior_fn is not None:
            try:
                raw, ev = _kg_prior_fn(self.engine, opponent_form, opponent_phase, provider=provider)
                prior_ev = ev or []
                kg_overlap = {k: float(v) for k, v in raw.items() if k in candidate_forms}
                prior = _norm(kg_overlap)
                if not prior:
                    prior = fvf_prior_map
                    if not prior_ev:
                        prior_ev = [{"source": "FVF.prior", "candidate_form": f, "support": float(s),
                                     "phase": opponent_phase, "provider": provider} for f, s in prior.items()]
            except Exception:
                prior = fvf_prior_map
                prior_ev = [{"source": "FVF.prior", "candidate_form": f, "support": float(s),
                             "phase": opponent_phase, "provider": provider} for f, s in prior.items()]
        else:
            prior = fvf_prior_map
            prior_ev = [{"source": "FVF.prior", "candidate_form": f, "support": float(s),
                         "phase": opponent_phase, "provider": provider} for f, s in prior.items()]

        alpha = float(alpha); beta = float(beta)
        forms = set(candidate_forms) | set(prior.keys())
        fused = {f: alpha * gnn_map.get(f, 0.0) + beta * prior.get(f, 0.0) for f in forms}
        fused = _norm(fused)

        ranking = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: max(1, topk)]
        rec_form, rec_conf = ranking[0]

        note = self._coach_note(opponent_form, opponent_phase, rec_form, tags, flavor=strategy)
        evidence = {
            "context": {"provider": provider, "our_side": our_side, "opponent_form": opponent_form,
                        "opponent_phase": opponent_phase, "strategy": strategy or "mixed",
                        "alpha": alpha, "beta": beta},
            "contributions": {"kg_prior": prior_ev, "gnn_probs": gnn_ev,
                              "style_tags_opponent": list(tags_opp), "style_tags_ours": list(tags_us)},
            "fusion": {"fused_scores": fused,
                       "topk": [{"formation": f, "score": float(s)} for f, s in ranking]},
            "reason_text": note,
        }

        return {
            "recommended_formation": rec_form,
            "confidence": float(rec_conf),
            "topk": [{"formation": f, "score": float(s)} for f, s in ranking],
            "style_tags": list(tags),
            "suggestion_text": note,
            "evidence": evidence,
        }
