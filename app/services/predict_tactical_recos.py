# app/services/predict_tactical_recos.py
import argparse, json
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine, text
from torch_geometric.data import Batch
from scipy.special import softmax

from app.services.gnn_team_phase_dataset import TeamPhasePassDataset
from app.services.gnn_team_phase_model import TeamPhaseGNN
from app.services.tactical_recommender_kg_new import TacticalRecommender
from app.services.kg_adapter import ensure_kg_tables

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
MODEL_PATH = "C:\\Users\\Harshita\\Documents\\FootBallProject\\app\\services\\models\\team_phase_gnn_new.pth"
ID2FORM_PATH = "C:\\Users\\Harshita\\Documents\\FootBallProject\\app\\services\\models\\id2form.json"

# TEMP tables
FVF_TABLE = "formation_vs_formation_new_temp"
OUT_TABLE = "tactical_recommendations_temp"

DEFAULT_ALPHA = 0.55
DEFAULT_BETA  = 0.45
TOPK = 3

def ensure_out_table(engine):
    with engine.begin() as con:
        con.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
            provider TEXT,
            match_id TEXT,
            our_side TEXT,
            opponent_form TEXT,
            opponent_phase TEXT,
            strategy TEXT,
            recommended_formation TEXT,
            confidence DOUBLE PRECISION,
            topk JSONB,
            style_tags JSONB,
            suggestion_text TEXT,
            evidence JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

def pick_context_from_fvf(engine, provider: str, our_side: str):
    df = pd.read_sql(text(f"SELECT * FROM {FVF_TABLE} WHERE provider = :p"),
                     engine, params={"p": provider})
    if df.empty:
        raise RuntimeError(f"No rows in {FVF_TABLE} for provider={provider}")

    if our_side == "home":
        df_ctx = df[["away_form", "phase_away", "pairs"]].dropna(subset=["away_form", "phase_away"]).rename(
            columns={"away_form":"opponent_form","phase_away":"opponent_phase"}
        )
    else:
        df_ctx = df[["home_form", "phase_home", "pairs"]].dropna(subset=["home_form", "phase_home"]).rename(
            columns={"home_form":"opponent_form","phase_home":"opponent_phase"}
        )
    if df_ctx.empty:
        raise RuntimeError(f"No opponent contexts for provider={provider}, side={our_side}")

    ctx = df_ctx.sort_values("pairs", ascending=False).iloc[0]
    return str(ctx["opponent_form"]), str(ctx["opponent_phase"])

def pick_graph_for_context(ds: TeamPhasePassDataset, our_side: str, opp_form: str, opp_phase: str):
    for i in range(len(ds)):
        d = ds[i]
        if str(d.our_side) == our_side and str(d.opponent_form) == opp_form and str(d.opponent_phase) == opp_phase:
            return d
    return ds[0]

# metrics (for style tags) from temp table
def build_metric_dicts(engine, provider, our_side, opponent_form, opponent_phase, table=FVF_TABLE):
    opp_form_col  = "away_form"  if our_side == "home" else "home_form"
    opp_phase_col = "phase_away" if our_side == "home" else "phase_home"

    df = pd.read_sql(
        text(f"""SELECT * FROM {table}
                 WHERE provider = :p AND {opp_form_col} = :f AND {opp_phase_col} = :ph"""),
        engine, params={"p": provider, "f": opponent_form, "ph": opponent_phase}
    )
    if df.empty:
        return {}, {}

    opp_prefix = "away" if our_side == "home" else "home"
    our_prefix = "home" if our_side == "home" else "away"

    def _m(prefix, name):
        col = f"{prefix}_{name}"
        return float(df[col].mean()) if col in df.columns and not df[col].isna().all() else float('nan')

    opp_metrics = {
        "ppda":           _m(opp_prefix, "ppda"),
        "ppda_press":     _m(opp_prefix, "ppda_press"),
        "press_rate":     _m(opp_prefix, "press_rate"),
        "avg_press":      _m(opp_prefix, "avg_press"),
        "width":          _m(opp_prefix, "width"),
        "depth":          _m(opp_prefix, "depth"),
        "possession_pct": _m(opp_prefix, "possession_pct"),
        "xt":             _m(opp_prefix, "xt"),
    }
    our_metrics = {
        "ppda":           _m(our_prefix, "ppda"),
        "ppda_press":     _m(our_prefix, "ppda_press"),
        "press_rate":     _m(our_prefix, "press_rate"),
        "avg_press":      _m(our_prefix, "avg_press"),
        "width":          _m(our_prefix, "width"),
        "depth":          _m(our_prefix, "depth"),
        "possession_pct": _m(our_prefix, "possession_pct"),
        "xt":             _m(our_prefix, "xt"),
    }
    return opp_metrics, our_metrics

def candidate_forms_from_fvf(engine, provider, our_side, opp_form, opp_phase, id2form, min_pairs=5):
    opp_form_col  = "away_form"  if our_side == "home" else "home_form"
    opp_phase_col = "phase_away" if our_side == "home" else "phase_home"
    our_form_col  = "home_form"  if our_side == "home" else "away_form"

    q = text(f"""
        SELECT {our_form_col} AS form, pairs
        FROM {FVF_TABLE}
        WHERE provider=:p AND {opp_form_col}=:f AND {opp_phase_col}=:ph
    """)
    df = pd.read_sql(q, engine, params={"p": provider, "f": opp_form, "ph": opp_phase})

    allowed = set(df.loc[df["pairs"].fillna(0) >= min_pairs, "form"].dropna().astype(str))
    full_list = [id2form[i] for i in sorted(id2form)]
    forms = [f for f in full_list if f in allowed]
    return forms or full_list

def main():
    ap = argparse.ArgumentParser(description="Generate tactical recommendations (attack/defense).")
    ap.add_argument("--provider", required=True, choices=["sportec", "metrica"])
    ap.add_argument("--our_side", required=True, choices=["home", "away"])
    ap.add_argument("--strategy", required=True, choices=["attack", "defense", "both"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta",  type=float, default=DEFAULT_BETA)
    ap.add_argument("--topk",  type=int, default=TOPK)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--id2form", default=ID2FORM_PATH)
    args = ap.parse_args()

    engine = create_engine(DB_URL)
    ensure_out_table(engine)
    ensure_kg_tables(engine)

    # 1) Context from temp FVF
    opponent_form, opponent_phase = pick_context_from_fvf(engine, args.provider, args.our_side)
    print(f"[INFO] Context → opp_form={opponent_form} | opp_phase={opponent_phase} | side={args.our_side} | strategy={args.strategy}")

    # 2) Graph
    ds = TeamPhasePassDataset(provider=args.provider)
    data_graph = pick_graph_for_context(ds, args.our_side, opponent_form, opponent_phase)

    # 3) Model
    with open(args.id2form, "r") as f:
        id2form = {int(k): v for k, v in json.load(f).items()}
    num_classes = len(id2form)

    model = TeamPhaseGNN(in_dim=6, hid=64, num_classes=num_classes)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    batch = Batch.from_data_list([data_graph])
    with torch.no_grad():
        logits, emb = model(batch)
        logits_np = logits.cpu().numpy()

    # GNN probs + candidate pool
    full_list = [id2form[i] for i in sorted(id2form)]
    idx_for_form = {f: i for i in enumerate(full_list)}
    full_probs = softmax(logits_np[0], axis=-1)

    cand_forms = candidate_forms_from_fvf(
        engine, args.provider, args.our_side, opponent_form, opponent_phase, id2form, min_pairs=5
    )

    MIN_K, TOP_N = 4, 6
    if len(cand_forms) < MIN_K:
        top_idx = np.argsort(full_probs)[::-1][:TOP_N]
        top_forms = [full_list[i] for i in top_idx]
        seen = set(cand_forms)
        for f in top_forms:
            if f not in seen:
                cand_forms.append(f); seen.add(f)
            if len(cand_forms) >= MIN_K:
                break

    probs_sub = np.array([full_probs[[full_list.index(f) for f in cand_forms]]])[None, :]
    id2form_sub = {i: f for i, f in enumerate(cand_forms)}

    # 4) Recommender
    rec = TacticalRecommender(db_url=DB_URL)
    opp_metrics, our_metrics = build_metric_dicts(engine, args.provider, args.our_side, opponent_form, opponent_phase)

    strategies = [args.strategy] if args.strategy != "both" else ["attack", "defense"]
    for strat in strategies:
        out = rec.recommend(
            provider=args.provider,
            our_side=args.our_side,
            opponent_form=opponent_form,
            opponent_phase=opponent_phase,
            gnn_logits=probs_sub,
            id2form=id2form_sub,
            data_graph=data_graph,
            alpha=args.alpha, beta=args.beta,
            topk=args.topk,
            strategy=strat,
            opp_metrics=opp_metrics,
            our_metrics=our_metrics,
        )

        safe_evidence = out.get("evidence", {
            "context": {
                "provider": args.provider,
                "our_side": args.our_side,
                "opponent_form": opponent_form,
                "opponent_phase": opponent_phase,
                "strategy": args.strategy,
                "alpha": args.alpha, "beta": args.beta
            },
            "contributions": {
                "kg_prior": [],
                "gnn_probs": out.get("topk", []),
                "style_tags": out.get("style_tags", []),
            },
            "fusion": {
                "fused_scores": {},
                "topk": out.get("topk", []),
            },
            "reason_text": out.get("suggestion_text", "")
        })

        row = {
            "provider": args.provider,
            "match_id": "auto",
            "our_side": args.our_side,
            "opponent_form": opponent_form,
            "opponent_phase": opponent_phase,
            "strategy": strat,
            "recommended_formation": out["recommended_formation"],
            "confidence": out["confidence"],
            "topk": json.dumps(out["topk"]),
            "style_tags": json.dumps(out["style_tags"]),
            "suggestion_text": out["suggestion_text"],
            "evidence": json.dumps(safe_evidence),
        }

        with engine.begin() as con:
            con.execute(text(f"""
                INSERT INTO {OUT_TABLE}
                (provider, match_id, our_side, opponent_form, opponent_phase, strategy,
                 recommended_formation, confidence, topk, style_tags, suggestion_text, evidence)
                VALUES (:provider, :match_id, :our_side, :opponent_form, :opponent_phase, :strategy,
                        :recommended_formation, :confidence,
                        CAST(:topk AS JSONB), CAST(:style_tags AS JSONB), :suggestion_text, CAST(:evidence AS JSONB));
            """), row)

        print(f"✅ saved {strat} recommendation → {out['recommended_formation']} (conf {out['confidence']:.2f})")

if __name__ == "__main__":
    main()
