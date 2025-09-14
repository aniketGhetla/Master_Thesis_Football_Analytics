# app/services/tactical_recommender_service.py
import json
import numpy as np
from sqlalchemy import create_engine, text
from app.services.tactical_recommender_kg_new import TacticalRecommender

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
RECO_TABLE = "tactical_recommendations_temp"   

def _ensure_reco_table(engine):
    with engine.begin() as con:
        con.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {RECO_TABLE} (
            id SERIAL PRIMARY KEY,
            provider TEXT,
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
        );"""))

def run_tactical_recommendation(
    provider: str,
    our_side: str,
    opponent_form: str,
    opponent_phase: str,
    *,
    id2form_path: str = "id2form.json",
    gnn_logits: np.ndarray | None = None,
    strategy: str | None = None,
    opp_metrics: dict | None = None,
    our_metrics: dict | None = None,
    data_graph=None,
    save_to_db: bool = True,
):
    # id2form mapping
    try:
        with open(id2form_path, "r") as f:
            id2form = {int(k): v for k, v in json.load(f).items()}
    except Exception:
        # fallback: a small space so it runs without the model
        id2form = {0: "4-2-3-1", 1: "4-4-2", 2: "4-3-3"}

    # logits
    if gnn_logits is None:
        # uniform over classes if model not wired yet
        gnn_logits = np.zeros((len(id2form),), dtype=np.float32)

    rec = TacticalRecommender(DB_URL).recommend(
        provider=provider,
        our_side=our_side,
        opponent_form=opponent_form,
        opponent_phase=opponent_phase,
        gnn_logits=gnn_logits,
        id2form=id2form,
        data_graph=data_graph,
        strategy=strategy,
        opp_metrics=opp_metrics,
        our_metrics=our_metrics,
        alpha=0.55,
        beta=0.45,
        topk=3,
    )

    if save_to_db:
        engine = create_engine(DB_URL)
        _ensure_reco_table(engine)
        with engine.begin() as con:
            con.execute(
                text(f"""
                INSERT INTO {RECO_TABLE} 
                (provider, our_side, opponent_form, opponent_phase, strategy,
                 recommended_formation, confidence, topk, style_tags, suggestion_text, evidence)
                VALUES (:prov, :side, :of, :oph, :strat, :recf, :conf, :topk, :tags, :text, :ev)
                """),
                {
                    "prov": provider,
                    "side": our_side,
                    "of": opponent_form,
                    "oph": opponent_phase,
                    "strat": strategy,
                    "recf": rec["recommended_formation"],
                    "conf": rec["confidence"],
                    "topk": json.dumps(rec["topk"]),
                    "tags": json.dumps(rec["style_tags"]),
                    "text": rec["suggestion_text"],
                    "ev": json.dumps(rec["evidence"]),
                },
            )
    return rec
