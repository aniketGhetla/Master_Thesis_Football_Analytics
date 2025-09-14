# app/services/reco_batch_step.py
from types import SimpleNamespace
from app.services.predict_tactical_recos import (
    ensure_out_table, ensure_kg_tables, pick_context_from_fvf,
    TeamPhasePassDataset, TeamPhaseGNN, TacticalRecommender,
    build_metric_dicts, candidate_forms_from_fvf, DB_URL, MODEL_PATH, ID2FORM_PATH, TOPK, DEFAULT_ALPHA, DEFAULT_BETA
)
import json, torch
import numpy as np
from sqlalchemy import create_engine
from torch_geometric.data import Batch
from scipy.special import softmax

def run_reco_batch(provider="sportec", our_side="home", strategy="both"):
    engine = create_engine(DB_URL)
    ensure_out_table(engine)
    ensure_kg_tables(engine)

    opp_form, opp_phase = pick_context_from_fvf(engine, provider, our_side)
    ds = TeamPhasePassDataset(provider=provider)
    data_graph = next(d for d in ds if str(d.our_side)==our_side and str(d.opponent_form)==opp_form and str(d.opponent_phase)==opp_phase)

    with open(ID2FORM_PATH, "r") as f:
        id2form = {int(k): v for k, v in json.load(f).items()}

    model = TeamPhaseGNN(in_dim=6, hid=64, num_classes=len(id2form))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    batch = Batch.from_data_list([data_graph])
    with torch.no_grad():
        logits, _ = model(batch)
        full_probs = softmax(logits.cpu().numpy()[0], axis=-1)

    full_list = [id2form[i] for i in sorted(id2form)]
    cand_forms = candidate_forms_from_fvf(engine, provider, our_side, opp_form, opp_phase, id2form, min_pairs=5)
    if len(cand_forms) < 4:
        top_idx = np.argsort(full_probs)[::-1][:6]
        for f in [full_list[i] for i in top_idx]:
            if f not in cand_forms:
                cand_forms.append(f)
            if len(cand_forms) >= 4:
                break

    probs_sub = np.array([[ full_probs[ full_list.index(f) ] for f in cand_forms ]])
    id2form_sub = {i: f for i, f in enumerate(cand_forms)}

    opp_metrics, our_metrics = build_metric_dicts(engine, provider, our_side, opp_form, opp_phase)
    rec = TacticalRecommender(DB_URL)
    strategies = [strategy] if strategy != "both" else ["attack","defense"]

    results = []
    for strat in strategies:
        out = rec.recommend(
            provider=provider, our_side=our_side,
            opponent_form=opp_form, opponent_phase=opp_phase,
            gnn_logits=probs_sub, id2form=id2form_sub, data_graph=data_graph,
            alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, topk=TOPK,
            strategy=strat, opp_metrics=opp_metrics, our_metrics=our_metrics,
        )
        results.append(SimpleNamespace(strategy=strat, **out))
    return {
        "opponent_form": opp_form, "opponent_phase": opp_phase,
        "results": [dict(r.__dict__) for r in results],
    }
