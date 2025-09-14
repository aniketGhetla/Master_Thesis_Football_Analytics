# app/services/run_recommender_once.py
from app.services.tactical_recommender_service import run_tactical_recommendation

if __name__ == "__main__":
    out = run_tactical_recommendation(
        provider="sportec",
        our_side="home",
        opponent_form="4-3-3",
        opponent_phase="mid-block",
        gnn_logits=None,
        save_to_db=True,
    )
    print(out)
