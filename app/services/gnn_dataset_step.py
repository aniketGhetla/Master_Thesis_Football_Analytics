# app/services/gnn_dataset_step.py
import argparse
from statistics import mean

from app.services.gnn_team_phase_dataset import TeamPhasePassDataset

def run_gnn_dataset_step(
    provider: str = "sportec",
    formation_table: str = "formation_vs_formation_new_temp",  # use temp table from previous step
    limit: int | None = 500,
):
    """
    Builds a TeamPhasePassDataset (without training) to ensure data is ready for GNN.
    Returns a short summary dict for the dashboard.
    """
    ds = TeamPhasePassDataset(provider=provider, formation_table=formation_table, limit=limit)

    n_samples = len(ds)
    # peek a few items to summarize node/edge sizes (safe & quick)
    sizes = []
    peek = min(5, n_samples)
    for i in range(peek):
        g = ds[i]
        sizes.append((g.num_nodes, g.num_edges))

    summary = {
        "provider": provider,
        "formation_table": formation_table,
        "limit": limit,
        "num_graphs": n_samples,
        "avg_nodes": (int(mean(s[0] for s in sizes)) if sizes else 0),
        "avg_edges": (int(mean(s[1] for s in sizes)) if sizes else 0),
    }
    print(f"[GNN] {provider}: graphs={n_samples}, avg_nodes={summary['avg_nodes']}, avg_edges={summary['avg_edges']}")
    return summary

# CLI
def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="sportec", choices=["sportec","metrica"])
    ap.add_argument("--formation_table", default="formation_vs_formation_new_temp")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    s = run_gnn_dataset_step(provider=args.provider, formation_table=args.formation_table, limit=args.limit)
    print(s)

if __name__ == "__main__":
    _main()
