# kg_bootstrap.py
from sqlalchemy import create_engine
from sqlalchemy import text

from app.services.kg_adapter import (
    ensure_kg_tables,
    ingest_counters_from_fvf,
    ingest_styles_from_recos,
)

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

def main():
    engine = create_engine(DB_URL)

    # 0) Ensure KG tables exist (this actually executes the DDL)
    ensure_kg_tables(engine)

    # 1) Ingest phase-aware "COUNTERS" edges from formation_vs_formation
    #    (this is the historical prior that explains *why* a form counters another in a phase)
    ingest_counters_from_fvf(engine, provider=None)   # or "sportec"/"metrica" to filter

    # 2) (Optional) Ingest style tags as "EXHIBITS" edges from your saved recos
    #    (helps later if you want style-aware priors/decay)
    ingest_styles_from_recos(engine, limit=200)

    # 3) Quick sanity prints
    with engine.begin() as con:
        n_edges = con.execute(text("SELECT COUNT(*) FROM kg_edges")).scalar()
        n_nodes = con.execute(text("SELECT COUNT(*) FROM kg_nodes")).scalar()
        print(f"[KG] nodes={n_nodes}, edges={n_edges}")

if __name__ == "__main__":
    main()
