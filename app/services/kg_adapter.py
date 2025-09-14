# kg_adapter.py
from typing import Dict, List, Tuple, Optional
import json, datetime as dt
import pandas as pd
from sqlalchemy import text

# ---------- PG schema for KG-lite ----------
DDL_NODES = """
CREATE TABLE IF NOT EXISTS kg_nodes (
  id SERIAL PRIMARY KEY,
  kind TEXT NOT NULL,
  key  TEXT NOT NULL,
  props JSONB DEFAULT '{}'::jsonb,
  UNIQUE(kind, key)
);
"""
DDL_EDGES = """
CREATE TABLE IF NOT EXISTS kg_edges (
  id SERIAL PRIMARY KEY,
  src_kind TEXT NOT NULL,
  src_key  TEXT NOT NULL,
  rel      TEXT NOT NULL,
  dst_kind TEXT NOT NULL,
  dst_key  TEXT NOT NULL,
  props JSONB DEFAULT '{}'::jsonb
);
"""
DDL_IDX = """
CREATE INDEX IF NOT EXISTS kg_edges_lookup ON kg_edges (src_kind, src_key, rel, dst_kind, dst_key);
CREATE INDEX IF NOT EXISTS kg_nodes_lookup ON kg_nodes (kind, key);
-- optional: speed up props lookups
CREATE INDEX IF NOT EXISTS kg_edges_props_gin ON kg_edges USING GIN (props);
"""

# Unique index for COUNTERS edges keyed by (src,dst,phase) so we can UPSERT
# Note: uses an expression on props->>'phase'
DDL_UNIQ_COUNTERS = """
CREATE UNIQUE INDEX IF NOT EXISTS kg_edges_counters_uniq
ON kg_edges (src_kind, src_key, rel, dst_kind, dst_key, (props->>'phase'))
WHERE rel = 'COUNTERS';
"""

def ensure_kg_tables(engine):
    with engine.begin() as con:
        con.execute(text(DDL_NODES))
        con.execute(text(DDL_EDGES))
        con.execute(text(DDL_IDX))
        con.execute(text(DDL_UNIQ_COUNTERS))

# ---------- helpers ----------
def _upsert_node(con, kind: str, key: str, props: dict = None):
    con.execute(text("""
        INSERT INTO kg_nodes(kind, key, props)
        VALUES (:k, :key, COALESCE(:p, '{}'::jsonb))
        ON CONFLICT (kind, key) DO UPDATE
        SET props = kg_nodes.props || EXCLUDED.props;
    """), {"k": kind, "key": key, "p": json.dumps(props or {})})

# --- replace your _insert_edge_agg_support with this version ---
def _insert_edge_agg_support(con, src, rel, dst, props: dict):
    """
    For rel='COUNTERS', UPSERT by (src,dst,phase) and add to support.
    For others, simple insert.
    """
    if rel != "COUNTERS":
        con.execute(text("""
            INSERT INTO kg_edges(src_kind, src_key, rel, dst_kind, dst_key, props)
            VALUES (:sk, :skey, :rel, :dk, :dkey, CAST(:p AS JSONB));
        """), {
            "sk": src[0], "skey": src[1], "rel": rel,
            "dk": dst[0], "dkey": dst[1],
            "p": json.dumps(props or {})
        })
        return

    # For COUNTERS: upsert and ADD support
    con.execute(text("""
        INSERT INTO kg_edges(src_kind, src_key, rel, dst_kind, dst_key, props)
        VALUES (:sk, :skey, 'COUNTERS', :dk, :dkey, CAST(:p AS JSONB))
        ON CONFLICT (src_kind, src_key, rel, dst_kind, dst_key, (props->>'phase'))
        WHERE kg_edges.rel = 'COUNTERS'
        DO UPDATE SET props = jsonb_set(
            kg_edges.props,
            '{support}',
            to_jsonb(
                COALESCE((kg_edges.props->>'support')::float, 0)
              + COALESCE((EXCLUDED.props->>'support')::float, 0)
            ),
            true
        );
    """), {
        "sk": src[0], "skey": src[1],
        "dk": dst[0], "dkey": dst[1],
        "p": json.dumps(props or {})
    })
# ---------- Ingest formation_vs_formation into COUNTERS edges ----------
def _clean_phase(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    x = str(x).strip().lower().replace(" ", "-")
    return x

def ingest_counters_from_fvf(engine, provider: Optional[str] = None):
    q = """
    SELECT provider, home_form, away_form, phase_home, phase_away, home_strength, away_strength
    FROM formation_vs_formation
    """
    if provider:
        q += " WHERE provider = :prov"
    df = pd.read_sql(text(q), engine, params={"prov": provider} if provider else None)
    if df.empty:
        print("[KG] No formation_vs_formation rows."); return

    # Build two directed sets (candidate -> opponent) with normalized phase
    home_rows = df[["provider","home_form","away_form","phase_away","home_strength"]].copy()
    home_rows.columns = ["provider","cand","opp","phase","support"]
    away_rows = df[["provider","away_form","home_form","phase_home","away_strength"]].copy()
    away_rows.columns = ["provider","cand","opp","phase","support"]
    all_rows = pd.concat([home_rows, away_rows], ignore_index=True)

    all_rows["phase"] = all_rows["phase"].map(_clean_phase)
    all_rows = all_rows.dropna(subset=["cand","opp","phase"])
    # coerce numeric, fill NaN with 0
    all_rows["support"] = pd.to_numeric(all_rows["support"], errors="coerce").fillna(0.0)

    # Aggregate to avoid duplicates
    grp = (all_rows.groupby(["provider","cand","opp","phase"], dropna=False)["support"]
           .sum()
           .reset_index())

    with engine.begin() as con:
        for r in grp.itertuples(index=False):
            prov, cand, opp, phase, sup = r
            _upsert_node(con, "Formation", str(cand))
            _upsert_node(con, "Formation", str(opp))
            _upsert_node(con, "Phase",     str(phase))
            _insert_edge_agg_support(
                con,
                src=("Formation", str(cand)),
                rel="COUNTERS",
                dst=("Formation", str(opp)),
                props={"phase": str(phase), "support": float(sup), "provider": prov}
            )
    print(f"[KG] COUNTERS edges ingested: {len(grp)} aggregated rows.")

# ---------- Ingest recent style tags from tactical_recommendations ----------
def ingest_styles_from_recos(engine, limit: Optional[int] = None):
    q = """
    SELECT opponent_form, opponent_phase, style_tags, created_at
    FROM tactical_recommendations
    ORDER BY created_at DESC
    """
    if limit: q += f" LIMIT {int(limit)}"
    df = pd.read_sql(text(q), engine)
    if df.empty:
        print("[KG] No recommendations yet."); return

    with engine.begin() as con:
        for r in df.itertuples(index=False):
            tags = []
            try:
                tags = json.loads(r.style_tags) if isinstance(r.style_tags, str) else (r.style_tags or [])
            except Exception:
                pass
            opp_form = str(r.opponent_form)
            opp_phase = _clean_phase(r.opponent_phase)
            tp_key = f"{opp_form}|{opp_phase}"
            _upsert_node(con, "TeamPhase", tp_key, {"opponent_form": opp_form, "opponent_phase": opp_phase})
            for tag in tags:
                tag = str(tag)
                _upsert_node(con, "StyleTag", tag)
                _insert_edge_agg_support(
                    con,
                    src=("TeamPhase", tp_key),
                    rel="EXHIBITS",
                    dst=("StyleTag", tag),
                    props={"created_at": str(r.created_at), "weight": 1.0}
                )
    print("[KG] Style tags ingested.")

# ---------- PRIOR with EVIDENCE ----------
def kg_prior_with_evidence(
    engine,
    opponent_form: str,
    opponent_phase: str,
    *,
    provider: Optional[str] = None,
    limit: int = 12,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Returns (prior_map, evidence_rows) from kg_edges with rel='COUNTERS'.
    Tries in order:
      1) exact phase + provider-preferred (provider or NULL)
      2) exact phase, any provider
      3) any phase, provider-preferred
      4) any phase, any provider
    """
    of = str(opponent_form).strip()
    # normalize phase a bit (lowercase, unify separators)
    ph = str(opponent_phase).strip().lower().replace(" ", "-").replace("_", "-")
    prov = (None if provider in (None, "", "None") else str(provider).strip())

    def _run(sql, params):
        with engine.begin() as con:
            return pd.read_sql(text(sql), con, params=params)

    # base SELECT
    base = """
        SELECT
            src_key AS candidate_form,
            dst_key AS opponent_form,
            COALESCE((props->>'phase'), '') AS phase,
            (props->>'provider') AS provider,
            COALESCE((props->>'support')::float, 0.0) AS support
        FROM kg_edges
        WHERE rel = 'COUNTERS'
          AND dst_key = :opp_form
    """

    # 1) exact phase, prefer provider (provider=prov OR NULL), order provider match first
    q1 = base + """
      AND LOWER(REPLACE(COALESCE(props->>'phase',''), ' ', '-')) = :phase
      AND (:prov IS NULL OR (COALESCE(props->>'provider','') = :prov OR props->>'provider' IS NULL))
      ORDER BY
        CASE WHEN :prov IS NOT NULL AND COALESCE(props->>'provider','') = :prov THEN 0
             WHEN props->>'provider' IS NULL THEN 1
             ELSE 2 END,
        support DESC
      LIMIT :lim
    """
    # 2) exact phase, any provider
    q2 = base + """
      AND LOWER(REPLACE(COALESCE(props->>'phase',''), ' ', '-')) = :phase
      ORDER BY support DESC
      LIMIT :lim
    """
    # 3) any phase, prefer provider
    q3 = base + """
      AND (:prov IS NULL OR (COALESCE(props->>'provider','') = :prov OR props->>'provider' IS NULL))
      ORDER BY
        CASE WHEN :prov IS NOT NULL AND COALESCE(props->>'provider','') = :prov THEN 0
             WHEN props->>'provider' IS NULL THEN 1
             ELSE 2 END,
        support DESC
      LIMIT :lim
    """
    # 4) any phase, any provider
    q4 = base + """
      ORDER BY support DESC
      LIMIT :lim
    """

    params = {"opp_form": of, "phase": ph, "prov": prov, "lim": int(limit)}

    df = _run(q1, params)
    if df.empty:
        df = _run(q2, params)
    if df.empty:
        df = _run(q3, params)
    if df.empty:
        df = _run(q4, params)

    if df.empty:
        return {}, []

    # Build prior by normalizing support
    supp = df["support"].astype(float).clip(lower=0.0).to_numpy()
    ssum = float(supp.sum())
    if ssum <= 1e-9:
        prior_map = {f: 1.0 / len(df) for f in df["candidate_form"]}
    else:
        prior_map = {f: float(s / ssum) for f, s in zip(df["candidate_form"], supp)}

    evidence = [{
        "source": "KG.COUNTERS",
        "candidate_form": str(r["candidate_form"]),
        "provider": (r["provider"] if pd.notna(r["provider"]) and r["provider"] != "" else None),
        "phase": str(r["phase"]) if pd.notna(r["phase"]) else "",
        "support": float(r["support"]),
    } for _, r in df.iterrows()]

    return prior_map, evidence