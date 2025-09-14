# app/services/apply_shaw_ordering.py
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

# Use the temp tables in your pipeline
SRC_TABLE = "formation_templates_temp"
DST_TABLE = "formation_templates_shaw_temp"

def ensure_dst(engine):
    with engine.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {DST_TABLE} (
            provider TEXT,
            side TEXT,
            phase TEXT,
            period_id INT,
            block_id BIGINT,
            block_start_frame INT,
            block_end_frame INT,
            num_frames INT,
            agg_seconds DOUBLE PRECISION,
            role_index INT,              -- Shaw-ordered index (1..k)
            orig_role_index INT,         -- from source table
            mu_x DOUBLE PRECISION,
            mu_y DOUBLE PRECISION,
            cov_xx DOUBLE PRECISION,
            cov_xy DOUBLE PRECISION,
            cov_yy DOUBLE PRECISION
        );
        """))

def nearest_teammate_chain(points: np.ndarray) -> np.ndarray:
    """
    points: (P,2) μ positions (meters), already centered.
    Returns an ordering (indices 0..P-1) by:
      1) start at the 'centre' player (min distance to their 3rd-nearest neighbour),
      2) greedily step to the nearest unvisited teammate,
      3) canonicalize left->right along x.
    """
    P = points.shape[0]
    if P <= 1:
        return np.arange(P)

    # --- 1) choose start: min distance to 3rd-nearest neighbour ---
    # pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)  # ignore self
    # third-nearest distance per player (k=3)
    third_nn = np.sort(D, axis=1)[:, 2] if P >= 4 else np.sort(D, axis=1)[:, min(2, P-1)]
    start = int(np.argmin(third_nn))

    # --- 2) greedy NN walk ---
    used = np.zeros(P, dtype=bool)
    order = [start]
    used[start] = True
    for _ in range(P - 1):
        last = order[-1]
        dist = D[last].copy()
        dist[used] = np.inf
        nxt = int(np.argmin(dist))
        order.append(nxt)
        used[nxt] = True

    order = np.array(order)

    # --- 3) left->right canonicalization along x ---
    # if the chain ends left of where it starts (in x), flip it
    if points[order[-1], 0] < points[order[0], 0]:
        order = order[::-1]
    return order

def run_apply_shaw() -> int:
    """
    Applies Shaw ordering per (provider, side, phase, period_id, block_id)
    from SRC_TABLE to DST_TABLE. Returns number of rows written.
    """
    engine = create_engine(DB_URL)
    ensure_dst(engine)

    src = pd.read_sql(f"SELECT * FROM {SRC_TABLE}", engine)
    if src.empty:
        print(f"No rows in {SRC_TABLE}; nothing to do.")
        return 0

    # truncate destination for a clean write
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {DST_TABLE};"))

    group_cols = [
        "provider", "side", "phase", "period_id", "block_id",
        "block_start_frame", "block_end_frame", "num_frames", "agg_seconds"
    ]

    rows_out = []
    for keys, g in src.groupby(group_cols, sort=False):
        g = g.sort_values("role_index").reset_index(drop=True)

        mus = g[["mu_x", "mu_y"]].to_numpy(dtype=float)
        order = nearest_teammate_chain(mus)

        for new_idx, src_idx in enumerate(order, start=1):
            rows_out.append({
                "provider": keys[0],
                "side": keys[1],
                "phase": keys[2],
                "period_id": int(keys[3]),
                "block_id": int(keys[4]),
                "block_start_frame": int(keys[5]),
                "block_end_frame": int(keys[6]),
                "num_frames": int(keys[7]),
                "agg_seconds": float(keys[8]),
                "role_index": int(new_idx),                              # Shaw order
                "orig_role_index": int(g.loc[src_idx, "role_index"]),    # original
                "mu_x": float(g.loc[src_idx, "mu_x"]),
                "mu_y": float(g.loc[src_idx, "mu_y"]),
                "cov_xx": float(g.loc[src_idx, "cov_xx"]),
                "cov_xy": float(g.loc[src_idx, "cov_xy"]),
                "cov_yy": float(g.loc[src_idx, "cov_yy"]),
            })

    out = pd.DataFrame(rows_out)
    out.to_sql(DST_TABLE, con=engine, if_exists="append", index=False)

    n_blocks = out["block_id"].nunique() if not out.empty else 0
    print(f"✅ Wrote {len(out)} rows into {DST_TABLE} across {n_blocks} blocks")

    # quick sanity check
    with engine.begin() as conn:
        r = conn.execute(text(f"SELECT COUNT(*), COUNT(DISTINCT block_id) FROM {DST_TABLE}")).fetchone()
        print(f"Rows: {r[0]}, Blocks: {r[1]} (should match source blocks)")

    return len(out)

# CLI
if __name__ == "__main__":
    run_apply_shaw()
