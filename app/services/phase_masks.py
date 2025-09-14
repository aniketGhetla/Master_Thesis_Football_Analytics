# app/services/phase_masks.py
# ------------------------------------------------------------
# Builds phase eligibility/mask columns on sportec_normalized_wide_temp:
#   set_piece_temp, transition_temp, drop3_home_temp, drop3_away_temp,
#   eligible_home_form_temp, eligible_away_form_temp
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
FPS = 25
TRANSITION_SECONDS = 3

# Accept both lowercase and CamelCase variants commonly seen
SET_PIECES = {
    "set piece","corner","freekick","throwin","penalty","goalkick","kickoff","offside","drop_ball",
    "cornerkick","freekick","penalty"
}

# Target table/columns (Sportec only, temp naming)
TABLE = "sportec_normalized_wide_temp"
JOIN_PERIOD_COL = "period_id"
PRED_HOME_COL = "phase_home_pred_temp"
PRED_AWAY_COL = "phase_away_pred_temp"

# Output column names (all *_temp)
SET_PIECE_COL = "set_piece_temp"
TRANSITION_COL = "transition_temp"
DROP3_HOME_COL = "drop3_home_temp"
DROP3_AWAY_COL = "drop3_away_temp"
ELIG_HOME_COL = "eligible_home_form_temp"
ELIG_AWAY_COL = "eligible_away_form_temp"

def _table_cols(engine, table):
    return pd.read_sql(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        engine, params=(table,)
    )["column_name"].tolist()

def _load_sportec(engine):
    cols = _table_cols(engine, TABLE)

    sel = []
    # period & frame
    if JOIN_PERIOD_COL in cols: sel.append(f"{JOIN_PERIOD_COL} AS period_id")
    if "frame" in cols:         sel.append("frame")
    # event label for set-piece detection
    # try both 'original_event' and 'event_type' if present
    if "original_event" in cols: sel.append("original_event")
    elif "event_type" in cols:   sel.append("event_type AS original_event")
    # possession (home/away)
    if "team_possession" in cols: sel.append("team_possession")
    # phase predictions
    if PRED_HOME_COL in cols: sel.append(PRED_HOME_COL)
    if PRED_AWAY_COL in cols: sel.append(PRED_AWAY_COL)
    # filter to alive if available
    where = "WHERE ball_status = 'alive'" if "ball_status" in cols else ""

    if not sel:
        return pd.DataFrame(), TABLE, JOIN_PERIOD_COL

    q = f"""
      SELECT {", ".join(sel)}
      FROM {TABLE}
      {where}
      ORDER BY {JOIN_PERIOD_COL}, frame;
    """
    return pd.read_sql(q, engine), TABLE, JOIN_PERIOD_COL

def _set_piece_mask(df):
    if "original_event" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    ev = df["original_event"].astype(str).str.lower()
    return ev.isin(SET_PIECES).fillna(False).to_numpy(dtype=bool)

def _transition_mask(df):
    n = len(df); pad = int(TRANSITION_SECONDS * FPS)
    if "team_possession" not in df.columns:
        return np.zeros(n, dtype=bool)

    s = df["team_possession"].astype("string").str.lower()
    changes = (s.ne(s.shift(1)) & s.notna() & s.shift(1).notna()).fillna(False).to_numpy(dtype=bool)

    trans = np.zeros(n, dtype=bool)
    idxs = np.flatnonzero(changes)
    for i in idxs:
        trans[i:min(i + pad, n - 1) + 1] = True
    return trans

def _drop_first_3s_mask(phase_series):
    if phase_series is None:
        return np.zeros(0, dtype=bool)
    phase = phase_series.astype("string")
    starts = (phase.ne(phase.shift(1)) & phase.notna()).fillna(False).to_numpy(dtype=bool)
    pad = int(3 * FPS)

    dropped = np.zeros(len(phase), dtype=bool)
    idxs = np.flatnonzero(starts)
    for i in idxs:
        dropped[i:min(i + pad, len(phase) - 1) + 1] = True
    return dropped

def _write_stage(engine, table, join_period_col, df_out):
    cols_sql = {
        SET_PIECE_COL: "BOOLEAN",
        TRANSITION_COL: "BOOLEAN",
        DROP3_HOME_COL: "BOOLEAN",
        DROP3_AWAY_COL: "BOOLEAN",
        ELIG_HOME_COL: "BOOLEAN",
        ELIG_AWAY_COL: "BOOLEAN",
    }
    stage = f"{table}_phase_masks_stage_temp"
    with engine.begin() as conn:
        # Ensure target columns exist on main table
        for name, dtype in cols_sql.items():
            conn.execute(text(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {dtype}'))
        # Create/replace stage table
        conn.execute(text(f"DROP TABLE IF EXISTS {stage}"))
        df_out.to_sql(stage, con=engine, if_exists="replace", index=False)
        # UPDATE main table from stage
        set_clause = ", ".join([f"{c}=s.{c}" for c in df_out.columns if c not in ("period_id","frame")])
        conn.execute(text(f"""
            UPDATE {table} t
               SET {set_clause}
              FROM {stage} s
             WHERE t.frame = s.frame AND t.{join_period_col} = s.period_id;
        """))
        conn.execute(text(f"DROP TABLE IF EXISTS {stage}"))

def run_phase_masks() -> int:
    """
    Builds masks and eligibility flags and writes them to TABLE.
    Returns number of rows updated (rows in stage).
    """
    engine = create_engine(DB_URL)
    df, table, join_period_col = _load_sportec(engine)
    if df.empty:
        print("sportec: no rows to mask")
        return 0

    # Build masks (numpy bool arrays)
    setp = _set_piece_mask(df)
    trans = _transition_mask(df)

    has_home = (df.get(PRED_HOME_COL) is not None) and df[PRED_HOME_COL].notna()
    has_away = (df.get(PRED_AWAY_COL) is not None) and df[PRED_AWAY_COL].notna()

    dropH = _drop_first_3s_mask(df[PRED_HOME_COL]) if PRED_HOME_COL in df.columns else np.zeros(len(df), dtype=bool)
    dropA = _drop_first_3s_mask(df[PRED_AWAY_COL]) if PRED_AWAY_COL in df.columns else np.zeros(len(df), dtype=bool)

    base = np.logical_and(~setp, ~trans)
    eligH = np.logical_and.reduce([base, has_home.to_numpy(dtype=bool) if hasattr(has_home, "to_numpy") else np.zeros(len(df),dtype=bool), ~dropH])
    eligA = np.logical_and.reduce([base, has_away.to_numpy(dtype=bool) if hasattr(has_away, "to_numpy") else np.zeros(len(df),dtype=bool), ~dropA])

    out = pd.DataFrame({
        "period_id": df["period_id"].astype("Int64"),
        "frame": df["frame"].astype("Int64"),
        SET_PIECE_COL: setp,
        TRANSITION_COL: trans,
        DROP3_HOME_COL: dropH,
        DROP3_AWAY_COL: dropA,
        ELIG_HOME_COL: eligH,
        ELIG_AWAY_COL: eligA,
    }).dropna(subset=["period_id","frame"]).astype({"period_id":"int64","frame":"int64"})

    _write_stage(engine, table, join_period_col, out)
    print(f"✅ sportec: wrote masks + eligibility to {table} ({len(out)} rows)")
    return len(out)

if __name__ == "__main__":
    run_phase_masks()
