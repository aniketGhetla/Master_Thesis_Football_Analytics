# app/services/ingestion.py
# ------------------------------------------------------------
# Ingestion service: wraps your three scripts into functions
# 1) load_tracking_and_meta(...)  -> writes sportec_databall_wide_temp  (APPEND)
# 2) load_event_data(...)         -> writes sportec_event_new_temp      (APPEND)
# 3) merge_tracking_and_events()  -> writes sportec_databall_merged_temp(APPEND)
# 4) process_upload(...)          -> orchestrator used by Flask route
# ------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import uuid
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from databallpy import get_game
from kloppy import sportec as kloppy_sportec
from kloppy.domain import Provider


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DBConfig:
    user: str = "postgres"
    password: str = "123"
    host: str = "localhost"
    port: str = "5432"
    database: str = "Football"

    def url(self) -> str:
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class TableNames:
    tracking_wide: str = "sportec_databall_wide_temp"
    event: str = "sportec_event_new_temp"
    merged: str = "sportec_databall_merged_temp"


# -----------------------------
# DB Helpers
# -----------------------------
def _get_engine(db: DBConfig) -> Engine:
    return create_engine(db.url())


def _pg_ident(name: str) -> str:
    """
    Safely quote a Postgres identifier.
    """
    return '"' + name.replace('"', '""') + '"'


def _get_existing_columns(engine: Engine, table: str, schema: str = "public") -> set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"schema": schema, "table": table}).fetchall()
    return {r[0] for r in rows}


def _map_dtype_to_pg(series: pd.Series) -> str:
    """
    Map pandas dtype to a Postgres column type.
    """
    dt = series.dtype

    if pd.api.types.is_float_dtype(dt):
        return "DOUBLE PRECISION"
    if pd.api.types.is_integer_dtype(dt):
        # large frames can exceed INT; go BIGINT
        return "BIGINT"
    if pd.api.types.is_bool_dtype(dt):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "TIMESTAMPTZ"
    if pd.api.types.is_timedelta64_dtype(dt):
        return "INTERVAL"
    # fallbacks: object, category, string, mixed -> TEXT
    return "TEXT"


def _ensure_table_exists(engine: Engine, table: str, df: pd.DataFrame) -> None:
    """
    Ensure the table exists. If not, create an empty table with the df schema.
    Pandas to_sql(..., if_exists='append') will create if missing, but we make it explicit.
    """
    existing = _get_existing_columns(engine, table)
    if existing:
        return
    # Create empty table with df's columns (0 rows)
    df.head(0).to_sql(table, engine, index=False, if_exists="append")


def _ensure_table_schema(engine: Engine, table: str, df: pd.DataFrame, schema: str = "public") -> None:
    """
    Ensure the table has all columns appearing in df. If any are missing,
    ALTER TABLE ADD COLUMN with a reasonable type.
    """
    _ensure_table_exists(engine, table, df)
    existing = _get_existing_columns(engine, table, schema=schema)
    new_cols = [c for c in df.columns if c not in existing]

    if not new_cols:
        return

    # Build and run ALTERs
    alters = []
    for col in new_cols:
        col_type = _map_dtype_to_pg(df[col])
        alters.append(f"ALTER TABLE {_pg_ident(schema)}.{_pg_ident(table)} ADD COLUMN IF NOT EXISTS {_pg_ident(col)} {col_type};")

    ddl = "\n".join(alters)
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"🔧 Schema extended on '{table}': added {len(new_cols)} columns")


# -----------------------------
# Step 1: DataBallPy tracking + meta (+event for sync)
# -----------------------------
def load_tracking_and_meta(
    tracking_file: str,
    meta_file: str,
    event_file: str,
    db: Optional[DBConfig] = None,
    tables: Optional[TableNames] = None,
    sync_offset_seconds: float = 1.0,
    run_id: str = "",
) -> Tuple[str, str]:
    """
    Loads Sportec tracking via DataBallPy, synchronizes with events,
    filters to ball_status == 'alive', enriches with match metadata,
    and APPENDs to 'sportec_databall_wide_temp'.

    Returns:
        (match_id, unique_match_id)
    """
    db = db or DBConfig()
    tables = tables or TableNames()

    # Build game
    game = get_game(
        tracking_data_loc=tracking_file,
        tracking_metadata_loc=meta_file,
        tracking_data_provider="sportec",
        event_data_loc=event_file,
        event_metadata_loc=meta_file,
        event_data_provider="sportec",
    )

    # Synchronize tracking & events
    game.synchronise_tracking_and_event_data(
        n_batches="smart",
        offset=sync_offset_seconds,
        verbose=True
    )

    # Wide tracking dataframe (alive only)
    # Note: ._data is deprecated; acceptable for now. We'll migrate to public APIs later.
    tracking_wide_df = pd.DataFrame(game.tracking_data._data)
    if "ball_status" in tracking_wide_df.columns:
        tracking_wide_df = tracking_wide_df[tracking_wide_df["ball_status"] == "alive"].copy()

    # Event DF (for optional enrichment via event_id)
    event_df = pd.DataFrame(game.event_data._data)
    if "event_id" in tracking_wide_df.columns and "event_id" in event_df.columns:
        enriched_event_df = event_df.drop_duplicates(subset=["event_id"])
        tracking_wide_df = tracking_wide_df.merge(
            enriched_event_df, on="event_id", how="left", suffixes=("", "_event")
        )

    # Match metadata
    meta = {
        "match_id": game.name,
        "home_team": game.home_team_name,
        "away_team": game.away_team_name,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "pitch_length": game.pitch_dimensions[0],
        "pitch_width": game.pitch_dimensions[1],
    }
    for k, v in meta.items():
        tracking_wide_df[k] = v

    # Per-run unique id
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
    unique_match_id = f"{meta['match_id']}_{run_id}"
    tracking_wide_df["unique_match_id"] = unique_match_id

    # Write to Postgres (APPEND) with schema guard
    engine = _get_engine(db)
    _ensure_table_schema(engine, tables.tracking_wide, tracking_wide_df)
    tracking_wide_df.to_sql(tables.tracking_wide, engine, index=False, if_exists="append")
    print(f"✅ Wrote {len(tracking_wide_df)} rows to '{tables.tracking_wide}'")

    return meta["match_id"], unique_match_id


# -----------------------------
# Step 2: Kloppy events → PFF → flip → write
# -----------------------------
def load_event_data(
    event_file: str,
    meta_file: str,
    match_id: str,
    unique_match_id: str,
    db: Optional[DBConfig] = None,
    tables: Optional[TableNames] = None,
) -> None:
    """
    Loads Sportec events via Kloppy, transforms to PFF coordinates,
    flips signs to align with DataBallPy orientation, adds match_id + unique_match_id,
    and APPENDs to 'sportec_event_new_temp'.
    """
    db = db or DBConfig()
    tables = tables or TableNames()

    print(f"📂 Loading event dataset via Kloppy: {event_file}")
    event_dataset = kloppy_sportec.load(event_data=event_file, meta_data=meta_file)
    event_dataset = event_dataset.transform(to_coordinate_system=Provider.PFF)

    df = event_dataset.to_pandas()
    df["match_id"] = match_id
    df["unique_match_id"] = unique_match_id

    # Flip coordinates to align with DataBallPy's pitch orientation
    for col in ["coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y"]:
        if col in df.columns:
            df[col] = -1 * df[col]

    engine = _get_engine(db)
    _ensure_table_schema(engine, tables.event, df)
    df.to_sql(tables.event, engine, if_exists="append", index=False)
    print(f"✅ Wrote {len(df)} rows to '{tables.event}'")


# -----------------------------
# Step 3: Merge tracking & events into final table
# -----------------------------
def merge_tracking_and_events(
    db: Optional[DBConfig] = None,
    tables: Optional[TableNames] = None,
    unique_match_id: str = "",
) -> int:
    """
    Merges the CURRENT RUN ONLY:
    'sportec_databall_wide_temp' + 'sportec_event_new_temp' filtered by unique_match_id,
    using cleaned event IDs, and APPENDs to 'sportec_databall_merged_temp'.

    Returns number of merged rows.
    """
    db = db or DBConfig()
    tables = tables or TableNames()
    engine = _get_engine(db)

    if not unique_match_id:
        raise ValueError("unique_match_id is required to merge a single run.")

    print("📥 Loading tables for merge (filtered by unique_match_id)...")
    df_wide = pd.read_sql_query(
        f"SELECT * FROM {tables.tracking_wide} WHERE unique_match_id = %(umid)s",
        engine, params={"umid": unique_match_id}
    )
    df_event = pd.read_sql_query(
        f"SELECT * FROM {tables.event} WHERE unique_match_id = %(umid)s",
        engine, params={"umid": unique_match_id}
    )

    # Keep only alive events if column present
    if "ball_state" in df_event.columns:
        df_event_alive = df_event[df_event["ball_state"] == "alive"].copy()
    else:
        df_event_alive = df_event.copy()

    # Clean event IDs
    if "original_event_id" in df_wide.columns:
        df_wide["original_event_id_clean"] = (
            df_wide["original_event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
    else:
        df_wide["original_event_id_clean"] = pd.NA

    if "event_id" in df_event_alive.columns:
        df_event_alive["event_id_clean"] = df_event_alive["event_id"].astype(str).str.extract(r"^(\d+)", expand=False)
    else:
        df_event_alive["event_id_clean"] = pd.NA

    # Deduplicate on cleaned ID
    df_event_alive = df_event_alive.drop_duplicates(subset=["event_id_clean"])

    # Select desired columns from events
    keep_cols = [
        "event_id_clean", "event_type", "team_id", "player_id",
        "coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y",
        "receiver_player_id", "result", "success"
    ]
    keep_cols = [c for c in keep_cols if c in df_event_alive.columns]
    df_event_alive = df_event_alive[keep_cols]

    # Merge
    merged_df = df_wide.merge(
        df_event_alive,
        left_on="original_event_id_clean",
        right_on="event_id_clean",
        suffixes=('', '_event'),
        how="left"
    )
    for c in ["original_event_id_clean", "event_id_clean"]:
        if c in merged_df.columns:
            merged_df.drop(columns=[c], inplace=True)

    # Tag with unique_match_id (safety)
    merged_df["unique_match_id"] = unique_match_id

    # Write to Postgres (APPEND) with schema guard
    _ensure_table_schema(engine, tables.merged, merged_df)
    merged_df.to_sql(tables.merged, engine, if_exists="append", index=False)
    print(f"✅ Merge complete: wrote {len(merged_df)} rows to '{tables.merged}'")
    return len(merged_df)


# -----------------------------
# Orchestrator
# -----------------------------
def process_upload(
    tracking_file: str,
    event_file: str,
    meta_file: str,
    db: Optional[DBConfig] = None,
    tables: Optional[TableNames] = None,
    sync_offset_seconds: float = 1.0,
) -> str:
    """
    Runs:
      1) load_tracking_and_meta
      2) load_event_data
      3) merge_tracking_and_events

    Returns: unique_match_id (string) for UI confirmation.
    """
    db = db or DBConfig()
    tables = tables or TableNames()

    run_id = uuid.uuid4().hex[:8]  # per-run suffix

    match_id, unique_match_id = load_tracking_and_meta(
        tracking_file=tracking_file,
        meta_file=meta_file,
        event_file=event_file,
        db=db,
        tables=tables,
        sync_offset_seconds=sync_offset_seconds,
        run_id=run_id,
    )

    load_event_data(
        event_file=event_file,
        meta_file=meta_file,
        match_id=match_id,
        unique_match_id=unique_match_id,
        db=db,
        tables=tables,
    )

    merge_tracking_and_events(
        db=db,
        tables=tables,
        unique_match_id=unique_match_id,
    )

    print(f"🎉 Ingestion finished for unique_match_id={unique_match_id}")
    return unique_match_id
