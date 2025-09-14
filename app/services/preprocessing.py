# app/services/preprocessing.py
# ------------------------------------------------------------
# Minimal preprocessing:
# - Load whole INPUT_TABLE
# - Normalize player/ball *_x, *_y to [0,1] using 105 x 68
# - Save to OUTPUT_TABLE (append)
# ------------------------------------------------------------

import pandas as pd
from sqlalchemy import create_engine

# === CONFIG ===
DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
INPUT_TABLE = "sportec_databall_merged_temp"   # or "sportec_databall_merged"
OUTPUT_TABLE = "sportec_normalized_wide_temp"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def normalize_coords_df(df: pd.DataFrame, x_cols, y_cols, length: float, width: float) -> pd.DataFrame:
    """
    Normalize coordinates:
      X: [-L/2, L/2] -> [0,1] via (x + L/2) / L
      Y: [-W/2, W/2] -> [0,1] via (y + W/2) / W
    """
    for x_col, y_col in zip(x_cols, y_cols):
        df[x_col] = (df[x_col] + (length / 2.0)) / length
        df[y_col] = (df[y_col] + (width / 2.0)) / width
    return df


# def run_normalization() -> str:
#     """
#     Main function to be called from Flask.
#     Loads INPUT_TABLE, normalizes coords, writes OUTPUT_TABLE.
#     Returns the name of the output table.
#     """
#     engine = create_engine(DB_URL)

#     print("🔄 Loading tracking data...")
#     df = pd.read_sql(f"SELECT * FROM {INPUT_TABLE}", engine)

#     # Find coordinate columns
#     x_cols = [c for c in df.columns if c.endswith("_x") and (c.startswith("home_") or c.startswith("away_") or c == "ball_x")]
#     y_cols = [c for c in df.columns if c.endswith("_y") and (c.startswith("home_") or c.startswith("away_") or c == "ball_y")]

#     print("⚙️ Normalizing player and ball positions...")
#     df = normalize_coords_df(df, x_cols, y_cols, PITCH_LENGTH, PITCH_WIDTH)

#     print(f"📤 Saving normalized data to table: {OUTPUT_TABLE}")
#     df.to_sql(OUTPUT_TABLE, engine, index=False, if_exists="append")

#     print(f"✅ Done! Normalized data available in table: {OUTPUT_TABLE}")
#     return OUTPUT_TABLE

def run_normalization(unique_match_id: str | None = None) -> str:
    """
    If unique_match_id is provided, normalize ONLY that run from INPUT_TABLE.
    Otherwise, normalize all rows (previous behavior).
    Returns the output table name.
    """
    engine = create_engine(DB_URL)

    if unique_match_id:
        print(f"🔄 Loading tracking data for unique_match_id={unique_match_id} ...")
        df = pd.read_sql(
            f"SELECT * FROM {INPUT_TABLE} WHERE unique_match_id = %(umid)s",
            engine, params={"umid": unique_match_id}
        )
    else:
        print("🔄 Loading tracking data (ALL rows)...")
        df = pd.read_sql(f"SELECT * FROM {INPUT_TABLE}", engine)

    x_cols = [c for c in df.columns if c.endswith("_x") and (c.startswith("home_") or c.startswith("away_") or c == "ball_x")]
    y_cols = [c for c in df.columns if c.endswith("_y") and (c.startswith("home_") or c.startswith("away_") or c == "ball_y")]

    print("⚙️ Normalizing player and ball positions...")
    df = normalize_coords_df(df, x_cols, y_cols, PITCH_LENGTH, PITCH_WIDTH)

    print(f"📤 Saving normalized data to table: {OUTPUT_TABLE}")
    df.to_sql(OUTPUT_TABLE, engine, index=False, if_exists="append")
    print(f"✅ Done! Normalized data available in table: {OUTPUT_TABLE}")
    return OUTPUT_TABLE



# For CLI use (python app/services/preprocessing.py)
if __name__ == "__main__":
    run_normalization()
