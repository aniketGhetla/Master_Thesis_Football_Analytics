# app/services/image_generation.py
# ------------------------------------------------------------
# Image generation from normalized Sportec tracking:
# - Reads sportec_normalized_wide_temp
# - Masks set-pieces and early-possession-transition frames
# - Requires a full FRAME_WINDOW window of eligible frames
# - Saves 3-channel images (home/away/ball) as .npz to OUTPUT_DIR
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

# === CONFIG ===
DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
TRACKING_TABLE = "sportec_normalized_wide_temp"   # source created by preprocessing
FRAME_WINDOW = 13                                  # must be odd
OUTPUT_DIR = "phase_images_databall_temp"
IMAGE_WIDTH, IMAGE_HEIGHT = 105, 68                # assumes coords already normalized [0,1]
FPS = 25                                           # adjust if your Sportec is not 25 Hz
MAX_SAMPLES = None                                 # set to an int to cap outputs for testing

# Treat these as set-pieces (tweak to match your taxonomy)
SET_PIECES = {
    "set piece", "corner", "free_kick", "throw_in", "penalty",
    "goal_kick", "kick_off", "offside", "drop_ball"
}
TRANSITION_SECONDS_AFTER_POSSESSION_CHANGE = 3     # can bump to 6 if preferred


# === UTILS ===
def normalize_coords(x, y):
    """Assumes x,y already normalized to [0,1] (from preprocessing)."""
    try:
        if pd.isna(x) or pd.isna(y):
            return None, None
        x_pixel = int(np.clip(round(x * IMAGE_WIDTH), 0, IMAGE_WIDTH - 1))
        y_pixel = int(np.clip(round(y * IMAGE_HEIGHT), 0, IMAGE_HEIGHT - 1))
        return x_pixel, y_pixel
    except Exception:
        return None, None


def get_home_away_ids(columns):
    """Infer jersey id prefixes present in wide columns (home_*, away_*)."""
    home_ids, away_ids = set(), set()
    for col in columns:
        if col.endswith("_x"):
            if col.startswith("home_"):
                home_ids.add(col.replace("_x", ""))
            elif col.startswith("away_"):
                away_ids.add(col.replace("_x", ""))
    return sorted(home_ids), sorted(away_ids)


def build_image(frames: pd.DataFrame, home_ids, away_ids):
    """Build a 3-channel image from a window of frames."""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    for _, row in frames.iterrows():
        # Ball (blue)
        bx, by = normalize_coords(row.get("ball_x"), row.get("ball_y"))
        if bx is not None and by is not None:
            image[by, bx, 2] = 1.0

        # Players (home red, away green)
        for pid in home_ids + away_ids:
            x_key, y_key = f"{pid}_x", f"{pid}_y"
            if x_key not in row or y_key not in row:
                continue
            x, y = row[x_key], row[y_key]
            if pd.isna(x) or pd.isna(y):
                continue
            x_pix, y_pix = normalize_coords(x, y)
            if x_pix is None or y_pix is None:
                continue
            chan = 0 if pid in home_ids else 1
            image[y_pix, x_pix, chan] = 1.0
    return image


def mark_set_pieces(df: pd.DataFrame) -> pd.Series:
    """
    True on frames that belong to set-pieces, using an event label column.
    Tries 'original_event', then 'event_type'; otherwise returns all False.
    """
    for col in ["original_event", "event_type"]:
        if col in df.columns:
            ev = df[col].astype(str).str.lower()
            return ev.isin(SET_PIECES)
    return pd.Series(False, index=df.index, dtype=bool)


def mark_transitions_from_possession(df: pd.DataFrame, fps: int) -> pd.Series:
    """
    True for the first N seconds after a change in team possession.
    Uses 'team_possession' with values like 'home'/'away'.
    Falls back to all False if column missing.
    """
    if "team_possession" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    series = df["team_possession"].astype("string").str.lower()
    changes = series.ne(series.shift(1)) & series.notna() & series.shift(1).notna()
    idxs = np.flatnonzero(changes.to_numpy())
    pad = int(TRANSITION_SECONDS_AFTER_POSSESSION_CHANGE * fps)

    is_transition = pd.Series(False, index=df.index, dtype=bool)
    for i in idxs:
        start, end = i, min(i + pad, len(df) - 1)
        is_transition.iloc[start:end + 1] = True
    return is_transition


def ensure_full_window_eligible(eligible_mask: pd.Series, window_radius: int) -> pd.Series:
    """
    Only keep center frames where ALL frames in the window are eligible.
    """
    arr = eligible_mask.values.astype(np.uint8)
    w = window_radius
    if len(arr) < 2 * w + 1:
        return pd.Series(False, index=eligible_mask.index, dtype=bool)
    csum = np.cumsum(np.insert(arr, 0, 0))
    size = 2 * w + 1
    totals = csum[size:] - csum[:-size]
    full_ok = np.zeros_like(arr, dtype=bool)
    full_ok[w:len(arr) - w] = totals == size
    return pd.Series(full_ok, index=eligible_mask.index, dtype=bool)


# === MAIN (callable from Flask) ===
# def generate_images_from_sportec() -> tuple[int, str]:
#     """
#     Generates masked images from normalized Sportec tracking.
#     Returns (num_samples_saved, output_dir).
#     """
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     engine = create_engine(DB_URL)

#     print("📥 Loading Sportec tracking…")
#     # Keep only 'alive' if present (preprocessing table usually retains it)
#     where_clause = "WHERE ball_status = 'alive'"  # safe even if column missing? We'll handle fallback
#     try:
#         df = pd.read_sql(f"SELECT * FROM {TRACKING_TABLE} {where_clause}", engine)
#     except Exception:
#         df = pd.read_sql(f"SELECT * FROM {TRACKING_TABLE}", engine)
#         if "ball_status" in df.columns:
#             df = df[df["ball_status"] == "alive"].copy()

#     # Sort robustly
#     sort_cols = [c for c in ["period_id", "frame"] if c in df.columns]
#     if sort_cols:
#         df = df.sort_values(sort_cols).reset_index(drop=True)

#     # Build masks
#     is_set_piece = mark_set_pieces(df)
#     is_transition = mark_transitions_from_possession(df, FPS)

#     eligible = (~is_set_piece) & (~is_transition)
#     if "in_play" in df.columns:
#         eligible &= df["in_play"].astype(bool)

#     # Require full window eligibility
#     full_window_ok = ensure_full_window_eligible(eligible, FRAME_WINDOW // 2)

#     # Stats
#     print(f"✅ Frames loaded: {len(df)}")
#     print(f"🚫 Set-piece masked: {int(is_set_piece.sum())}")
#     print(f"🚦 Transition masked: {int(is_transition.sum())}")
#     print(f"🪟 Eligible centers with full {FRAME_WINDOW}-frame window: {int(full_window_ok.sum())}")

#     # Player id prefixes (home_*, away_*)
#     home_ids, away_ids = get_home_away_ids(df.columns)
#     print(f"👥 Found {len(home_ids)} home and {len(away_ids)} away players")

#     # Save images
#     indices = np.flatnonzero(full_window_ok.values)
#     if MAX_SAMPLES is not None:
#         indices = indices[:MAX_SAMPLES]

#     samples = 0
#     half_w = FRAME_WINDOW // 2
#     for i in tqdm(indices, desc="Saving images"):
#         window = df.iloc[i - half_w: i + half_w + 1]
#         image = build_image(window, home_ids, away_ids)

#         center = df.iloc[i]
#         period = int(center.get("period_id", 0)) if "period_id" in df.columns else 0
#         frame_id = int(center["frame"]) if "frame" in df.columns else i

#         fname = f"sportec_{period}_{frame_id}.npz"
#         np.savez_compressed(
#             os.path.join(OUTPUT_DIR, fname),
#             image=image,
#             label="unknown_phase",     # fill after ResNet step
#             frame_id=frame_id,
#             period_id=period,
#             set_piece=bool(is_set_piece.iloc[i]),
#             transition=bool(is_transition.iloc[i]),
#             eligible=bool(eligible.iloc[i]),
#         )
#         samples += 1

#     print(f"🎉 {samples} image samples saved to {OUTPUT_DIR}")
#     return samples, OUTPUT_DIR

def generate_images_from_sportec(unique_match_id: str | None = None) -> tuple[int, str]:
    """
    Generates masked images from normalized Sportec tracking.
    If unique_match_id is provided, filters to that run and writes to a subfolder.
    Returns (num_samples_saved, output_dir).
    """
    out_dir = OUTPUT_DIR if not unique_match_id else os.path.join(OUTPUT_DIR, unique_match_id)
    os.makedirs(out_dir, exist_ok=True)
    engine = create_engine(DB_URL)

    # Load data
    try:
        if unique_match_id:
            df = pd.read_sql(
                f"SELECT * FROM {TRACKING_TABLE} WHERE unique_match_id = %(umid)s AND ball_status = 'alive'",
                engine, params={"umid": unique_match_id}
            )
        else:
            df = pd.read_sql(f"SELECT * FROM {TRACKING_TABLE} WHERE ball_status = 'alive'", engine)
    except Exception:
        # fallback if ball_status not present
        if unique_match_id:
            df = pd.read_sql(
                f"SELECT * FROM {TRACKING_TABLE} WHERE unique_match_id = %(umid)s",
                engine, params={"umid": unique_match_id}
            )
        else:
            df = pd.read_sql(f"SELECT * FROM {TRACKING_TABLE}", engine)
        if "ball_status" in df.columns:
            df = df[df["ball_status"] == "alive"].copy()

    sort_cols = [c for c in ["period_id", "frame"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    is_set_piece = mark_set_pieces(df)
    is_transition = mark_transitions_from_possession(df, FPS)
    eligible = (~is_set_piece) & (~is_transition)
    if "in_play" in df.columns:
        eligible &= df["in_play"].astype(bool)
    full_window_ok = ensure_full_window_eligible(eligible, FRAME_WINDOW // 2)

    home_ids, away_ids = get_home_away_ids(df.columns)
    indices = np.flatnonzero(full_window_ok.values)
    if MAX_SAMPLES is not None:
        indices = indices[:MAX_SAMPLES]

    samples = 0
    half_w = FRAME_WINDOW // 2
    for i in tqdm(indices, desc="Saving images"):
        window = df.iloc[i - half_w: i + half_w + 1]
        image = build_image(window, home_ids, away_ids)

        center = df.iloc[i]
        period = int(center.get("period_id", 0)) if "period_id" in df.columns else 0
        frame_id = int(center["frame"]) if "frame" in df.columns else i

        fname = f"sportec_{period}_{frame_id}.npz"
        np.savez_compressed(
            os.path.join(out_dir, fname),
            image=image,
            label="unknown_phase",
            frame_id=frame_id,
            period_id=period,
            set_piece=bool(is_set_piece.iloc[i]),
            transition=bool(is_transition.iloc[i]),
            eligible=bool(eligible.iloc[i]),
            unique_match_id=center.get("unique_match_id", None),
        )
        samples += 1

    print(f"🎉 {samples} image samples saved to {out_dir}")
    return samples, out_dir


# Optional: allow CLI execution
if __name__ == "__main__":
    n, out = generate_images_from_sportec()
    print(f"Saved {n} samples to {out}")
