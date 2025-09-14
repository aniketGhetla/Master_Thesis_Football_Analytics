# app/services/phase_inference.py
# ------------------------------------------------------------
# Phase-of-play inference:
# - Loads .npz images from phase_images_databall_temp[/<unique_match_id>]
# - Runs trained ResNetMultiTaskFootball
# - Writes predictions into sportec_normalized_wide_temp:
#     phase_home_pred_temp, phase_away_pred_temp,
#     phase_home_conf_temp, phase_away_conf_temp
# ------------------------------------------------------------

import os, re, glob, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sqlalchemy import create_engine, text

# ===== CONFIG (keep aligned with your setup) =====
DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

IMAGES_DIR = "phase_images_databall_temp"     # parent folder; per-run subfolders allowed
MODEL_PATH = "C:\\Users\\Harshita\\Documents\\FootBallProject\\app\\services\\models\\resnet_multitask_model.pth"
ENCODERS_PATH = "C:\\Users\\Harshita\\Documents\\FootBallProject\\app\\services\\models\\label_encoders.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
NUM_WORKERS = 0

# One-line switch for testing: set to a small int; use None for all files
MAX_FILES = None

# Must match training
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                     # np.float32 [0..1] -> tensor
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# filename pattern: provider_period_frame.npz (provider in {metrica,sportec})
FNAME_RE = re.compile(r"^(metrica|sportec)_(\d+)_(\d+)\.npz$", re.IGNORECASE)

# Table/column targets
SPORTec_TABLE = "sportec_normalized_wide_temp"   # <- write predictions here
JOIN_PERIOD_COL = "period_id"                    # column on normalized table for join

# Output column names (your requested *_temp naming)
PH_HOME_COL  = "phase_home_pred_temp"
PH_AWAY_COL  = "phase_away_pred_temp"
CF_HOME_COL  = "phase_home_conf_temp"
CF_AWAY_COL  = "phase_away_conf_temp"

SAVE_CONFIDENCE = True  # you asked to store *_conf_temp as well


# ===== Your model class import (as in your script) =====
from app.services.cnn_php_model import ResNetMultiTaskFootball
  # keep this path


# ===== Dataset =====
class NPZSingleFolder(Dataset):
    def __init__(self, folder: str, transform, max_files=None):
        files = []
        for p in glob.glob(os.path.join(folder, "*.npz")):
            name = os.path.basename(p)
            if FNAME_RE.match(name):
                files.append(p)
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise FileNotFoundError(f"No matching *.npz under {folder}. Expected 'provider_period_frame.npz'.")
        self.files = files
        self.t = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        m = FNAME_RE.match(os.path.basename(path))
        provider = m.group(1).lower()
        period = int(m.group(2))
        frame = int(m.group(3))
        with np.load(path) as npz:
            img = npz["image"].astype(np.float32)  # already 0..1
        x = self.t(img)  # CxHxW
        return x, {"provider": provider, "period": period, "frame": frame, "path": path}


# ===== Model / Encoders =====
def _load_model_and_encoders():
    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    model = ResNetMultiTaskFootball(
        len(enc["home_phase"].classes_),
        len(enc["away_phase"].classes_),
        len(enc["home_form"].classes_),
        len(enc["away_form"].classes_),
    )
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model, enc


def _decode_indices(indices, encoder):
    return encoder.inverse_transform(indices.astype(int))


# ===== DB writer =====
def _write_predictions_to_db(engine, df_preds: pd.DataFrame):
    """
    df_preds columns: period_id, frame, PH_HOME_COL, PH_AWAY_COL, (optional conf cols)
    Updates SPORTec_TABLE in-place (UPSERT by period_id + frame).
    """
    # Ensure target columns exist
    with engine.begin() as conn:
        conn.execute(text(f'ALTER TABLE {SPORTec_TABLE} ADD COLUMN IF NOT EXISTS {PH_HOME_COL} TEXT'))
        conn.execute(text(f'ALTER TABLE {SPORTec_TABLE} ADD COLUMN IF NOT EXISTS {PH_AWAY_COL} TEXT'))
        if SAVE_CONFIDENCE:
            conn.execute(text(f'ALTER TABLE {SPORTec_TABLE} ADD COLUMN IF NOT EXISTS {CF_HOME_COL} DOUBLE PRECISION'))
            conn.execute(text(f'ALTER TABLE {SPORTec_TABLE} ADD COLUMN IF NOT EXISTS {CF_AWAY_COL} DOUBLE PRECISION'))

        stage = f"{SPORTec_TABLE}_phase_stage_temp"
        # Create/replace stage table with predictions
        df_preds.to_sql(stage, con=engine, if_exists="replace", index=False)

        set_clause = f"""
            {PH_HOME_COL} = s.{PH_HOME_COL},
            {PH_AWAY_COL} = s.{PH_AWAY_COL}
        """
        if SAVE_CONFIDENCE:
            set_clause += f""",
            {CF_HOME_COL} = s.{CF_HOME_COL},
            {CF_AWAY_COL} = s.{CF_AWAY_COL}
            """

        q = f"""
            UPDATE {SPORTec_TABLE} t
            SET {set_clause}
            FROM {stage} s
            WHERE t.frame = s.frame
              AND t.{JOIN_PERIOD_COL} = s.period_id;
        """
        conn.execute(text(q))
        conn.execute(text(f"DROP TABLE IF EXISTS {stage}"))


# ===== Public entry point (call from Flask) =====
def run_phase_inference(unique_match_id: str | None = None) -> tuple[int, str]:
    """
    Runs phase inference on images:
      - If unique_match_id is given, reads from IMAGES_DIR/<umid>
      - Else, reads from IMAGES_DIR
    Writes predictions to SPORTec_TABLE with *_temp columns.
    Returns (num_rows_written, folder_used).
    """
    folder = IMAGES_DIR if unique_match_id is None else os.path.join(IMAGES_DIR, unique_match_id)
    print(f"🖼️  Reading NPZ images from: {folder}")

    model, enc = _load_model_and_encoders()
    ds = NPZSingleFolder(folder, TRANSFORM, max_files=MAX_FILES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    rows = []
    with torch.no_grad():
        for batch, meta in tqdm(dl, desc=f"Phase inference (max_files={MAX_FILES})"):
            batch = batch.to(DEVICE, non_blocking=True)
            out_hp, out_ap, _, _ = model(batch)

            hp_prob = F.softmax(out_hp, dim=1).cpu().numpy()
            ap_prob = F.softmax(out_ap, dim=1).cpu().numpy()
            hp_idx = hp_prob.argmax(1)
            ap_idx = ap_prob.argmax(1)

            hp_lbl = _decode_indices(hp_idx, enc["home_phase"])
            ap_lbl = _decode_indices(ap_idx, enc["away_phase"])

            for i in range(len(meta["frame"])):
                row = {
                    "period_id": int(meta["period"][i]),
                    "frame":     int(meta["frame"][i]),
                    PH_HOME_COL: str(hp_lbl[i]),
                    PH_AWAY_COL: str(ap_lbl[i]),
                }
                if SAVE_CONFIDENCE:
                    row[CF_HOME_COL] = float(hp_prob[i, hp_idx[i]])
                    row[CF_AWAY_COL] = float(ap_prob[i, ap_idx[i]])
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No predictions produced. Check folder and filenames.")
        return 0, folder

    # Optional audit CSV per-run
    csv_name = "phase_predictions_SAMPLE.csv" if MAX_FILES else "phase_predictions_all.csv"
    audit_csv = os.path.join(folder, csv_name)
    df.to_csv(audit_csv, index=False)
    print(f"📝 Wrote audit CSV → {audit_csv} ({len(df)} rows)")

    # Push to DB
    engine = create_engine(DB_URL)
    _write_predictions_to_db(engine, df)

    print(f"✅ Wrote {len(df)} predictions to {SPORTec_TABLE}")
    return len(df), folder


# Allow CLI testing
if __name__ == "__main__":
    n, f = run_phase_inference()
    print(f"Done: {n} rows from {f}")
