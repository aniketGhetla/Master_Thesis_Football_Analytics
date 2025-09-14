# app/services/formation_prediction.py
# ------------------------------------------------------------
# Predict formations per (provider/side/phase/block)
# Input : formation_templates_shaw_temp
# Output: formation_predictions_temp
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- DB & tables (temp) ---
DB_URL    = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
SRC_TABLE = "formation_templates_shaw_temp"
OUT_TABLE = "formation_predictions_temp"

RANDOM_STATE = 17
FPS = 25
FRAME_WINDOW = 13      # keep aligned with generator
IMG_W, IMG_H = 105, 68

N_OUTFIELD = 10
K_MIN, K_MAX = 3, 12

# global caps (still use robust per-group thresholds)
TEMPLATE_ABS_COST_MAX = 1500.0
TEMPLATE_MARGIN_MIN   = 50.0

ALLOWED_BY_PHASE = {
    "build-up":   ["2-4-3-1","2-4-4","2-1-4-3","3-4-3","3-1-4-2","3-5-2","3-4-2-1","4-3-3","4-2-3-1","4-4-2"],
    "mid-block":  ["4-2-3-1","4-4-2","4-1-4-1","4-3-2-1","5-3-2","5-2-3"],
    "high-block": ["4-4-2","4-2-3-1","4-3-3"],
    "low-block":  ["5-4-1","5-3-2","4-5-1","4-4-2"],
    "attacking-play": ["2-4-4","2-3-2-3","3-4-3","3-2-4-1","4-2-3-1","4-3-3"],
}

# ---------- output table ----------
def ensure_out_table(engine):
    with engine.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
            provider TEXT,
            side TEXT,
            phase TEXT,
            period_id INT,
            block_id BIGINT,
            predicted_formation TEXT,
            method TEXT,
            w2_cost DOUBLE PRECISION,
            w2_margin DOUBLE PRECISION,
            chosen_template_size INT,
            cluster_label INT,
            n_blocks_in_cluster INT,
            block_start_frame INT,
            block_end_frame INT,
            num_frames INT,
            agg_seconds DOUBLE PRECISION
        );
        """))

# ---------- W2 (Bures) + Hungarian ----------
def _sqrtm_spd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, 1e-12, None))) @ V.T

def w2_gaussians(m1, C1, m2, C2) -> float:
    dm = m1 - m2
    term_mean = float(dm @ dm)
    S2 = _sqrtm_spd(C2)
    S  = S2 @ C1 @ S2
    S_sqrt = _sqrtm_spd(S)
    term_cov = float(np.trace(C1 + C2 - 2.0 * S_sqrt))
    return term_mean + term_cov

def hungarian_cost(out10_A, out10_B, covs_B=None) -> float:
    if covs_B is None:
        covs_B = np.repeat(np.eye(2)[None,:,:] * 0.5, N_OUTFIELD, axis=0)
    C = np.zeros((N_OUTFIELD, N_OUTFIELD), float)
    for i in range(N_OUTFIELD):
        for j in range(N_OUTFIELD):
            C[i, j] = w2_gaussians(out10_A[i], np.eye(2)*0.5, out10_B[j], covs_B[j])
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum())

# ---------- canonical outfield-10 ----------
def squash_adjacent_to_n(pts_lr: np.ndarray, target: int) -> np.ndarray:
    pts = pts_lr.copy()
    while len(pts) > target:
        d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        j = int(np.argmin(d))
        merged = (pts[j] + pts[j+1]) / 2.0
        pts = np.vstack([pts[:j], merged, pts[j+2:]])
    return pts

def canonical_outfield_10(mus_lr: np.ndarray) -> np.ndarray | None:
    if mus_lr.shape[0] < 2:
        return None
    gk_idx = int(np.argmin(mus_lr[:, 1]))  # deepest y → GK
    out = np.delete(mus_lr, gk_idx, axis=0)
    if out.shape[0] < N_OUTFIELD:
        return None
    if out.shape[0] > N_OUTFIELD:
        out = squash_adjacent_to_n(out, N_OUTFIELD)
    return out

# ---------- clustering ----------
def pairwise_cost_matrix(blocks):
    n = len(blocks)
    M = np.zeros((n, n), float)
    for i in range(n):
        for j in range(i+1, n):
            c = hungarian_cost(blocks[i], blocks[j])
            M[i, j] = M[j, i] = c
    return M

def choose_k(D: np.ndarray, kmin=K_MIN, kmax=K_MAX) -> int:
    n = D.shape[0]
    if n <= 2:
        return n
    best_k, best_s = None, -1.0
    for k in range(max(2,kmin), min(kmax,n)+1):
        try:
            ac = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
            labels = ac.fit_predict(D)
            s = silhouette_score(D, labels, metric='precomputed')
            if s > best_s:
                best_s, best_k = s, k
        except Exception:
            pass
    return best_k if best_k else min(2, n)

def cluster_blocks(blocks):
    if len(blocks) == 1:
        return np.array([0]), 1, np.array([[0.0]])
    D = pairwise_cost_matrix(blocks)
    k = choose_k(D)
    ac = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = ac.fit_predict(D)
    return labels, k, D

# ---------- build templates ----------
def build_templates(blocks, labels):
    templates = []
    k = labels.max() + 1
    for c in range(k):
        idx = np.where(labels == c)[0]
        members = [blocks[i] for i in idx]
        # medoid by W2 sum
        costs = [sum(hungarian_cost(A, B) for B in members) for A in members]
        ref = members[int(np.argmin(costs))].copy()

        # align by L2 to ref
        aligned = []
        for X in members:
            C = np.zeros((N_OUTFIELD, N_OUTFIELD), float)
            for i in range(N_OUTFIELD):
                for j in range(N_OUTFIELD):
                    dm = X[i] - ref[j]
                    C[i, j] = dm @ dm
            r, c = linear_sum_assignment(C)
            X_aligned = X[r][np.argsort(c)]
            aligned.append(X_aligned)

        A = np.stack(aligned, axis=0)  # (m,10,2)
        mu = A.mean(axis=0)
        cov = np.zeros((N_OUTFIELD, 2, 2), float)
        for r_ in range(N_OUTFIELD):
            pts = A[:, r_, :]
            if len(pts) > 1:
                cov[r_] = np.cov(pts.T) + np.eye(2)*1e-6
            else:
                cov[r_] = np.eye(2)*1e-6
        templates.append({'mu': mu, 'cov': cov, 'idx': idx})
    return templates

# ---------- naming ----------
def counts_from_y(out10: np.ndarray, k_lines=3):
    y = out10[:, 1]
    qs = np.quantile(y, np.linspace(0, 1, k_lines+1))
    labels = np.digitize(y, qs[1:-1], right=True)
    return [int(np.sum(labels == i)) for i in range(k_lines)]

def best_counts(out10: np.ndarray):
    best, best_var = None, 1e9
    for k in (3,4,5):
        c = counts_from_y(out10, k)
        v = np.var(c)
        if v < best_var:
            best, best_var = c, v
    return best

def project_to_allowed(raw_counts, phase: str) -> str:
    allowed = ALLOWED_BY_PHASE.get(str(phase).lower())
    if not allowed:
        return "-".join(map(str, raw_counts))
    def dist(a, b):
        a = list(a); b = list(map(int, b.split("-")))
        p = abs(len(a)-len(b)) * 2
        s = sum(abs(x - y) for x, y in zip(a[:min(len(a),len(b))], b[:min(len(a),len(b))]))
        return p + s
    best_form, best_d = None, 1e9
    for f in allowed:
        d = dist(raw_counts, f)
        if d < best_d:
            best_form, best_d = f, d
    return best_form or "-".join(map(str, raw_counts))

# ---------- template matching ----------
def match_to_templates(out10: np.ndarray, template_names: list[str], Tdict: dict):
    best = {'formation': None, 'cost': np.inf}
    second = {'formation': None, 'cost': np.inf}
    for mirror in (False, True):
        pts = out10.copy()
        if mirror:
            pts[:, 0] *= -1.0
        for f in template_names:
            mu = Tdict[f]['mu']; cov = Tdict[f]['cov']
            c = hungarian_cost(pts, mu, cov)
            if c < best['cost']:
                second = best.copy()
                best = {'formation': f, 'cost': c}
            elif c < second['cost']:
                second = {'formation': f, 'cost': c}
    margin = second['cost'] - best['cost'] if np.isfinite(second['cost']) else np.inf
    return best['formation'], float(best['cost']), float(margin)

def robust_cost_thresholds(out10s, Tdict):
    names = list(Tdict.keys())
    if not names:
        return np.inf, 0.0
    best_costs, margins = [], []
    for out10 in out10s:
        best = np.inf; second = np.inf
        for mirror in (False, True):
            pts = out10.copy()
            if mirror: pts[:,0] *= -1.0
            for f in names:
                c = hungarian_cost(pts, Tdict[f]['mu'], Tdict[f]['cov'])
                if c < best:
                    second = best; best = c
                elif c < second:
                    second = c
        best_costs.append(best)
        margins.append(second - best if np.isfinite(second) else np.inf)
    best_costs = np.asarray(best_costs, float)
    med = float(np.median(best_costs))
    mad = float(np.median(np.abs(best_costs - med))) * 1.4826
    abs_thr = med + 3*mad if mad > 0 else med * 2.5
    finite_margins = np.asarray([m for m in margins if np.isfinite(m)], float)
    margin_thr = float(np.percentile(finite_margins, 10)) if finite_margins.size else 0.0
    if TEMPLATE_ABS_COST_MAX is not None:
        abs_thr = min(abs_thr, float(TEMPLATE_ABS_COST_MAX))
    if TEMPLATE_MARGIN_MIN is not None:
        margin_thr = max(margin_thr, float(TEMPLATE_MARGIN_MIN))
    return abs_thr, margin_thr

# ---------- ResNet fallback using DB frames ----------
# ---------- ResNet fallback using DB frames (self-contained) ----------
import os, pickle, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models as tv_models
from PIL import Image

# Paths to your trained assets
MODEL_PATH    = r"C:\Users\Harshita\Documents\FootBallProject\app\services\models\resnet_multitask_model.pth"
ENCODERS_PATH = r"C:\Users\Harshita\Documents\FootBallProject\app\services\models\label_encoders.pkl"

# --- tiny image helpers (reuse DB frames to build a 105x68x3 image) ---
def _normalize_coords_img(x, y):
    if pd.isna(x) or pd.isna(y):
        return None, None
    xi = int(np.clip(round(x * IMG_W), 0, IMG_W - 1))
    yi = int(np.clip(round(y * IMG_H), 0, IMG_H - 1))
    return xi, yi

def _get_home_away_ids(cols):
    home, away = set(), set()
    for c in cols:
        if c.endswith("_x"):
            if c.startswith("home_"): home.add(c[:-2])
            elif c.startswith("away_"): away.add(c[:-2])
    return sorted(home), sorted(away)

def _build_image_from_frames(frames: pd.DataFrame):
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    home_ids, away_ids = _get_home_away_ids(frames.columns)
    for _, row in frames.iterrows():
        # ball → blue
        bx, by = _normalize_coords_img(row.get("ball_x"), row.get("ball_y"))
        if bx is not None and by is not None:
            img[by, bx, 2] = 255
        # players
        for pid in home_ids + away_ids:
            xk, yk = f"{pid}_x", f"{pid}_y"
            if xk not in row or yk not in row: 
                continue
            x, y = row[xk], row[yk]
            if pd.isna(x) or pd.isna(y): 
                continue
            xi, yi = _normalize_coords_img(x, y)
            if xi is None or yi is None: 
                continue
            chan = 0 if pid in home_ids else 1  # home=red, away=green
            img[yi, xi, chan] = 255
    return Image.fromarray(img)

def _fetch_frames_for_block(engine, provider: str, period_id: int, start_f: int, end_f: int):
    if provider.lower().startswith("metrica"):
        table, period_col = "metrica_normalized_sorted", "period_id_x"
    else:
        table, period_col = "sportec_normalized_wide_temp", "period_id"

    center = (start_f + end_f) // 2
    half = FRAME_WINDOW // 2
    lo, hi = max(center - half, 0), center + half

    q = f"""
        SELECT * FROM {table}
        WHERE {period_col} = :pid AND frame BETWEEN :lo AND :hi
        ORDER BY frame
    """
    frames = pd.read_sql(text(q), create_engine(DB_URL), params={"pid": period_id, "lo": lo, "hi": hi})
    if len(frames) < FRAME_WINDOW:
        lo2, hi2 = max(start_f, center - half), min(end_f, center + half)
        q2 = f"""
            SELECT * FROM {table}
            WHERE {period_col} = :pid AND frame BETWEEN :lo2 AND :hi2
            ORDER BY frame
        """
        frames = pd.read_sql(text(q2), create_engine(DB_URL), params={"pid": period_id, "lo2": lo2, "hi2": hi2})
    return frames

# --- exact model head used for fallback classification of formations ---
class ResNetMultiTaskFootball(nn.Module):
    def __init__(self, num_home_phases, num_away_phases, num_home_forms, num_away_forms):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # (B,512,1,1)
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.head_hp = nn.Linear(256, num_home_phases)
        self.head_ap = nn.Linear(256, num_away_phases)
        self.head_hf = nn.Linear(256, num_home_forms)
        self.head_af = nn.Linear(256, num_away_forms)
    def forward(self, x):
        x = self.backbone(x)
        x = self.shared(x)
        return self.head_hp(x), self.head_ap(x), self.head_hf(x), self.head_af(x)

# lazy singleton loader
_resnet_cache = {"loaded": False, "model": None, "device": None, "enc": None, "tfm": None}
def _load_resnet_once():
    if _resnet_cache["loaded"]:
        return _resnet_cache["model"], _resnet_cache["enc"], _resnet_cache["tfm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    n_hp = len(enc["home_phase"].classes_)
    n_ap = len(enc["away_phase"].classes_)
    n_hf = len(enc["home_form"].classes_)
    n_af = len(enc["away_form"].classes_)
    model = ResNetMultiTaskFootball(n_hp, n_ap, n_hf, n_af)
    sd = torch.load(MODEL_PATH, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip possible prefixes
    new_sd = {}
    for k, v in sd.items():
        for p in ("module.", "model.", "net.", "network."):
            if k.startswith(p):
                k = k[len(p):]
        new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    _resnet_cache.update(dict(loaded=True, model=model, device=device, enc=enc, tfm=tfm))
    return model, enc, tfm

@torch.inference_mode()
def resnet_predict_from_db(engine, provider: str, side: str, period_id: int, start_f: int, end_f: int):
    model, enc, tfm = _load_resnet_once()
    device = next(model.parameters()).device

    frames = _fetch_frames_for_block(engine, provider, period_id, start_f, end_f)
    if frames.empty:
        return "unknown", 0.0

    pil_img = _build_image_from_frames(frames)
    x = tfm(pil_img).unsqueeze(0).to(device)
    _, _, out_hf, out_af = model(x)

    if side.lower().startswith("home"):
        logits = out_hf; classes = enc["home_form"].classes_
    else:
        logits = out_af; classes = enc["away_form"].classes_

    probs = torch.softmax(logits, dim=1)
    idx = int(torch.argmax(probs, dim=1).item())
    conf = float(probs[0, idx].item())
    label = str(classes[idx])
    return label, conf


# ---------- Public entry ----------
def run_formation_prediction() -> int:
    """
    Reads blocks from formation_templates_shaw_temp,
    writes predictions to formation_predictions_temp.
    Returns number of rows written.
    """
    engine = create_engine(DB_URL)
    ensure_out_table(engine)

    df = pd.read_sql(f"SELECT * FROM {SRC_TABLE}", engine)
    if df.empty:
        print("No rows in formation_templates_shaw_temp; run measure+shaw first.")
        return 0

    # group into blocks and build canonical outfield-10
    group_cols = ["provider","side","phase","period_id","block_id",
                  "block_start_frame","block_end_frame","num_frames","agg_seconds"]
    blocks_meta = []
    for keys, g in df.groupby(group_cols, sort=False):
        g = g.sort_values("role_index")
        mus_lr = g[["mu_x","mu_y"]].to_numpy(float)
        out10 = canonical_outfield_10(mus_lr)
        if out10 is None:
            continue
        blocks_meta.append({
            "provider": keys[0], "side": keys[1], "phase": str(keys[2]),
            "period_id": int(keys[3]), "block_id": int(keys[4]),
            "block_start_frame": int(keys[5]), "block_end_frame": int(keys[6]),
            "num_frames": int(keys[7]), "agg_seconds": float(keys[8]),
            "out10": out10
        })

    if not blocks_meta:
        print("No usable blocks after GK removal/squash.")
        return 0

    dfb = pd.DataFrame(blocks_meta)
    out_rows = []

    # per (provider,side,phase): cluster → templates → gates → classify
    for (provider, side, phase), grp in dfb.groupby(["provider","side","phase"], sort=False):
        out10s = grp["out10"].tolist()
        labels, k, _ = cluster_blocks(out10s)
        cluster_sizes = np.bincount(labels, minlength=labels.max()+1)
        templates = build_templates(out10s, labels)

        # name & merge by phase-aware naming
        named = {}
        for T in templates:
            cnts = best_counts(T['mu'])
            name = project_to_allowed(cnts, phase=phase)
            if name not in named:
                named[name] = {'mu_list':[T['mu']], 'cov_list':[T['cov']], 'members': list(T['idx'])}
            else:
                named[name]['mu_list'].append(T['mu'])
                named[name]['cov_list'].append(T['cov'])
                named[name]['members'].extend(list(T['idx']))

        Tdict = {}
        for name, pack in named.items():
            mu = np.mean(np.stack(pack['mu_list'], axis=0), axis=0)
            cov = np.mean(np.stack(pack['cov_list'], axis=0), axis=0) + np.eye(2)[None,:,:]*1e-6
            Tdict[name] = {'mu': mu, 'cov': cov, 'members': pack['members']}

        template_names = list(Tdict.keys())
        abs_thr, margin_thr = robust_cost_thresholds(out10s, Tdict) if template_names else (np.inf, 0.0)
        print(f"Gate {provider}/{side}/{phase}: abs≤{abs_thr:.1f}, margin≥{margin_thr:.1f}")

        for i, row in grp.reset_index(drop=True).iterrows():
            out10 = row["out10"]
            clabel = int(labels[i])
            n_in_cluster = int(cluster_sizes[clabel])

            pred_form, cost, margin = (None, np.inf, 0.0)
            if template_names:
                pred_form, cost, margin = match_to_templates(out10, template_names, Tdict)

            use_resnet = (pred_form is None) or (cost > abs_thr) or (margin < margin_thr)
            method = "w2_hungarian_template"
            chosen_size = None
            if use_resnet:
                pred_form, conf = resnet_predict_from_db(
                    engine,
                    provider=provider,
                    side=side,
                    period_id=int(row["period_id"]),
                    start_f=int(row["block_start_frame"]),
                    end_f=int(row["block_end_frame"]),
                )
                method = f"resnet_fallback_conf_{conf:.3f}"
                chosen_size = n_in_cluster
            else:
                if pred_form in Tdict:
                    chosen_size = int(len(Tdict[pred_form]['members']))

            out_rows.append({
                "provider": provider,
                "side": side,
                "phase": phase,
                "period_id": int(row["period_id"]),
                "block_id": int(row["block_id"]),
                "predicted_formation": pred_form,
                "method": method,
                "w2_cost": None if not np.isfinite(cost) else float(cost),
                "w2_margin": None if not np.isfinite(margin) else float(margin),
                "chosen_template_size": chosen_size,
                "cluster_label": clabel,
                "n_blocks_in_cluster": n_in_cluster,
                "block_start_frame": int(row["block_start_frame"]),
                "block_end_frame": int(row["block_end_frame"]),
                "num_frames": int(row["num_frames"]),
                "agg_seconds": float(row["agg_seconds"]),
            })

    if not out_rows:
        print("⚠️ No predictions produced.")
        return 0

    out = pd.DataFrame(out_rows)
    # replace so each run is clean
    out.to_sql(OUT_TABLE, con=engine, if_exists="replace", index=False)
    print(f"✅ wrote {len(out)} rows to {OUT_TABLE}")
    return len(out)

# CLI
if __name__ == "__main__":
    n = run_formation_prediction()
    print(f"Done: {n} rows.")
