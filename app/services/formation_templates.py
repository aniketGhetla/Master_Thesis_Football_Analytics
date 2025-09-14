import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

# ==== CONFIG ====
FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
ROLL_MIN = 2             # minutes
CHANGE_THRESH_M = 7.0    # meters
MIN_AGG_SECONDS = 45     # per block (keep it; we’ll print why blocks are skipped)

# ✅ Updated to use *_temp columns
PHASE_COL = {"home": "phase_home_pred_temp", "away": "phase_away_pred_temp"}
ELIG_COL  = {"home": "eligible_home_form_temp", "away": "eligible_away_form_temp"}

# ==== DB helpers ====
def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS formation_templates_temp (
            provider TEXT,
            side TEXT,
            phase TEXT,
            period_id INT,
            block_id BIGINT,
            block_start_frame INT,
            block_end_frame INT,
            num_frames INT,
            agg_seconds DOUBLE PRECISION,
            role_index INT,
            mu_x DOUBLE PRECISION,
            mu_y DOUBLE PRECISION,
            cov_xx DOUBLE PRECISION,
            cov_xy DOUBLE PRECISION,
            cov_yy DOUBLE PRECISION
        );"""))

# ==== data helpers ====
def load_provider(engine, provider):
    if provider == "metrica":
        table, period_col = "metrica_normalized_sorted", "period_id_x"
    else:
        # ✅ Updated to temp table
        table, period_col = "sportec_normalized_wide_temp", "period_id"
    q = f"""
      SELECT * FROM {table}
      WHERE ball_status='alive'
      ORDER BY {period_col}, frame;
    """
    df = pd.read_sql(q, engine).rename(columns={period_col: "period_id"})
    return df, table

def player_ids(df, side):
    xs = {c[:-2] for c in df.columns if c.startswith(f"{side}_") and c.endswith("_x")}
    ys = {c[:-2] for c in df.columns if c.startswith(f"{side}_") and c.endswith("_y")}
    return sorted(xs & ys)

def to_meters(xy):            # xy: (T,P,2) in [0..1]
    out = xy.copy()
    out[:, :, 0] *= PITCH_L
    out[:, :, 1] *= PITCH_W
    return out

def center_by_centroid(xy_m): # (T,P,2)
    cent = np.nanmean(xy_m, axis=1, keepdims=True)
    return xy_m - cent

def rolling_mean_nan(x, win):  # nan-aware rolling mean
    T, P, _ = x.shape
    out = np.empty_like(x)
    # cumulative sums with NaNs treated as 0; track counts
    x0 = np.nan_to_num(x, copy=True)
    cs = np.cumsum(x0, axis=0)
    cnt = np.cumsum(~np.isnan(x).any(axis=2), axis=0).astype(float)  # valid frames per player
    cnt[cnt == 0] = 1.0
    for t in range(T):
        a = max(0, t - win + 1); b = t
        s  = cs[b]  - (cs[a-1]  if a>0 else 0)
        c  = cnt[b] - (cnt[a-1] if a>0 else 0)
        c[c == 0] = 1.0
        out[t] = s / c[..., None]
    return out

def detect_blocks(centered_m):
    """Appendix B: new block at t-3min if max per-player shift between R[t] and R[t-3min] > threshold."""
    T = centered_m.shape[0]
    win = int(ROLL_MIN * 60 * FPS)
    if T < 2 * win:
        return [(0, T-1)]
    R = rolling_mean_nan(centered_m, win)
    cuts = [0]
    for t in range(win, T):
        prev = t - win
        diff = R[t] - R[prev]                 # (P,2)
        d = np.nanmax(np.linalg.norm(diff, axis=1))
        if d > CHANGE_THRESH_M:
            cuts.append(max(prev, 0))         # cut at t-3min
    cuts.append(T-1)
    cuts = sorted(set(cuts))
    blocks = []
    for i in range(len(cuts)-1):
        a, b = cuts[i], cuts[i+1]-1
        if a <= b: blocks.append((a, b))
    # merge tiny (<5s)
    min_len = 5 * FPS
    merged = []
    for a, b in blocks:
        if not merged: merged.append([a, b]); continue
        if (b - a + 1) < min_len:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]

def estimate_template(seg):   # seg: (T,P,2) meters, may contain NaNs
    T, P, _ = seg.shape
    mus = np.zeros((P, 2), dtype=float)
    covs = np.zeros((P, 2, 2), dtype=float)
    for p in range(P):
        pts = seg[:, p, :]
        pts = pts[~np.isnan(pts).any(axis=1)]
        if pts.shape[0] == 0:
            mus[p]  = [0.0, 0.0]
            covs[p] = np.eye(2) * 1e-6
        else:
            mus[p]  = np.mean(pts, axis=0)
            covs[p] = np.cov(pts.T) if pts.shape[0] > 1 else np.eye(2) * 1e-6
    return mus, covs

# ==== main ====
def main():
    engine = create_engine(DB_URL)
    ensure_table(engine)

    total_blocks_written = 0

    for provider in ("metrica", "sportec"):
        df, _ = load_provider(engine, provider)
        if df.empty:
            print(f"{provider}: no rows"); 
            continue

        for side in ("home", "away"):
            ids = player_ids(df, side)
            if not ids:
                print(f"{provider}/{side}: no player columns"); 
                continue

            Xcols = [f"{pid}_x" for pid in ids]
            Ycols = [f"{pid}_y" for pid in ids]
            phase_col, elig_col = PHASE_COL[side], ELIG_COL[side]

            if phase_col not in df.columns or elig_col not in df.columns:
                print(f"{provider}/{side}: missing {phase_col} or {elig_col}"); 
                continue

            phases = sorted(df[phase_col].dropna().unique())
            if not phases:
                print(f"{provider}/{side}: no phase values"); 
                continue

            for ph in phases:
                sub = df[(df[phase_col] == ph) & (df[elig_col])].copy()
                if sub.empty:
                    print(f"{provider}/{side}/{ph}: 0 eligible frames")
                    continue

                total_eligible_secs = len(sub) / FPS
                print(f"{provider}/{side}/{ph}: eligible ~{total_eligible_secs:.1f}s across {sub['period_id'].nunique()} period(s)")

                blocks_written_this_phase = 0

                for period_id, g in sub.groupby("period_id"):
                    g = g.sort_values("frame")
                    arr = g[Xcols + Ycols].to_numpy()  # keep NaNs; we handle them later

                    P = len(ids)
                    X = np.stack([arr[:, i]     for i in range(P)], axis=1)
                    Y = np.stack([arr[:, P + i] for i in range(P)], axis=1)
                    XY = np.stack([X, Y], axis=2)                    # (T,P,2) [0..1]
                    XYm = to_meters(XY)
                    centered = center_by_centroid(XYm)

                    total_secs = len(centered) / FPS
                    if total_secs < MIN_AGG_SECONDS:
                        # Not enough eligible time in this period for this phase
                        continue

                    blocks = detect_blocks(centered)
                    kept_rows = []

                    for (a, b) in blocks:
                        if b <= a:
                            continue
                        seg = centered[a:b+1]
                        secs = seg.shape[0] / FPS
                        if secs < MIN_AGG_SECONDS:
                            # too short block — skip
                            continue

                        # remove frames where ALL players are NaN (just in case)
                        seg = seg[~np.isnan(seg).all(axis=(1,2))]
                        if seg.shape[0] == 0:
                            continue

                        mus, covs = estimate_template(seg)
                        start_f = int(g.iloc[a]["frame"])
                        end_f   = int(g.iloc[b]["frame"])
                        block_id = (int(period_id) * 10_000_000) + start_f

                        for r in range(P):
                            C = covs[r]
                            kept_rows.append({
                                "provider": provider,
                                "side": side,
                                "phase": str(ph),
                                "period_id": int(period_id),
                                "block_id": block_id,
                                "block_start_frame": start_f,
                                "block_end_frame": end_f,
                                "num_frames": int(seg.shape[0]),
                                "agg_seconds": float(secs),
                                "role_index": r + 1,
                                "mu_x": float(mus[r, 0]),
                                "mu_y": float(mus[r, 1]),
                                "cov_xx": float(C[0, 0]),
                                "cov_xy": float(C[0, 1]),
                                "cov_yy": float(C[1, 1]),
                            })

                    if kept_rows:
                        out = pd.DataFrame(kept_rows)
                        # ✅ write to temp table
                        out.to_sql("formation_templates_temp", con=engine, if_exists="append", index=False)
                        blocks_written = out["block_id"].nunique()
                        blocks_written_this_phase += blocks_written
                        total_blocks_written += blocks_written
                        print(f"  period {period_id}: +{blocks_written} block(s) kept")

                if blocks_written_this_phase == 0:
                    print(f"{provider}/{side}/{ph}: no blocks >= {MIN_AGG_SECONDS}s survived (check masks/thresholds)")

    if total_blocks_written == 0:
        print("⚠️ No blocks written. Check: eligible seconds, MIN_AGG_SECONDS, and player columns.")
    else:
        print(f"🎉 Done — wrote {total_blocks_written} block(s) into formation_templates_temp")

if __name__ == "__main__":
    main()
