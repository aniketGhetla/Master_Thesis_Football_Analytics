# app/services/formation_vs_formation.py
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# if your xG helper lives in app/services/xg_model.py:
from app.services.xg_model import compute_xg  # adjust if your path differs

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"
OUT_TABLE = "formation_vs_formation_new_temp"

FPS = 25
MIN_OVERLAP_SEC = 10
INCLUDE_PHASE = True  # keep phases in the output (phase_home / phase_away)

# -------- provider configs (TEMP tables) --------
PROVIDERS = {
    "sportec": {
        "table": "sportec_normalized_wide_temp",   # <-- temp
        "period_col": "period_id", "frame_col": "frame",
        "possession_col": "team_possession",   # 'home' / 'away'
        "home_tag": "home", "away_tag": "away",
        "event_col": "event_type",
        "event_shot_values": {"SHOTWIDE","SUCCESSFULSHOT","SAVEDSHOT","BLOCKEDSHOT","SHOTWOODWORK"},
        "event_pass_values": {"PASS","CROSS","FREEKICKPASS","GOAL_KICK"},
        "event_def_values": {"TACKLE","INTERCEPTION","CHALLENGE","CLEARANCE","BALL_RECOVERY","FOUL"},
        "event_fallback_col": "original_event",
        "ball_x": "ball_x", "ball_y": "ball_y",
    },
    "metrica": {
        "table": "metrica_normalized_sorted",
        "period_col": "period_id_x", "frame_col": "frame",
        "possession_col": "team_id",           # 'FIFATMA' / 'FIFATBA'
        "home_tag": "FIFATMA", "away_tag": "FIFATBA",
        "event_col": "event_type",
        "event_shot_values": {"SHOT","GOAL"},
        "event_pass_values": {"PASS","CROSS"},
        "event_def_values": {"TACKLE","INTERCEPTION","CHALLENGE","CLEARANCE","RECOVERY","FOUL"},
        "event_fallback_col": "original_event",
        "ball_x": "ball_x", "ball_y": "ball_y",
    }
}

# ----------------- helpers -----------------
def ensure_out_table(engine):
    cols_phase = "phase_home TEXT, phase_away TEXT," if INCLUDE_PHASE else ""
    with engine.begin() as conn:
        conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {OUT_TABLE} (
            provider TEXT,
            {cols_phase}
            home_form TEXT, away_form TEXT,
            pairs INT,
            home_xg DOUBLE PRECISION, away_xg DOUBLE PRECISION,
            home_possession_pct DOUBLE PRECISION, away_possession_pct DOUBLE PRECISION,
            home_recovery_s DOUBLE PRECISION, away_recovery_s DOUBLE PRECISION,
            home_strength DOUBLE PRECISION, away_strength DOUBLE PRECISION,
            home_strength_ml DOUBLE PRECISION, away_strength_ml DOUBLE PRECISION,
            home_xt DOUBLE PRECISION, away_xt DOUBLE PRECISION,
            home_ppda DOUBLE PRECISION, away_ppda DOUBLE PRECISION,
            home_ppda_press DOUBLE PRECISION, away_ppda_press DOUBLE PRECISION,
            home_press_rate DOUBLE PRECISION, away_press_rate DOUBLE PRECISION,
            home_avg_press DOUBLE PRECISION, away_avg_press DOUBLE PRECISION,
            home_width DOUBLE PRECISION, home_depth DOUBLE PRECISION, home_stretch DOUBLE PRECISION,
            away_width DOUBLE PRECISION, away_depth DOUBLE PRECISION, away_stretch DOUBLE PRECISION
        );"""))
        conn.execute(text(f"TRUNCATE TABLE {OUT_TABLE};"))

def get_block_windows(engine):
    # from TEMP Shaw table
    q = """
    SELECT provider, side, phase, period_id, block_id,
           MIN(block_start_frame) AS start_frame,
           MAX(block_end_frame)   AS end_frame
    FROM formation_templates_shaw_temp
    GROUP BY provider, side, phase, period_id, block_id;
    """
    return pd.read_sql(q, engine)

def get_block_forms(engine):
    # from TEMP predictions table
    q = "SELECT provider, side, phase, period_id, block_id, predicted_formation FROM formation_predictions_temp;"
    return pd.read_sql(q, engine)

# --- shots/xG ---
def detect_shots(df, cfg):
    shot = pd.Series(False, index=df.index)
    evc, evf, vals = cfg["event_col"], cfg["event_fallback_col"], cfg["event_shot_values"]
    if evc in df.columns:
        shot |= df[evc].astype("string").str.upper().isin(vals)
    if evf in df.columns:
        s = df[evf].astype("string").str.lower()
        shot |= s.str.contains("shot|goal", regex=True, na=False)
    return shot.fillna(False)

def side_xg_from_wide(wb, cfg, side):
    pcol = cfg["possession_col"]
    if not pcol or pcol not in wb.columns: return 0.0
    tag = cfg["home_tag"] if side == "home" else cfg["away_tag"]
    shot_mask = detect_shots(wb, cfg)
    if not shot_mask.any(): return 0.0
    poss = wb[pcol].astype("string").str.lower().fillna("")
    rows = wb[shot_mask & (poss == str(tag).lower())]
    if rows.empty: return 0.0
    shots = rows[[cfg["ball_x"], cfg["ball_y"]]].dropna().rename(
        columns={cfg["ball_x"]:"coordinates_x", cfg["ball_y"]:"coordinates_y"})
    if shots.empty: return 0.0
    try:
        xg_shots = compute_xg(shots, x_col="coordinates_x", y_col="coordinates_y")
        return float(xg_shots["xG"].sum())
    except Exception:
        return 0.0

# --- possession & recovery ---
def poss_and_recovery(wblock, cfg, side):
    pcol = cfg["possession_col"]
    if not pcol or pcol not in wblock.columns: return np.nan, np.nan
    tag_l = (cfg["home_tag"] if side=="home" else cfg["away_tag"]).lower()
    poss = wblock[pcol].astype("string").str.lower().fillna("")
    total = len(poss); poss_pct = (poss==tag_l).sum()/total*100.0 if total else np.nan
    changes = poss.ne(poss.shift(1)).fillna(False).to_numpy()
    last = poss.shift(1).fillna(poss.iloc[0]).to_numpy()
    lost_i=None; recs=[]
    for i,ch in enumerate(changes):
        if not ch: continue
        cur=poss.iloc[i]; prev=last[i]
        if prev==tag_l and cur!=tag_l: lost_i=i
        elif cur==tag_l and lost_i is not None: recs.append(i-lost_i); lost_i=None
    avg_rec_s = (np.mean(recs)/FPS) if recs else np.nan
    return poss_pct, (float(avg_rec_s) if not np.isnan(avg_rec_s) else np.nan)

# --- legacy quick strength ---
def team_strength(xg, poss_pct, rec_s):
    poss = 0.0 if (poss_pct is None or np.isnan(poss_pct)) else float(poss_pct)
    invr = (1.0/float(rec_s)) if (rec_s is not None and not np.isnan(rec_s) and rec_s>0) else 0.0
    return 0.5*float(xg) + 0.3*poss + 0.2*invr

# --- xT (toy model; goal to the right) ---
def xt_at_xy(x, y, L=53.0):
    d = np.hypot(L - x, y)
    ang = np.arctan2(np.abs(y), max(1e-6, L - x))
    return np.exp(-d/20.0) * (1.0 - ang/(np.pi/2))

def window_xt(wb, cfg, side):
    pcol = cfg["possession_col"]
    if not pcol or pcol not in wb.columns: return 0.0
    tag_l = (cfg["home_tag"] if side=="home" else cfg["away_tag"]).lower()
    poss = wb[pcol].astype("string").str.lower().fillna("")
    sub = wb[poss==tag_l][[cfg["ball_x"],cfg["ball_y"]]].dropna()
    if sub.empty: return 0.0
    return float(np.mean([xt_at_xy(x,y) for x,y in sub.to_numpy(float)]))

# --- PPDA / pressure ---
def detect_pass(df, cfg):
    out = pd.Series(False, index=df.index)
    evc, evf = cfg["event_col"], cfg["event_fallback_col"]
    if evc in df.columns:
        out |= df[evc].astype("string").str.upper().isin(cfg["event_pass_values"])
    if evf in df.columns:
        out |= df[evf].astype("string").str.contains("pass|cross", case=False, na=False)
    return out

def detect_def(df, cfg):
    out = pd.Series(False, index=df.index)
    evc, evf = cfg["event_col"], cfg["event_fallback_col"]
    if evc in df.columns:
        out |= df[evc].astype("string").str.upper().isin(cfg["event_def_values"])
    if evf in df.columns:
        out |= df[evf].astype("string").str.contains("tackle|intercept|recover|clear|foul|duel", case=False, na=False)
    return out

def ppda(wb, cfg, defending_side):
    pcol = cfg["possession_col"]
    if pcol not in wb.columns: return np.nan
    tag_def = (cfg["home_tag"] if defending_side=="home" else cfg["away_tag"]).lower()
    tag_att = (cfg["away_tag"] if defending_side=="home" else cfg["home_tag"]).lower()
    poss = wb[pcol].astype("string").str.lower().fillna("")
    passes_att = int((detect_pass(wb, cfg) & (poss==tag_att)).sum())
    def_acts = int((detect_def(wb, cfg) & (poss==tag_def)).sum())
    switches = int(((poss!=poss.shift(1)) & (poss==tag_def)).sum())
    denom = def_acts + switches
    return float(passes_att) / max(1, denom)

def _side_xy_cols(df, side):
    xs = [c for c in df.columns if c.startswith(f"{side}_") and c.endswith("_x")]
    ys = [c[:-2]+"_y" for c in xs]
    return xs, ys

def _frame_metrics_xy(row, xs, ys):
    X = row[xs].to_numpy(float); Y = row[ys].to_numpy(float)
    m = ~np.isnan(X) & ~np.isnan(Y); X,Y = X[m], Y[m]
    if len(X)<3: return np.nan, np.nan, np.nan
    width  = np.nanpercentile(Y,90) - np.nanpercentile(Y,10)
    depth  = np.nanpercentile(X,90) - np.nanpercentile(X,10)
    pts = np.stack([X,Y],1); s=0.0; c=0
    for i in range(len(pts)):
        d = np.hypot(pts[i,0]-pts[i+1:,0], pts[i,1]-pts[i+1:,1])
        s += d.sum(); c += len(d)
    stretch = s/c if c else np.nan
    return float(width), float(depth), float(stretch)

def spacing_metrics(wb, side):
    xs, ys = _side_xy_cols(wb, side)
    if not xs: return (np.nan, np.nan, np.nan)
    vals = [ _frame_metrics_xy(r, xs, ys) for _,r in wb.iterrows() ]
    arr = np.array(vals, float)
    return tuple(np.nanmean(arr, axis=0))

def _goal_xy(team_tag, pitch_len=105.0):
    return (pitch_len/2, 0) if str(team_tag).lower().startswith("home") or str(team_tag).upper()=="FIFATMA" else (-pitch_len/2, 0)

def _nearest_player_to_ball(frame, team_prefix):
    cols = [c[:-2] for c in frame.index if c.startswith(team_prefix) and c.endswith("_x")]
    if not cols: return None
    bx, by = frame.get("ball_x", np.nan), frame.get("ball_y", np.nan)
    if np.isnan(bx) or np.isnan(by): return None
    xy = np.array([[frame[f"{p}_x"], frame[f"{p}_y"]] for p in cols], float)
    m = ~np.isnan(xy).any(axis=1)
    if not m.any(): return None
    idx = np.argmin(np.hypot(xy[m,0]-bx, xy[m,1]-by))
    return [p for p,ok in zip(cols,m) if ok][idx]

def _pressure_on_carrier(frame, att_prefix, def_prefix, pitch_len=105.0, d_back=3.0, max_d_front=9.0, q=1.75):
    carrier = _nearest_player_to_ball(frame, att_prefix)
    if carrier is None: return np.nan
    ax, ay = frame[f"{carrier}_x"], frame[f"{carrier}_y"]
    if np.isnan(ax) or np.isnan(ay): return np.nan
    gx, gy = _goal_xy(att_prefix, pitch_len)
    pg = np.array([gx-ax, gy-ay], float); pg_n = np.linalg.norm(pg) + 1e-9
    player_goal_dist = np.linalg.norm([gx-ax, gy-ay])
    d_front = max_d_front - 0.05*(pitch_len - player_goal_dist)
    tot = 0.0
    for p in [c[:-2] for c in frame.index if c.startswith(def_prefix) and c.endswith("_x")]:
        dx, dy = frame[f"{p}_x"], frame[f"{p}_y"]
        if np.isnan(dx) or np.isnan(dy): continue
        dist = float(np.hypot(dx-ax, dy-ay))
        if dist > max(d_front, d_back): continue
        pd = np.array([dx-ax, dy-ay], float); pd_n = np.linalg.norm(pd) + 1e-9
        z = (1.0 + float(np.clip(np.dot(pg, pd)/(pg_n*pd_n), -1.0, 1.0))) / 2.0
        L = d_back + (d_front - d_back) * ((z**3 + 0.3*z) / 1.3)
        cur = (max(1.0 - dist/max(L,1e-4), 0.0))**q * 100.0
        tot += cur
    return tot

def pressure_ppda(wb, cfg, defending_side, thr=60.0):
    pcol = cfg["possession_col"]
    if pcol not in wb.columns: return np.nan, 0.0, np.nan
    def_tag = cfg["home_tag"] if defending_side=="home" else cfg["away_tag"]
    att_tag = cfg["away_tag"] if defending_side=="home" else cfg["home_tag"]
    poss = wb[pcol].astype("string").str.lower().fillna("")
    att_l = str(att_tag).lower(); def_l = str(def_tag).lower()
    passes_att = int((detect_pass(wb, cfg) & (poss==att_l)).sum())
    press_frames = 0; press_vals = []
    att_prefix = att_l if att_l in ["home","away"] else ("home" if "fifatma" in att_l else "away")
    def_prefix = def_l if def_l in ["home","away"] else ("home" if "fifatma" in def_l else "away")
    for _, fr in wb[poss==att_l].iterrows():
        pr = _pressure_on_carrier(fr, att_prefix, def_prefix)
        if np.isnan(pr): continue
        press_vals.append(pr)
        if pr >= thr: press_frames += 1
    press_rate = (press_frames / max(1, (poss==att_l).sum()))
    avg_press  = (np.mean(press_vals) if press_vals else np.nan)
    ppda_press = float(passes_att) / max(1, press_frames)
    return ppda_press, press_rate, avg_press

# ---- Strength scoring (equal-weight + optional ML) ----
POS_METRICS = ["xg","xt","press_rate","avg_press","possession_pct"]
NEG_METRICS = ["ppda","ppda_press","recovery_s"]

def _robust_minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if not np.isfinite(lo): lo = np.nanmin(s)
    if not np.isfinite(hi) or hi == lo: hi = lo + 1.0
    out = (s - lo) / (hi - lo)
    return out.clip(0, 1)

def add_equal_weight_strength(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in ["home","away"]:
        comps = []
        for m in POS_METRICS:
            col = f"{side}_{m}"
            if col in df: comps.append(_robust_minmax(df[col]))
        for m in NEG_METRICS:
            col = f"{side}_{m}"
            if col in df: comps.append(1.0 - _robust_minmax(df[col]))
        if comps:
            M = np.stack([c.fillna(c.mean()).to_numpy(float) for c in comps], axis=1)
            df[f"{side}_strength_eq"] = np.nanmean(M, axis=1)
        else:
            df[f"{side}_strength_eq"] = np.nan
    return df

def add_ml_strength(df: pd.DataFrame, min_samples:int=50) -> pd.DataFrame:
    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
    except Exception:
        df["home_strength_ml"] = df.get("home_strength_eq", 0.5)
        df["away_strength_ml"] = 1.0 - df["home_strength_ml"]
        return df

    df = df.copy()
    feats = {}
    for m in POS_METRICS:
        h = _robust_minmax(df.get(f"home_{m}", pd.Series(np.nan, index=df.index)))
        a = _robust_minmax(df.get(f"away_{m}", pd.Series(np.nan, index=df.index)))
        feats[f"diff_{m}"] = (h - a)
    for m in NEG_METRICS:
        h = 1.0 - _robust_minmax(df.get(f"home_{m}", pd.Series(np.nan, index=df.index)))
        a = 1.0 - _robust_minmax(df.get(f"away_{m}", pd.Series(np.nan, index=df.index)))
        feats[f"diff_{m}"] = (h - a)
    X = pd.DataFrame(feats).fillna(0.0)
    y_reg = (df.get("home_xg", 0.0) - df.get("away_xg", 0.0)).fillna(0.0)
    y = (y_reg > 0).astype(int)

    if len(X) < min_samples or y.nunique() < 2:
        df["home_strength_ml"] = df.get("home_strength_eq", 0.5)
        df["away_strength_ml"] = 1.0 - df["home_strength_ml"]
        return df

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    clf = LogisticRegressionCV(cv=5, max_iter=2000, scoring="roc_auc")
    clf.fit(Xs, y.values)
    proba = clf.predict_proba(Xs)[:, 1]  # P(home better by ΔxG)
    df["home_strength_ml"] = proba
    df["away_strength_ml"] = 1.0 - proba
    return df

def pair_overlaps(home_blocks: pd.DataFrame, away_blocks: pd.DataFrame, min_overlap_frames: int) -> pd.DataFrame:
    i = j = 0
    pairs = []
    hb = home_blocks.sort_values("start_frame").reset_index(drop=True)
    ab = away_blocks.sort_values("start_frame").reset_index(drop=True)

    while i < len(hb) and j < len(ab):
        hs, he = int(hb.loc[i, "start_frame"]), int(hb.loc[i, "end_frame"])
        as_, ae = int(ab.loc[j, "start_frame"]), int(ab.loc[j, "end_frame"])
        start, end = max(hs, as_), min(he, ae)

        if end >= start and (end - start + 1) >= min_overlap_frames:
            pairs.append({
                "home_block_id": int(hb.loc[i, "block_id"]),
                "away_block_id": int(ab.loc[j, "block_id"]),
                "pair_start": start,
                "pair_end": end,
                "phase_home": hb.loc[i, "phase"],
                "phase_away": ab.loc[j, "phase"],
                "home_form": hb.loc[i, "predicted_formation"],
                "away_form": ab.loc[j, "predicted_formation"],
            })

        if he < ae: 
            i += 1
        else:
            j += 1

    return pd.DataFrame(pairs)

# ----------------- public entry -----------------
def run_formation_vs_formation() -> int:
    engine = create_engine(DB_URL)
    ensure_out_table(engine)

    windows = get_block_windows(engine)
    forms   = get_block_forms(engine)
    blocks  = windows.merge(forms, on=["provider","side","phase","period_id","block_id"], how="left")
    if blocks.empty:
        print("No blocks found."); 
        return 0

    min_overlap_frames = int(MIN_OVERLAP_SEC * FPS)
    out_rows = []

    for prov, cfg in PROVIDERS.items():
        wide = pd.read_sql(
            f"SELECT * FROM {cfg['table']} WHERE ball_status='alive' ORDER BY {cfg['period_col']}, {cfg['frame_col']};",
            engine
        ).rename(columns={cfg["period_col"]:"period_id", cfg["frame_col"]:"frame"})
        if wide.empty: 
            continue

        pblocks = blocks[blocks["provider"] == prov]
        if pblocks.empty: 
            continue

        for period_id, gP in pblocks.groupby("period_id"):
            hb = gP[gP["side"]=="home"][["block_id","phase","start_frame","end_frame","predicted_formation"]]
            ab = gP[gP["side"]=="away"][["block_id","phase","start_frame","end_frame","predicted_formation"]]
            if hb.empty or ab.empty: 
                continue

            pairs = pair_overlaps(hb, ab, min_overlap_frames)
            if pairs.empty: 
                continue

            wP = wide[wide["period_id"] == int(period_id)].copy()
            for _, r in pairs.iterrows():
                wb = wP[(wP["frame"]>=r["pair_start"]) & (wP["frame"]<=r["pair_end"])]
                if wb.empty: 
                    continue

                # base metrics
                hxg = side_xg_from_wide(wb, cfg, "home"); axg = side_xg_from_wide(wb, cfg, "away")
                hposs, hrec = poss_and_recovery(wb, cfg, "home"); aposs, arec = poss_and_recovery(wb, cfg, "away")
                hstr = team_strength(hxg, hposs, hrec); astr = team_strength(axg, aposs, arec)

                # NEW metrics
                hxt = window_xt(wb, cfg, "home"); axt = window_xt(wb, cfg, "away")
                hppda = ppda(wb, cfg, "home"); appda = ppda(wb, cfg, "away")
                hpp, hprate, havgpr = pressure_ppda(wb, cfg, "home")
                app, aprate, aavgpr = pressure_ppda(wb, cfg, "away")
                hw, hd, hs = spacing_metrics(wb, "home"); aw, ad, as_ = spacing_metrics(wb, "away")

                row = {
                    "provider": prov,
                    "home_form": str(r["home_form"] or ""), "away_form": str(r["away_form"] or ""),
                    "home_xg": float(hxg), "away_xg": float(axg),
                    "home_possession_pct": None if np.isnan(hposs) else float(hposs),
                    "away_possession_pct": None if np.isnan(aposs) else float(aposs),
                    "home_recovery_s": None if np.isnan(hrec) else float(hrec),
                    "away_recovery_s": None if np.isnan(arec) else float(arec),
                    "home_strength": hstr, "away_strength": astr,
                    "home_xt": hxt, "away_xt": axt,
                    "home_ppda": hppda, "away_ppda": appda,
                    "home_ppda_press": hpp, "away_ppda_press": app,
                    "home_press_rate": hprate, "away_press_rate": aprate,
                    "home_avg_press": None if np.isnan(havgpr) else float(havgpr),
                    "away_avg_press": None if np.isnan(aavgpr) else float(aavgpr),
                    "home_width": hw, "home_depth": hd, "home_stretch": hs,
                    "away_width": aw, "away_depth": ad, "away_stretch": as_,
                }
                if INCLUDE_PHASE:
                    row["phase_home"] = r["phase_home"]
                    row["phase_away"] = r["phase_away"]
                out_rows.append(row)

    if not out_rows:
        print("No overlapping pairs found — try reducing MIN_OVERLAP_SEC or check columns.")
        return 0

    # Build dataframe → add strengths
    pairs_df = pd.DataFrame(out_rows)
    pairs_df = add_equal_weight_strength(pairs_df)       # adds home_strength_eq / away_strength_eq
    pairs_df = add_ml_strength(pairs_df, min_samples=50) # adds home_strength_ml / away_strength_ml

    # aggregate to formation vs formation
    group_cols = ["provider","home_form","away_form"]
    if INCLUDE_PHASE:
        group_cols = ["provider","phase_home","phase_away","home_form","away_form"]

    agg = pairs_df.groupby(group_cols, dropna=False).agg(
        pairs=("home_xg","count"),
        home_xg=("home_xg","mean"), away_xg=("away_xg","mean"),
        home_possession_pct=("home_possession_pct","mean"),
        away_possession_pct=("away_possession_pct","mean"),
        home_recovery_s=("home_recovery_s","mean"),
        away_recovery_s=("away_recovery_s","mean"),
        home_strength=("home_strength_eq","mean"),
        away_strength=("away_strength_eq","mean"),
        home_strength_ml=("home_strength_ml","mean"),
        away_strength_ml=("away_strength_ml","mean"),
        home_xt=("home_xt","mean"), away_xt=("away_xt","mean"),
        home_ppda=("home_ppda","mean"), away_ppda=("away_ppda","mean"),
        home_ppda_press=("home_ppda_press","mean"), away_ppda_press=("away_ppda_press","mean"),
        home_press_rate=("home_press_rate","mean"), away_press_rate=("away_press_rate","mean"),
        home_avg_press=("home_avg_press","mean"), away_avg_press=("away_avg_press","mean"),
        home_width=("home_width","mean"), home_depth=("home_depth","mean"), home_stretch=("home_stretch","mean"),
        away_width=("away_width","mean"), away_depth=("away_depth","mean"), away_stretch=("away_stretch","mean"),
    ).reset_index()

    agg.to_sql(OUT_TABLE, con=engine, if_exists="replace", index=False)
    print(f"✅ wrote {len(agg)} rows to {OUT_TABLE}")
    return len(agg)

# CLI
if __name__ == "__main__":
    n = run_formation_vs_formation()
    print(f"Done: {n} rows.")
