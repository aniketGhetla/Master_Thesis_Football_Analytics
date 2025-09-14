# app/routes/dashboard_new.py
# Flask version of your dashboard (dark theme)
from __future__ import annotations
import os, math, json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request
from sqlalchemy import create_engine, text
import plotly.graph_objects as go

bp = Blueprint("dashboard_new", __name__, url_prefix="/dashboard-new")

# ---------------------------------------------------------------------
# Config / DB / provider map
# ---------------------------------------------------------------------
DB_URL = os.environ.get(
    "FOOTBALL_DB_URL",
    "postgresql+psycopg2://postgres:123@localhost:5432/Football",
)
_engine = None

RECO_TABLE = "tactical_recommendations_new"
KG_EDGES   = "kg_edges"  # rel='COUNTERS'



def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, future=True)
    return _engine


def qdf(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine().begin() as con:
        return pd.read_sql(text(sql), con, params=params or {})


PROVIDERS = {
    "sportec": {
        "table": "sportec_normalized_wide_temp",
        "wide_table": "sportec_normalized_wide_temp",
        "period_col": "period_id",
        "frame_col": "frame",
        "possession_col": "team_possession",  # 'home'/'away'
        "home_tag": "home",
        "away_tag": "away",
        "ball_x": "ball_x",
        "ball_y": "ball_y",

        "event_col": "original_event",
        "pass_value": "pass",
        "player_id": "player_id",
        "receiver_id": "receiver_player_id",
        "team_id": "team_id",
        "x": "coordinates_x",
        "y": "coordinates_y",
        "x2": "end_coordinates_x",
        "y2": "end_coordinates_y",
        "shot_values": [
            "shotwide",
            "successfulshot",
            "savedshot",
            "blockedshot",
            "shotwoodwork",
        ],
        "pitch_x": 105.0,
        "pitch_y": 68.0,
        "success_col": "success",
    },
    # add more providers if you need
}


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def auto_scale_coords(
    df: pd.DataFrame,
    px: float,
    py: float,
    cols=("x", "y", "x2", "y2"),
    stretch_to_pitch: bool = True,
    margin_ratio: float = 0.075,
) -> pd.DataFrame:
    """Coerce coords to numeric, convert [-1..1] / [0..1] domains, stretch to [0..px/py] with margins."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    def _safe_minmax(s: pd.Series):
        if s is None or s.empty:
            return (None, None)
        lo = pd.to_numeric(s, errors="coerce").min(skipna=True)
        hi = pd.to_numeric(s, errors="coerce").max(skipna=True)
        return (None if pd.isna(lo) else lo, None if pd.isna(hi) else hi)

    def _scale_series(s: pd.Series, lo, hi, target):
        if s is None or lo is None or hi is None:
            return s
        # [-1..1]
        if lo >= -1.01 and hi <= 1.01 and (lo < 0 or hi <= 1.01):
            return (s + 1.0) / 2.0 * target
        # [0..1]
        if lo >= -1e-6 and hi <= 1.0 + 1e-6:
            return s * target
        return s

    # convert domains if needed
    for name, target in (("x", px), ("y", py)):
        if name in out:
            lo, hi = _safe_minmax(out[name])
            out[name] = _scale_series(out[name], lo, hi, target)
    for name, target in (("x2", px), ("y2", py)):
        if name in out:
            lo, hi = _safe_minmax(out[name])
            out[name] = _scale_series(out[name], lo, hi, target)

    # stretch with margins so it fills the pitch neatly
    if stretch_to_pitch:
        def _stretch(s: pd.Series, target, mratio):
            vals = pd.to_numeric(s, errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return s
            s_min, s_max = float(vals.min()), float(vals.max())
            if s_max <= s_min:
                return s
            m = mratio * target
            return (s - s_min) * ((target - 2 * m) / (s_max - s_min)) + m

        for name, target in (("x", px), ("y", py), ("x2", px), ("y2", py)):
            if name in out:
                out[name] = _stretch(out[name], target, margin_ratio)

    # clip + drop invalid starts
    for c, lim in (("x", px), ("x2", px), ("y", py), ("y2", py)):
        if c in out:
            out[c] = out[c].clip(lower=0.0, upper=lim)
    need = [c for c in ("x", "y") if c in out]
    if need:
        out = out.dropna(subset=need)
    return out


def add_pitch(fig: go.Figure, px=105.0, py=68.0, line="#6E6D6D", bg="#FFFFFF") -> go.Figure:
    # outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=px, y1=py, line=dict(color=line, width=2))
    # half, circle
    fig.add_shape(type="line", x0=px / 2, y0=0, x1=px / 2, y1=py, line=dict(color=line))
    fig.add_shape(
        type="circle",
        x0=px / 2 - 9.15,
        y0=py / 2 - 9.15,
        x1=px / 2 + 9.15,
        y1=py / 2 + 9.15,
        line=dict(color=line),
    )
    # boxes + goals
    for side in [0, 1]:
        x0 = 0 if side == 0 else px - 16.5
        x1 = 16.5 if side == 0 else px
        fig.add_shape(type="rect", x0=x0, y0=py / 2 - 20.15, x1=x1, y1=py / 2 + 20.15, line=dict(color=line))
        x0 = 0 if side == 0 else px - 5.5
        x1 = 5.5 if side == 0 else px
        fig.add_shape(type="rect", x0=x0, y0=py / 2 - 9.16, x1=x1, y1=py / 2 + 9.16, line=dict(color=line))
        g0 = -1 if side == 0 else px
        g1 = 1 if side == 0 else px + 1
        fig.add_shape(type="rect", x0=g0, y0=py / 2 - 3.66, x1=g1, y1=py / 2 + 3.66, line=dict(color=line))
    fig.update_layout(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        xaxis=dict(range=[-2, px + 2], visible=False),
        yaxis=dict(range=[-2, py + 8], visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def pitch_figure(px=105.0, py=68.0) -> go.Figure:
    f = go.Figure()
    return add_pitch(f, px, py)


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
def get_distinct_team_ids(provider: str) -> list[str]:
    cfg = PROVIDERS[provider]
    sql = f"""
      SELECT DISTINCT {cfg['team_id']} AS team_id
      FROM {cfg['table']}
      WHERE LOWER({cfg['event_col']}) LIKE :ev
      ORDER BY 1
    """
    df = qdf(sql, {"ev": "%pass%"})
    return df["team_id"].astype(str).tolist()


def load_passes(provider: str, team_id: str | int, stretch=True) -> pd.DataFrame:
    cfg = PROVIDERS[provider]
    succ_sel = f", {cfg['success_col']} AS success" if cfg.get("success_col") else ""
    sql = f"""
      SELECT {cfg['player_id']} AS passer,
             {cfg['receiver_id']} AS receiver,
             {cfg['team_id']}    AS team_id,
             {cfg['x']} AS x, {cfg['y']} AS y,
             {cfg['x2']} AS x2, {cfg['y2']} AS y2
             {succ_sel}
      FROM {cfg['table']}
      WHERE LOWER({cfg['event_col']}) LIKE :pass_like
        AND {cfg['team_id']} = :tid
    """
    df = qdf(sql, {"pass_like": "%pass%", "tid": team_id})
    if df.empty:
        return df

    if "success" not in df.columns:
        df["success"] = df[["x2", "y2"]].notna().all(axis=1)
    else:
        s = df["success"].astype("string").str.lower()
        df["success"] = s.isin(["true", "t", "1", "yes", "y", "successful"])

    df = auto_scale_coords(df, cfg["pitch_x"], cfg["pitch_y"], cols=("x", "y", "x2", "y2"), stretch_to_pitch=stretch)
    return df


def load_shots(provider: str, team_id: str | int, stretch=True) -> pd.DataFrame:
    cfg = PROVIDERS[provider]
    ev_col = cfg["event_col"]
    succ_col = cfg.get("success_col")

    placeholders = ", ".join([f":s{i}" for i, _ in enumerate(cfg["shot_values"])])
    params = {f"s{i}": v for i, v in enumerate(cfg["shot_values"])}
    params["tid"] = team_id

    sql = f"""
      SELECT {cfg['team_id']} AS team_id,
             {cfg['x']} AS x, {cfg['y']} AS y,
             {cfg['x2']} AS x2, {cfg['y2']} AS y2,
             LOWER({ev_col}) AS ev
             {(", " + succ_col + " AS succ") if succ_col else ""}
      FROM {cfg['table']}
      WHERE LOWER({ev_col}) IN ({placeholders})
        AND {cfg['team_id']} = :tid
    """
    df = qdf(sql, params)
    if df.empty:
        return df

    # coerce + fallback start from end if needed
    for c in ("x", "y", "x2", "y2"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df["x"].isna() | df["y"].isna()
    if "x2" in df and "y2" in df:
        df.loc[m, "x"] = pd.to_numeric(df.loc[m, "x2"], errors="coerce")
        df.loc[m, "y"] = pd.to_numeric(df.loc[m, "y2"], errors="coerce")

# --- robust goal flag ---
    is_goal = pd.Series(False, index=df.index)

    if "succ" in df.columns:
        s = df["succ"].astype(str).str.strip().str.lower()
        # try numeric truthiness first (0/1)
        s_num = pd.to_numeric(df["succ"], errors="coerce")
        num_mask = s_num.notna()
        is_goal.loc[num_mask] = s_num.loc[num_mask] > 0

        # string tokens that should count as True
        TRUE_TOKENS = {"true", "t", "y", "yes", "1", "success", "successful", "goal"}
        is_goal |= s.isin(TRUE_TOKENS)

    # also treat explicit goal labels in the event column as goals
    if "ev" in df.columns:
        is_goal |= df["ev"].astype(str).str.lower().str.contains("goal", na=False)

    df["is_goal"] = is_goal.astype(bool)

    df = df.dropna(subset=["x", "y"])
    df = auto_scale_coords(df, cfg["pitch_x"], cfg["pitch_y"], cols=("x", "y", "x2", "y2"), stretch_to_pitch=stretch)
    keep = ["x", "y", "is_goal"]
    if "x2" in df and "y2" in df:
        keep += ["x2", "y2"]
    return df[keep]


# ---- Avg positions (heatmap + centroids) ------------------------------
def team_heatmap_trace(df: pd.DataFrame, px: float, py: float, bins_x=30, bins_y=20, colorscale="YlOrRd", opacity=0.75):
    if df.empty:
        return None
    xy = df[["x", "y"]].dropna()
    xy = xy[(xy["x"] >= 0) & (xy["x"] <= px) & (xy["y"] >= 0) & (xy["y"] <= py)]
    if xy.empty:
        return None
    return go.Histogram2d(
        x=xy["x"],
        y=xy["y"],
        xbins=dict(start=0, end=px, size=px / bins_x),
        ybins=dict(start=0, end=py, size=py / bins_y),
        colorscale=colorscale,
        showscale=True,
        opacity=opacity,
        hovertemplate="density<extra></extra>",
    )


def player_centroids_and_spread(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player", "mx", "my", "r"])
    g = df.groupby("passer", dropna=True)
    out = g.agg(mx=("x", "mean"), my=("y", "mean"), sx=("x", "std"), sy=("y", "std"), n=("x", "size")).reset_index()
    out["player"] = out["passer"].astype(str)
    out["r"] = out[["sx", "sy"]].mean(axis=1).fillna(2.5).clip(lower=1.5, upper=12.0)
    return out[["player", "mx", "my", "r", "n"]]


def add_player_centroids(fig: go.Figure, cents: pd.DataFrame, color="#1f78b4"):
    if cents.empty:
        return fig
    # soft rings
    for _, r in cents.iterrows():
        fig.add_shape(
            type="circle",
            x0=r["mx"] - r["r"],
            y0=r["my"] - r["r"],
            x1=r["mx"] + r["r"],
            y1=r["my"] + r["r"],
            line=dict(color="rgba(0,0,0,0.25)", width=1),
        )
    fig.add_trace(
        go.Scatter(
            x=cents["mx"],
            y=cents["my"],
            mode="markers+text",
            marker=dict(size=10, color="white", line=dict(color=color, width=2)),
            text=cents["player"].astype(str),
            textposition="middle center",
            hovertemplate="Player %{text}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
            name="Avg positions",
        )
    )
    return fig


# ---- Avg team position (blobs) ---------------------------------------
def player_centroids_cov(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player", "mx", "my", "sx", "sy", "n"])
    g = df.groupby("passer", dropna=True)
    out = g.agg(mx=("x", "mean"), my=("y", "mean"), sx=("x", "std"), sy=("y", "std"), n=("x", "size")).reset_index()
    out["player"] = out["passer"].astype(str)
    out["sx"] = out["sx"].fillna(2.0).clip(lower=1.5, upper=10.0)
    out["sy"] = out["sy"].fillna(2.0).clip(lower=1.5, upper=10.0)
    return out[["player", "mx", "my", "sx", "sy", "n"]]


def add_team_position_blobs(fig: go.Figure, cents: pd.DataFrame, scale=1.8, show_dots=True, show_labels=False):
    if cents.empty:
        return fig
    for _, r in cents.iterrows():
        rx, ry = float(r["sx"]) * scale, float(r["sy"]) * scale
        fig.add_shape(
            type="circle",
            x0=r["mx"] - rx,
            y0=r["my"] - ry,
            x1=r["mx"] + rx,
            y1=r["my"] + ry,
            line=dict(color="rgba(0,0,0,0.15)", width=1),
            fillcolor="rgba(255,0,0,0.30)",
            layer="below",
        )
    if show_dots or show_labels:
        fig.add_trace(
            go.Scatter(
                x=cents["mx"],
                y=cents["my"],
                mode=("markers+text" if show_labels else "markers"),
                marker=dict(size=6, color="rgba(180,0,0,0.9)"),
                text=cents["player"] if show_labels else None,
                textposition="middle right",
                hovertemplate="Player %{text}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
            )
        )
    return fig


# ---- Pressing: possession flips -> ball coords ------------------------
def _detect_flip_targets(poss: pd.Series, to_value: str | int) -> pd.Series:
    s = poss.astype("string")
    to = str(to_value).lower()
    prev = s.shift(1)
    return (s != prev) & (s.str.lower() == to)


def load_recoveries(provider: str, our_side: str, our_team_id: str | int) -> pd.DataFrame:
    cfg = PROVIDERS[provider]
    wt = cfg.get("wide_table")
    poss_col = cfg.get("possession_col")
    if not wt or not poss_col:
        return pd.DataFrame(columns=["x", "y"])

    sql = f"""
      SELECT {cfg['period_col']} AS period_id,
             {cfg['frame_col']}  AS frame,
             {poss_col} AS poss,
             {cfg['ball_x']} AS bx,
             {cfg['ball_y']} AS by,
             ball_status
      FROM {wt}
      ORDER BY {cfg['period_col']}, {cfg['frame_col']}
    """
    df = qdf(sql)
    if df.empty:
        return df

    if "ball_status" in df.columns:
        df = df[df["ball_status"].astype("string").str.lower() == "alive"]

    sample = df["poss"].astype("string").str.lower().head(200)
    uses_homeaway = sample.isin(["home", "away"]).any()

    if uses_homeaway:
        tag = cfg["home_tag"] if our_side == "home" else cfg["away_tag"]
        flips = _detect_flip_targets(df["poss"], tag)
    else:
        flips = _detect_flip_targets(df["poss"], str(our_team_id))

    pts = df.loc[flips, ["bx", "by"]].rename(columns={"bx": "x", "by": "y"}).dropna()
    if pts.empty:
        return pts
    pts = auto_scale_coords(pts, cfg["pitch_x"], cfg["pitch_y"], cols=("x", "y"), stretch_to_pitch=True)
    return pts[["x", "y"]]


def pressing_heat_trace(pts: pd.DataFrame, px: float, py: float, bins_x=30, bins_y=20):
    if pts.empty:
        return None
    pts = pts[(pts["x"] >= 0) & (pts["x"] <= px) & (pts["y"] >= 0) & (pts["y"] <= py)]
    if pts.empty:
        return None
    return go.Histogram2d(
        x=pts["x"],
        y=pts["y"],
        xbins=dict(start=0, end=px, size=px / bins_x),
        ybins=dict(start=0, end=py, size=py / bins_y),
        colorscale="Reds",
        showscale=True,
        opacity=0.85,
        hovertemplate="press density<extra></extra>",
    )


# ---------------------------------------------------------------------
# Pass map helpers (arrows)
# ---------------------------------------------------------------------
def team_and_opponent_names(provider: str, our_side: str, team_id: str | int) -> tuple[str | None, str | None]:
    """
    Returns (team_name, opponent_name) for the selected team_id and our_side.
    Picks the most-common (home_team, away_team) pairing for that team_id.
    """
    if not team_id:
        return None, None

    cfg = PROVIDERS[provider]
    sql = f"""
      SELECT home_team, away_team, COUNT(*) AS cnt
      FROM {cfg['table']}
      WHERE {cfg['team_id']} = :tid
      GROUP BY home_team, away_team
      ORDER BY cnt DESC
      LIMIT 1
    """
    df = qdf(sql, {"tid": team_id})
    if df.empty:
        return None, None

    home_team = str(df.iloc[0]["home_team"]) if "home_team" in df.columns else None
    away_team = str(df.iloc[0]["away_team"]) if "away_team" in df.columns else None

    if our_side == "home":
        return home_team, away_team
    else:
        return away_team, home_team


def team_dropdown_options(provider: str) -> list[tuple[str, str]]:
    """
    Build (value, label) options for the Team select.
    Label is 'TeamName (TeamID)' if we can infer a name; else just TeamID.
    """
    cfg = PROVIDERS[provider]
    # distinct IDs
    ids = qdf(f"SELECT DISTINCT {cfg['team_id']} AS team_id FROM {cfg['table']} ORDER BY 1")["team_id"].astype(str)
    # for each id, get a frequent name (home or away, whichever appears most)
    rows = qdf(f"""
      SELECT {cfg['team_id']} AS team_id, home_team, away_team, COUNT(*) AS cnt
      FROM {cfg['table']}
      GROUP BY {cfg['team_id']}, home_team, away_team
    """)
    label_map: dict[str, str] = {}
    if not rows.empty:
        # pick the name that occurs most with that id
        for tid, g in rows.groupby("team_id"):
            # prefer a non-null non-empty name
            # take the longest non-empty among home/away for the highest count row
            g2 = g.sort_values("cnt", ascending=False)
            name = None
            for _, r in g2.iterrows():
                for col in ("home_team", "away_team"):
                    val = r.get(col)
                    if pd.notna(val) and str(val).strip():
                        name = str(val).strip()
                        break
                if name:
                    break
            if name:
                label_map[str(tid)] = name

    options = []
    for tid in ids:
        label = label_map.get(tid)
        options.append((tid, f"{label} ({tid})" if label else tid))
    return options


def team_names_for_team_id(provider: str, team_id: str | int) -> tuple[str, Optional[str]]:
    """
    Return (team_name, opponent_name) for a given team_id by looking at
    home_team / away_team in sportec_normalized_wide_temp rows for that team.

    If ambiguous, picks the most frequent value seen.
    """
    cfg = PROVIDERS[provider]
    table = cfg["table"]
    # columns we expect to exist in your table
    cols = ["home_team", "away_team", cfg["team_id"]]
    sql = f"SELECT {', '.join(cols)} FROM {table} WHERE {cfg['team_id']} = :tid LIMIT 20000"
    df = qdf(sql, {"tid": team_id})
    if df.empty:
        return str(team_id), None

    # most frequent team name across both columns for this team_id
    name_counts = {}
    for col in ("home_team", "away_team"):
        if col in df.columns:
            vc = df[col].dropna().astype(str).value_counts()
            name_counts.update(vc.to_dict())

    team_name = max(name_counts, key=name_counts.get) if name_counts else str(team_id)

    # estimate opponent: most frequent value of the *other* column
    opponent_name = None
    if {"home_team", "away_team"}.issubset(df.columns):
        # which side was this team more often?
        is_home = (df["home_team"].astype(str) == team_name).sum()
        is_away = (df["away_team"].astype(str) == team_name).sum()
        if is_home >= is_away:
            opp_vc = df["away_team"].dropna().astype(str).value_counts()
        else:
            opp_vc = df["home_team"].dropna().astype(str).value_counts()
        if not opp_vc.empty:
            opponent_name = opp_vc.index[0]

    return team_name, opponent_name


def draw_pass_arrows(fig: go.Figure, df: pd.DataFrame, color="#1fab24"):
    if df.empty:
        return fig
    HEAD_LEN, HEAD_WID = 2.2, 1.2
    for _, r in df.iterrows():
        if pd.isna(r["x2"]) or pd.isna(r["y2"]):
            continue
        x0, y0, x1, y1 = float(r["x"]), float(r["y"]), float(r["x2"]), float(r["y2"])
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if not np.isfinite(L) or L < 1e-6:
            continue
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
        ux, uy = dx / L, dy / L
        vx, vy = -uy, ux
        bx, by = x1 - HEAD_LEN * ux, y1 - HEAD_LEN * uy
        lx, ly = bx + HEAD_WID * vx, by + HEAD_WID * vy
        rx, ry = bx - HEAD_WID * vx, by - HEAD_WID * vy
        fig.add_trace(go.Scatter(x=[x1, lx], y=[y1, ly], mode="lines",
                                 line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x1, rx], y=[y1, ry], mode="lines",
                                 line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
    return fig


def add_pass_network(fig: go.Figure, passes_df: pd.DataFrame, min_edge=3):
    """
    Build a simple pass network on the pitch.
    - Nodes are average locations of passers/receivers.
    - Edges are (passer -> receiver) with count >= min_edge.
    """
    if passes_df.empty:
        return fig

    # Need passer & receiver info
    net_df = passes_df.dropna(subset=["receiver"]).copy()
    if net_df.empty:
        return fig

    net_df["passer"] = net_df["passer"].astype(str)
    net_df["receiver"] = net_df["receiver"].astype(str)

    # Edge counts
    edges = (
        net_df.groupby(["passer", "receiver"])
        .size()
        .reset_index(name="count")
    )
    edges = edges[edges["count"] >= max(1, int(min_edge))]
    if edges.empty:
        return fig

    # Node positions:
    # - for passers use average start (x,y)
    # - for receivers prefer average end (x2,y2) if available
    pos_src = (
        net_df.groupby("passer")[["x", "y"]]
        .mean()
        .rename(columns={"x": "nx", "y": "ny"})
    )  # index name 'passer'

    pos_dst = (
        net_df.groupby("receiver")[["x2", "y2"]]
        .mean()
        .rename(columns={"x2": "nx", "y2": "ny"})
    )  # index name 'receiver'

    # Merge both sets of positions; resulting index is from pos_src but
    # combine_first keeps all players via union of indices.
    nodes = pos_src.combine_first(pos_dst)
    nodes.index.name = "player"            # <-- make the index name consistent
    nodes = nodes.reset_index()            # gives a 'player' column

    # Size nodes by how often they receive
    recv_ct = net_df["receiver"].value_counts()

    def size_for(p, lo=20, hi=40):
        if recv_ct.empty:
            return (lo + hi) / 2
        v = float(recv_ct.get(p, 0.0))
        q10, q90 = np.percentile(recv_ct.values, 10), np.percentile(recv_ct.values, 90)
        if q90 <= q10:
            q90 = q10 + 1.0
        z = np.clip((v - q10) / (q90 - q10), 0.0, 1.0)
        return lo + z * (hi - lo)

    sizes = [size_for(p) for p in nodes["player"]]

    # Draw nodes
    fig.add_trace(
        go.Scatter(
            x=nodes["nx"],
            y=nodes["ny"],
            mode="markers+text",
            marker=dict(size=sizes, color="lightblue", line=dict(color="#185d8a", width=2)),
            text=nodes["player"],
            textposition="middle center",
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Build a quick lookup for coordinates
    pos_map = nodes.set_index("player")[["nx", "ny"]]

    # Draw edges with thin lines (and invisible midpoints for hover)
    midx, midy, midtxt = [], [], []
    for _, e in edges.iterrows():
        u = e["passer"]
        v = e["receiver"]
        if u not in pos_map.index or v not in pos_map.index:
            continue
        x0, y0 = pos_map.loc[u, "nx"], pos_map.loc[u, "ny"]
        x1, y1 = pos_map.loc[v, "nx"], pos_map.loc[v, "ny"]

        w = max(0.7, math.log2(1 + int(e["count"])) * 0.9)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=w, color="#666"), opacity=0.7,
                hoverinfo="skip", showlegend=False
            )
        )
        midx.append((x0 + x1) / 2)
        midy.append((y0 + y1) / 2)
        midtxt.append(f"{u} → {v}<br>passes: {int(e['count'])}")

    fig.add_trace(
        go.Scatter(
            x=midx, y=midy, mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            text=midtxt,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    return fig

# ---------- Tactical panel helpers (put above the ROUTE section) ----------
def latest_reco_row(db_eng, provider: str, our_side: str) -> dict | None:
    sql = f"""
        SELECT *
        FROM {RECO_TABLE}
        WHERE provider = :p AND our_side = :s
        ORDER BY created_at DESC
        LIMIT 1
    """
    with db_eng.begin() as con:
        df = pd.read_sql(text(sql), con, params={"p": provider, "s": our_side})
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    for k in ("topk", "style_tags", "evidence"):
        v = row.get(k)
        if isinstance(v, str):
            try:
                row[k] = json.loads(v)
            except Exception:
                pass
    return row


def kg_counters(db_eng, opponent_form: str, opponent_phase: str, *, limit: int = 12, provider: str | None = None) -> pd.DataFrame:
    phase_norm = str(opponent_phase).strip().lower().replace(" ", "-")
    base = f"""
        SELECT
            src_key AS candidate_form,
            dst_key AS opponent_form,
            COALESCE((props->>'phase'),'') AS phase,
            (props->>'provider') AS provider,
            COALESCE((props->>'support')::float, 0.0) AS support
        FROM {KG_EDGES}
        WHERE rel='COUNTERS'
          AND dst_key = :opp
          AND LOWER(REPLACE(COALESCE(props->>'phase',''), ' ', '-')) = :ph
    """
    prefer_provider = """
      AND (:prov IS NULL OR (COALESCE(props->>'provider','') = :prov OR props->>'provider' IS NULL))
      ORDER BY
        CASE WHEN :prov IS NOT NULL AND COALESCE(props->>'provider','') = :prov THEN 0
             WHEN props->>'provider' IS NULL THEN 1
             ELSE 2 END,
        support DESC
      LIMIT :lim
    """
    any_provider = " ORDER BY support DESC LIMIT :lim"
    with db_eng.begin() as con:
        df = pd.read_sql(
            text(base + prefer_provider),
            con, params={"opp": opponent_form, "ph": phase_norm, "prov": provider, "lim": int(limit)}
        )
        if df.empty:
            df = pd.read_sql(
                text(base + any_provider),
                con, params={"opp": opponent_form, "ph": phase_norm, "lim": int(limit)}
            )
    return df


def build_kg_circle_html(kg_df: pd.DataFrame, opp_form: str, opp_phase: str) -> str | None:
    if kg_df is None or kg_df.empty:
        return None
    cx, cy, R = 0.0, 0.0, 1.0
    uniq = kg_df["candidate_form"].astype(str).unique().tolist()
    fig = go.Figure()
    fig.update_layout(
        title=f"COUNTERS → {opp_form} ({opp_phase})",
        width=720, height=520,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    nodes = [{"name": opp_form, "x": cx, "y": cy}]
    for i, f in enumerate(uniq[:10]):
        th = 2 * np.pi * i / max(1, len(uniq[:10]))
        nodes.append({"name": f, "x": cx + R * np.cos(th), "y": cy + R * np.sin(th)})
    fig.add_trace(go.Scatter(
        x=[n["x"] for n in nodes], y=[n["y"] for n in nodes],
        mode="markers+text", text=[n["name"] for n in nodes],
        marker=dict(size=[26] + [18]*(len(nodes)-1), color=["#ffcc00"] + ["#80b1d3"]*(len(nodes)-1),
                    line=dict(color="#444", width=1)),
        textposition="top center",
    ))
    opp = nodes[0]
    for _, r in kg_df.iterrows():
        src = next((n for n in nodes if n["name"] == r["candidate_form"]), None)
        if src is None:
            continue
        fig.add_trace(go.Scatter(
            x=[src["x"], opp["x"]], y=[src["y"], opp["y"]],
            mode="lines",
            line=dict(width=max(1.0, np.log1p(float(r["support"]))), color="#666"),
            hovertemplate=f"{r['candidate_form']} → {opp_form}<br>support={float(r['support']):.2f}<extra></extra>"
        ))
    return fig.to_html(full_html=False, include_plotlyjs=False)

# ---------------------------------------------------------------------
# ROUTE
# ---------------------------------------------------------------------
@bp.route("/", methods=["GET"])
def dashboard_new():
    provider = request.args.get("provider", "sportec")
    our_side = request.args.get("our_side", "home")
    strategy = request.args.get("strategy", "attack")
    team_id = request.args.get("team_id")

    # Opponent context (for KG table)
    opp_form = request.args.get("opp_form", "4-2-3-1")
    opp_phase = request.args.get("opp_phase", "mid-block")

    # local filters (pass map + network)
    pass_which = request.args.get("pass_which", "both")          # both/successful/unsuccessful
    pass_half = request.args.get("pass_half", "both")            # both/def/att
    min_edge = int(request.args.get("min_edge", 1))

    # avg heatmap controls
    bins_x = int(request.args.get("bins_x", 30))
    bins_y = int(request.args.get("bins_y", 20))
    show_cents = 1 if request.args.get("show_cents", "1") == "1" else 0

    # blobs controls
    blob_scale = float(request.args.get("blob_scale", 1.8))
    show_labels = request.args.get("blob_labels", "0") == "1"
    show_dots = request.args.get("blob_dots", "1") == "1"

    # pressing controls
    press_bins_x = int(request.args.get("press_bins_x", 30))
    press_bins_y = int(request.args.get("press_bins_y", 20))

   # team choices with friendly labels in dropdown
    opt_pairs = team_dropdown_options(provider)   # [(value, label), ...]
    team_options = [v for (v, _) in opt_pairs]
    team_option_labels = {v: lbl for (v, lbl) in opt_pairs}

    if team_id is None and team_options:
        team_id = team_options[0]

    # names for titles (recompute every request)
    team_name, opponent_name = team_and_opponent_names(provider, our_side, team_id)


    team_name, opponent_name = (str(team_id), None)
    if team_id:
        team_name, opponent_name = team_names_for_team_id(provider, team_id)

    cfg = PROVIDERS[provider]
    PX, PY = cfg["pitch_x"], cfg["pitch_y"]

    # --- load data
    passes = load_passes(provider, team_id) if team_id else pd.DataFrame()
    shots = load_shots(provider, team_id) if team_id else pd.DataFrame()

    # filter pass map (status + half)
    pf = passes.copy()
    if not pf.empty:
        mid_x = PX / 2.0
        if pass_which == "successful":
            pf = pf[pf["success"]]
        elif pass_which == "unsuccessful":
            pf = pf[~pf["success"]]
        if pass_half == "def":
            pf = pf[pf["x"] <= mid_x]
        elif pass_half == "att":
            pf = pf[pf["x"] > mid_x]
    # --- counts for Pass Map header
    pass_ok = int(pf["success"].sum()) if (not pf.empty and "success" in pf) else 0
    pass_fail = int((~pf["success"]).sum()) if (not pf.empty and "success" in pf) else 0
    pass_total = pass_ok + pass_fail


    # ------------- PASS MAP -------------
    fig_pm = pitch_figure(PX, PY)
    # direction arrow + label
    fig_pm.add_shape(type="line", x0=PX * 0.25, y0=PY + 3.5, x1=PX * 0.75, y1=PY + 3.5, line=dict(color="#323334", width=3))
    fig_pm.add_annotation(x=PX * 0.75, y=PY + 3.5, ax=PX * 0.70, ay=PY + 3.5,
                          xref="x", yref="y", axref="x", ayref="y",
                          showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor="#323334")
    fig_pm.add_annotation(x=PX * 0.5, y=PY + 6.0, text="Playing direction (left → right)",
                          showarrow=False, font=dict(color="#19191A", size=12))

    if not pf.empty:
        # black start dots for everything shown
        fig_pm.add_trace(go.Scatter(x=pf["x"], y=pf["y"], mode="markers",
                                    marker=dict(size=3.5, color="#111"), hoverinfo="skip", showlegend=False))
        # success arrows
        ok = pf[pf["success"]]
        fail = pf[~pf["success"]]
        if not ok.empty:
            draw_pass_arrows(fig_pm, ok, color="#1fab24")
        # failed (end→red dot + arrows if end available)
        if not fail.empty:
            with_end = fail[fail[["x2", "y2"]].notna().all(axis=1)]
            if not with_end.empty:
                draw_pass_arrows(fig_pm, with_end, color="#d9534f")
            fail_x = fail["x2"].where(fail["x2"].notna(), fail["x"])
            fail_y = fail["y2"].where(fail["y2"].notna(), fail["y"])
            pts = pd.DataFrame({"x": pd.to_numeric(fail_x, errors="coerce"),
                                "y": pd.to_numeric(fail_y, errors="coerce")}).dropna()
            if not pts.empty:
                fig_pm.add_trace(go.Scatter(x=pts["x"], y=pts["y"], mode="markers",
                                            marker=dict(size=4, color="#d9534f", line=dict(color="#7f1d1d", width=0.5)),
                                            hoverinfo="skip", showlegend=False))
    passes_html = fig_pm.to_html(full_html=False, include_plotlyjs="cdn")

    # ------------- PASS NETWORK -------------
    fig_net = pitch_figure(PX, PY)
    add_pass_network(fig_net, passes, min_edge=min_edge)
    net_html = fig_net.to_html(full_html=False, include_plotlyjs=False)

    # ------------- SHOTS -------------
    fig_shots = pitch_figure(PX, PY)
    if not shots.empty:
        goal_y0, goal_y1 = PY / 2 - 3.66, PY / 2 + 3.66
        for _, r in shots.iterrows():
            x, y = float(r["x"]), float(r["y"])
            x2 = float(r["x2"]) if "x2" in shots.columns and pd.notna(r.get("x2")) else min(PX, x + 6.0)
            y2 = float(r["y2"]) if "y2" in shots.columns and pd.notna(r.get("y2")) else y
            if bool(r.get("is_goal", False)):
                x2 = PX - 1.0
                y2 = float(np.clip(y, goal_y0, goal_y1))
                col = "green"
            else:
                col = "red"
            fig_shots.add_trace(go.Scatter(x=[x], y=[y], mode="markers",
                                           marker=dict(color="black", size=5), hoverinfo="skip", showlegend=False))
            fig_shots.add_annotation(x=x2, y=y2, ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y",
                                     showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor=col)
    shots_html = fig_shots.to_html(full_html=False, include_plotlyjs=False)
    goals = int(shots.get("is_goal", pd.Series([], dtype=bool)).sum()) if not shots.empty else 0
    others = int(len(shots) - goals)

    # ------------- AVG POS HEATMAP -------------
    fig_avg = pitch_figure(PX, PY)
    ht = team_heatmap_trace(passes, PX, PY, bins_x=bins_x, bins_y=bins_y)
    if ht is not None:
        fig_avg.add_trace(ht)
    if show_cents:
        cents = player_centroids_and_spread(passes)
        fig_avg = add_player_centroids(fig_avg, cents)
    avgpos_html = fig_avg.to_html(full_html=False, include_plotlyjs=False)

    # ------------- BLOBS -------------
    fig_blobs = pitch_figure(PX, PY)
    cents_cov = player_centroids_cov(passes)
    fig_blobs = add_team_position_blobs(fig_blobs, cents_cov, scale=blob_scale, show_dots=show_dots, show_labels=show_labels)
    teampos_html = fig_blobs.to_html(full_html=False, include_plotlyjs=False)

    # ------------- PRESSING HEATMAP -------------
    press_pts = load_recoveries(provider, our_side, team_id)
    fig_press = pitch_figure(PX, PY)
    ph = pressing_heat_trace(press_pts, PX, PY, bins_x=press_bins_x, bins_y=press_bins_y)
    if ph is not None:
        fig_press.add_trace(ph)
    press_html = fig_press.to_html(full_html=False, include_plotlyjs=False)

    # dropdown options
    strategy_options = ["attack", "defense", "both"]
    phase_options = ["build-up", "mid-block", "high-press", "transition"]
    opp_form_options = ["3-5-2", "4-4-2", "4-3-3", "4-2-3-1"]

    # ---------- Tactical Recommendation + WHY ----------
    # load the latest reco for this provider & side
    reco = latest_reco_row(engine(), provider, our_side)

    # defaults for the template (safe even if there's no reco yet)
    reco_text: str = ""
    reco_form: str | None = None
    reco_conf: float | None = None

    kg_prior_rows: list[dict] = []
    gnn_rows: list[dict] = []
    fused_rows: list[dict] = []

    opp_tags: list[str] = []
    our_tags: list[str] = []

    kg_rows: list[dict] = []     # KG COUNTERS table rows (header context)
    kg_fig_html: str | None = None

    if reco:
        # text + headline numbers
        reco_text = str(reco.get("suggestion_text") or "")
        reco_form = str(reco.get("recommended_formation") or "")
        try:
            reco_conf = float(reco.get("confidence") or 0.0)
        except Exception:
            reco_conf = 0.0

        # evidence blocks (already parsed to dicts in latest_reco_row)
        ev = reco.get("evidence") or {}
        contrib = ev.get("contributions") or {}
        fusion  = ev.get("fusion") or {}

        kg_prior_rows = list(contrib.get("kg_prior") or [])
        gnn_rows      = list(contrib.get("gnn_probs") or [])
        fused_rows    = list(fusion.get("topk") or [])
        opp_tags      = list(contrib.get("style_tags_opponent") or reco.get("style_tags") or [])
        our_tags      = list(contrib.get("style_tags_ours") or [])

    # Build the KG panel from the header context (works with or without a reco row)
    kg_df = kg_counters(engine(), opp_form, opp_phase, limit=12, provider=provider)
    if not kg_df.empty:
        kg_rows = kg_df.sort_values("support", ascending=False).to_dict("records")
        kg_fig_html = build_kg_circle_html(kg_df, opp_form, opp_phase)


    return render_template(
        "dashboard_new.html",
        provider=provider,
        our_side=our_side,
        strategy=strategy,
        strategy_options=strategy_options,
        team_id=team_id,
        team_options=team_options,
        opp_form=opp_form,
        opp_phase=opp_phase,
        opp_form_options=opp_form_options,
        phase_options=phase_options,
        team_name=team_name,
        opponent_name=opponent_name,
        team_option_labels=team_option_labels,
        # pass-map/network state
        pass_which=pass_which,
        pass_half=pass_half,
        min_edge=min_edge,
        pass_ok=pass_ok,
        pass_fail=pass_fail,
        pass_total=pass_total,
        # avg pos heatmap state
        bins_x=bins_x,
        bins_y=bins_y,
        show_cents=show_cents,
        # blobs state
        blob_scale=blob_scale,
        blob_labels=int(show_labels),
        blob_dots=int(show_dots),
        # pressing state
        press_bins_x=press_bins_x,
        press_bins_y=press_bins_y,
        # figures
        passes_html=passes_html,
        net_html=net_html,
        shots_html=shots_html,
        avgpos_html=avgpos_html,
        teampos_html=teampos_html,
        press_html=press_html,
        goals=goals,
        others=others,
                # --- Tactical Recommendation panel ---
        reco=reco,
        reco_text=reco_text,
        reco_form=reco_form,
        reco_conf=reco_conf,
        kg_prior_rows=kg_prior_rows,
        gnn_rows=gnn_rows,
        fused_rows=fused_rows,
        opp_tags=opp_tags,
        our_tags=our_tags,
        kg_rows=kg_rows,
        kg_fig_html=kg_fig_html,

    )
