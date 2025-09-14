# gnn_team_phase_dataset.py
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from torch_geometric.data import Data, Dataset

DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

# Pitch (meters)
FIELD_X, FIELD_Y = 105.0, 68.0


class TeamPhasePassDataset(Dataset):
    """
    Builds team-phase passing graphs and labels them with the aggregated
    best counter formation derived from `formation_vs_formation`.

    Node features per player (N x 6):
      [passes_made, passes_recv, succ_rate, avg_x_norm, avg_y_norm, avg_pass_len_norm]

    Edge features per directed pair (E x 3):
      [count_norm, succ_rate, mean_len_norm]
    """

    def __init__(
        self,
        provider: str,
        formation_table: str = "formation_vs_formation_new",
        limit: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.provider = provider
        self.formation_table = formation_table
        self.limit = limit
        self.engine = create_engine(DB_URL)
        random.seed(seed)

        # Provider-specific schema for wide tables
        if provider == "metrica":
            self.table = "metrica_normalized_sorted"
            self.pass_val = "pass"
            self.receiver_col = "to_player_id"
            self.x_start, self.y_start = "start_x", "start_y"
            self.x_end, self.y_end = "end_x", "end_y"
        elif provider == "sportec":
            self.table = "sportec_normalized_wide"
            self.pass_val = "Pass"
            self.receiver_col = "receiver_player_id"
            self.x_start, self.y_start = "coordinates_x", "coordinates_y"
            self.x_end, self.y_end = "end_coordinates_x", "end_coordinates_y"
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Build label lookups from formation_vs_formation
        (
            self.best_counter_map,   # provider-specific best
            self.form2id,
            self.id2form,
            self.global_best_map,    # provider-agnostic best
            self.global_most_common  # global fallback
        ) = self._build_best_counter_lookup()

        # Preload slice metadata for this provider
        self.samples = self._load_slices()

    # ----------------------------- label building -----------------------------

    def _build_best_counter_lookup(self):
        """
        Creates aggregated best counter formation per:
          (provider, our_side, opponent_form, opponent_phase)

        Also returns provider-agnostic best and a global-most-common fallback.
        """
        fvf = pd.read_sql(f"SELECT * FROM {self.formation_table}", self.engine)

        # Normalize two perspectives
        home_rows = fvf.assign(
            our_side="home",
            opponent_form=fvf["away_form"],
            opponent_phase=fvf["phase_away"],
            our_form=fvf["home_form"],
            strength=fvf["home_strength"],
        )
        away_rows = fvf.assign(
            our_side="away",
            opponent_form=fvf["home_form"],
            opponent_phase=fvf["phase_home"],
            our_form=fvf["away_form"],
            strength=fvf["away_strength"],
        )
        base = pd.concat([home_rows, away_rows], ignore_index=True).dropna(
            subset=["opponent_form", "opponent_phase", "our_form"]
        )

        # Provider-specific aggregation
        agg = (
            base.groupby(
                ["provider", "our_side", "opponent_form", "opponent_phase", "our_form"],
                as_index=False,
            )
            .agg(mean_strength=("strength", "mean"))
        )
        # argmax per context
        idx = agg.groupby(
            ["provider", "our_side", "opponent_form", "opponent_phase"]
        )["mean_strength"].idxmax()
        best = agg.loc[idx].rename(
            columns={"our_form": "best_counter_formation", "mean_strength": "best_counter_strength"}
        )

        best_map = {}
        all_forms = set()
        for r in best.itertuples(index=False):
            key = (r.provider, r.our_side, str(r.opponent_form), str(r.opponent_phase))
            val = str(r.best_counter_formation)
            best_map[key] = val
            all_forms.add(val)

        # Provider-agnostic best
        agg_g = (
            base.groupby(["our_side", "opponent_form", "opponent_phase", "our_form"], as_index=False)
            .agg(mean_strength=("strength", "mean"))
        )
        idx_g = agg_g.groupby(["our_side", "opponent_form", "opponent_phase"])["mean_strength"].idxmax()
        best_g = agg_g.loc[idx_g]
        global_best_map = {
            (r.our_side, str(r.opponent_form), str(r.opponent_phase)): str(r.our_form)
            for r in best_g.itertuples(index=False)
        }
        all_forms.update(global_best_map.values())

        # Global most-common (for last-resort fallback)
        if all_forms:
            form_counts = pd.Series(list(all_forms)).value_counts()
            global_most_common = form_counts.index[0]
        else:
            global_most_common = "4-2-3-1"  # harmless default if empty DB

        # Label maps
        id2form = {i: f for i, f in enumerate(sorted(all_forms))}
        form2id = {f: i for i, f in id2form.items()}

        return best_map, form2id, id2form, global_best_map, global_most_common

    def _load_slices(self):
        """
        Build (our_side, opponent_form, opponent_phase) slices for this provider,
        then attach label ids using lookups with fallback rules.
        """
        fvf = pd.read_sql(
            f"SELECT * FROM {self.formation_table} WHERE provider = '{self.provider}'",
            self.engine,
        )

        home = fvf.assign(
            our_side="home", opponent_form=fvf["away_form"], opponent_phase=fvf["phase_away"]
        )
        away = fvf.assign(
            our_side="away", opponent_form=fvf["home_form"], opponent_phase=fvf["phase_home"]
        )
        combined = pd.concat([home, away], ignore_index=True).dropna(
            subset=["opponent_form", "opponent_phase"]
        )

        def pick_label(r):
            key = (r.provider, r.our_side, str(r.opponent_form), str(r.opponent_phase))
            if key in self.best_counter_map:
                return self.best_counter_map[key]
            key_g = (r.our_side, str(r.opponent_form), str(r.opponent_phase))
            if key_g in self.global_best_map:
                return self.global_best_map[key_g]
            return self.global_most_common  # final fallback

        combined["label_form"] = combined.apply(pick_label, axis=1)
        combined["label_id"] = combined["label_form"].map(self.form2id)

        if self.limit:
            combined = combined.sample(n=min(self.limit, len(combined)), random_state=42)

        return combined.reset_index(drop=True)

    # --------------------------------- dataset ---------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        import pandas as pd
        import torch
        from torch_geometric.data import Data
        from collections import defaultdict

        row = self.samples.iloc[idx]
        our_side = row.our_side
        y_class = int(row.label_id)

        # Optional IDs (may be None/NaN or strings for metrica)
        match_id = row["match_id"] if "match_id" in row else None
        team_id = row["team_id"] if "team_id" in row else None
        target_team_id = None
        if our_side == "home":
            target_team_id = row["home_team_id"] if "home_team_id" in row else None
        else:
            target_team_id = row["away_team_id"] if "away_team_id" in row else None
        if target_team_id is None or (isinstance(target_team_id, float) and pd.isna(target_team_id)):
            target_team_id = team_id  # fallback

        # ---- helpers for robust WHERE building ----
        def is_nullish(val):
            return val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val == "")

        def sql_eq(col, val):
            # quote strings, leave numbers unquoted
            if isinstance(val, (int, np.integer)):
                return f"{col} = {int(val)}"
            try:
                # numeric-looking strings are fine as numbers
                if isinstance(val, str) and val.strip().lstrip("-").isdigit():
                    return f"{col} = {int(val)}"
            except Exception:
                pass
            # escape single quotes in strings
            sval = str(val).replace("'", "''")
            return f"{col} = '{sval}'"

        filters = []
        if not is_nullish(match_id):
            filters.append(sql_eq("match_id", match_id))
        if not is_nullish(target_team_id):
            filters.append(sql_eq("team_id", target_team_id))
        filters.append(f"LOWER(original_event) = LOWER('{self.pass_val}')")
        filters.append(f"{self.receiver_col} IS NOT NULL")
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""

        q_passes = f"SELECT * FROM {self.table} {where_clause}"
        passes = pd.read_sql(q_passes, self.engine)
        if passes.empty:
            raise IndexError("No pass data for slice")

        # If team_id wasn’t known or didn’t match, infer it from the slice and re-filter (no int() cast!)
        if is_nullish(target_team_id) and "team_id" in passes.columns and not passes["team_id"].isna().all():
            target_team_id = passes["team_id"].mode().iloc[0]
            passes = passes[passes["team_id"] == target_team_id].copy()
            if passes.empty:
                raise IndexError("No pass data after inferring team_id")

        # ---- Build nodes for ONE TEAM ----
        player_ids = pd.unique(pd.concat([passes["player_id"], passes[self.receiver_col]])).tolist()
        if len(player_ids) == 0:
            raise IndexError("No players in pass data")

        pid2idx = {pid: i for i, pid in enumerate(player_ids)}
        N = len(player_ids)

        # Aggregate per-player (use passer start and receiver end locations)
        agg = defaultdict(lambda: {
            "made": 0, "recv": 0, "succ": 0.0,
            "sx": 0.0, "sy": 0.0, "cnt_made": 0,
            "rx": 0.0, "ry": 0.0, "cnt_recv": 0,
            "len": 0.0
        })

        for r in passes.itertuples(index=False):
            pid_p = r.player_id
            pid_r = getattr(r, self.receiver_col)

            agg[pid_p]["made"] += 1
            agg[pid_p]["succ"] += float(getattr(r, "is_successful", False))
            agg[pid_p]["sx"] += float(getattr(r, self.x_start))
            agg[pid_p]["sy"] += float(getattr(r, self.y_start))
            agg[pid_p]["cnt_made"] += 1
            dx = float(getattr(r, self.x_end)) - float(getattr(r, self.x_start))
            dy = float(getattr(r, self.y_end)) - float(getattr(r, self.y_start))
            agg[pid_p]["len"] += float(np.hypot(dx, dy))

            agg[pid_r]["recv"] += 1
            agg[pid_r]["rx"] += float(getattr(r, self.x_end))
            agg[pid_r]["ry"] += float(getattr(r, self.y_end))
            agg[pid_r]["cnt_recv"] += 1

        X = np.zeros((N, 6), dtype=np.float32)
        for pid, idxp in pid2idx.items():
            a = agg[pid]
            made = int(a["cnt_made"])
            rec  = int(a["cnt_recv"])
            touches = max(1, made + rec)

            succ_rate = a["succ"] / max(1, made)
            avg_x = (a["sx"] + a["rx"]) / touches
            avg_y = (a["sy"] + a["ry"]) / touches
            avg_len = a["len"] / max(1, made)

            X[idxp] = [
                float(a["made"]),
                float(a["recv"]),
                float(np.clip(succ_rate, 0.0, 1.0)),
                float(np.nan_to_num(avg_x, nan=0.0)),
                float(np.nan_to_num(avg_y, nan=0.0)),
                float(np.nan_to_num(avg_len / 105.0, nan=0.0)),
            ]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Edges (one team) ----
        total_passes = max(1, len(passes))
        edges, eattr = [], []
        for (u, v), g in passes.groupby(["player_id", self.receiver_col]):
            if u not in pid2idx or v not in pid2idx:
                continue
            u_idx, v_idx = pid2idx[u], pid2idx[v]
            count = float(len(g))
            succ = float(np.nan_to_num(g["is_successful"].astype(float).mean(), nan=0.0))
            mean_len = float(
                np.hypot(
                    (g[self.x_end] - g[self.x_start]).astype(float).mean(),
                    (g[self.y_end] - g[self.y_start]).astype(float).mean(),
                )
            )
            edges.append([u_idx, v_idx])
            eattr.append([count / total_passes, np.clip(succ, 0.0, 1.0), mean_len / 105.0])

        if not edges:
            raise IndexError("No edges built for slice")

        edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(eattr, dtype=np.float32))
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y_class, dtype=torch.long)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.provider = self.provider
        data.our_side = our_side
        data.opponent_form = row.opponent_form
        data.opponent_phase = row.opponent_phase
        return data
