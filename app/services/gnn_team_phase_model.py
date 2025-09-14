# gnn_team_phase_model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

class TeamPhaseGNN(nn.Module):
    def __init__(self, in_dim=6, hid=64, num_classes=10, heads=2, dropout=0.1):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hid, heads=heads, concat=True, dropout=dropout)
        self.g2 = GATv2Conv(hid * heads, hid, heads=1, concat=True, dropout=dropout)

        # Readout: mean + max pooling concatenated
        self.proj = nn.Sequential(
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(hid, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = torch.relu(self.g1(x, edge_index))
        h = torch.relu(self.g2(h, edge_index))

        # Proper batch vector for pooling (supports batch_size >= 1)
        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        hg_mean = global_mean_pool(h, batch)  # [B, hid]
        hg_max  = global_max_pool(h, batch)   # [B, hid]
        hg = torch.cat([hg_mean, hg_max], dim=-1)  # [B, 2*hid]

        emb = self.proj(hg)           # [B, hid] → use this embedding for style tags / fusion
        logits = self.fc(emb)         # [B, num_classes]
        return logits, emb
