"""Graph detector architectures. Every model maps a PyG batch with dense node
features `x` to one logit per graph (P(vulnerable) = sigmoid)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GatedGraphConv, GCNConv, GeneralConv, GraphNorm, RGCNConv,
                                SAGPooling, TransformerConv, global_add_pool, global_max_pool,
                                global_mean_pool)


class ResGGNNBlock(nn.Module):
    def __init__(self, hidden, steps=2):
        super().__init__()
        self.conv = GatedGraphConv(hidden, num_layers=steps)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, ei):
        return self.norm(x + self.conv(x, ei))


class Reveal(nn.Module):
    """REVEAL phase 1: residual gated GNN, tanh, mean+max readout (embedding used by phase 2)."""

    def __init__(self, in_dim, hidden=200, blocks=4, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResGGNNBlock(hidden) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def embed(self, b, x):
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h, b.edge_index)
        h = torch.tanh(h)
        return torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], 1)

    def forward(self, b, x):
        return self.head(self.embed(b, x)).squeeze(-1)


class Devign(nn.Module):
    """Gated GNN (6 steps) over AST/CFG/DDG with the GGNN gated readout (Li et al. 2016)."""

    def __init__(self, in_dim, hidden=200, steps=6, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.ggnn = GatedGraphConv(hidden, num_layers=steps)
        self.gate = nn.Linear(hidden + in_dim, 1)
        self.val = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, b, x):
        h = self.ggnn(self.proj(x), b.edge_index)
        g = torch.sigmoid(self.gate(torch.cat([h, x], 1))) * torch.tanh(self.val(h))
        return self.head(global_add_pool(g, b.batch)).squeeze(-1)


class RGCN(nn.Module):
    def __init__(self, in_dim, n_rel, hidden=128, layers=3, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([RGCNConv(hidden, hidden, n_rel, num_bases=min(n_rel, 4)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, b, x):
        h = F.relu(self.proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(h + conv(h, b.edge_index, b.edge_type)))
        return self.head(torch.cat([global_mean_pool(h, b.batch), global_max_pool(h, b.batch)], 1)).squeeze(-1)


class Angle(nn.Module):
    """SAGPool refinement, then alternating GCN (local) and graph-transformer (global) streams."""

    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.1, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pool = SAGPooling(hidden, ratio=0.5)
        self.gcns = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.gts = nn.ModuleList([TransformerConv(hidden, hidden, heads=1, dropout=dropout) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2 * layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, b, x):
        h, ei, batch = F.relu(self.proj(x)), b.edge_index, b.batch
        if ei.size(1) > 0:
            h, ei, _, batch, _, _ = self.pool(h, ei, batch=batch)
        a, t = h, h
        for i, (gcn, gt) in enumerate(zip(self.gcns, self.gts)):
            a = self.drop(F.relu(self.norms[2 * i](a + gcn(a, ei))))
            t = self.drop(F.relu(self.norms[2 * i + 1](t + gt(t, ei))))
        return self.head(torch.cat([global_mean_pool(a, batch), global_mean_pool(t, batch)], 1)).squeeze(-1)


class VulGNN(nn.Module):
    """6 x (GeneralConv with dot-product attention + PReLU + GraphNorm + dropout), mean pool."""

    def __init__(self, in_dim, n_rel, hidden=128, layers=6, edge_dim=4, dropout=0.08, cb_dim=0, **_):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.edge = nn.Embedding(n_rel, edge_dim)
        self.convs = nn.ModuleList([GeneralConv(hidden, hidden, in_edge_channels=edge_dim, aggr="mean",
                                                attention=True, attention_type="dot_product") for _ in range(layers)])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.cb_dim = cb_dim
        self.head = nn.Linear(hidden + cb_dim, 1)

    def forward(self, b, x):
        h, ea = F.relu(self.proj(x)), self.edge(b.edge_type)
        for conv, act, norm in zip(self.convs, self.acts, self.norms):
            h = self.drop(norm(act(conv(h, b.edge_index, edge_attr=ea)), b.batch))
        g = global_mean_pool(h, b.batch)
        if self.cb_dim:  # ablation arm 2: function-level CodeBERT [CLS] (lexical channel)
            g = torch.cat([g, b.cb], 1)
        return self.head(g).squeeze(-1)


ARCHS = {"reveal": Reveal, "devign": Devign, "rgcn": RGCN, "angle": Angle, "vulgnn": VulGNN}
