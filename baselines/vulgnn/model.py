"""
VulGNN model — implemented from:
  "Software Vulnerability Detection Using a Lightweight Graph Neural Network"
  arXiv 2603.29216 (2026)

Architecture (adapted for Devign/our CPG format):
  - Node type embedding (14 types → embed_dim=16)
  - Edge type embedding (2 types → edge_dim=4)
  - Linear input projection to hidden=128
  - 6 × ConvGroup(GeneralConv[dot-product attn] + PReLU + GraphNorm + Dropout)
  - Global mean pooling
  - Linear(128, 2) classifier

Key difference from paper: paper uses full token sequences (V=49k) for node
features; we use discrete node types (14 classes) which they also support as
an alternative embedding mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv, GraphNorm, global_mean_pool


class ConvGroup(nn.Module):
    """One ConvGroup block: GeneralConv → PReLU → GraphNorm → Dropout."""

    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: int = 4, dropout: float = 0.08):
        super().__init__()
        self.conv = GeneralConv(
            in_channels,
            out_channels,
            in_edge_channels=edge_dim,
            aggr='mean',
            attention=True,
            attention_type='dot_product',
        )
        self.act  = nn.PReLU()
        self.norm = GraphNorm(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.act(x)
        x = self.norm(x, batch)
        x = self.drop(x)
        return x


class VulGNN(nn.Module):
    """
    Lightweight GNN for whole-graph vulnerability classification.

    Args:
        num_node_types: vocabulary of node type labels (14 for Devign CPGs)
        num_edge_types: vocabulary of edge type labels (2: Ast, Cfg)
        embed_dim:      node type embedding dimension (paper: 16)
        edge_dim:       edge type embedding dimension (paper: 4)
        hidden:         hidden channel width (paper: D=128)
        num_layers:     ConvGroup blocks (paper: 6)
        dropout:        dropout rate (paper: 0.08)
    """

    def __init__(
        self,
        num_node_types: int = 14,
        num_edge_types: int = 2,
        embed_dim:  int = 16,
        edge_dim:   int = 4,
        hidden:     int = 128,
        num_layers: int = 6,
        dropout:    float = 0.08,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, embed_dim)
        self.edge_embed = nn.Embedding(num_edge_types, edge_dim)

        self.input_proj = nn.Linear(embed_dim, hidden)

        self.convs = nn.ModuleList([
            ConvGroup(hidden, hidden, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(hidden, 2)

    def forward(self, data):
        x         = self.node_embed(data.x)          # [N, embed_dim]
        x         = F.relu(self.input_proj(x))        # [N, hidden]
        edge_attr = self.edge_embed(data.edge_attr)   # [E, edge_dim]

        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr, data.batch)

        x = global_mean_pool(x, data.batch)           # [B, hidden]
        return self.classifier(x)                      # [B, 2]
