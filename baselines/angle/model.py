"""
ANGLE model — implemented from:
  "Keep It Simple: Towards Accurate Vulnerability Detection for Large Code Graphs"
  arXiv 2412.10164, December 2024

Architecture:
  1. Word2Vec node embeddings (100-dim → hidden=64 via linear projection)
  2. SAGPooling (ratio=0.5) — hierarchical graph refinement to reduce noise
  3. Alternating GCN (local) + TransformerConv (global) layers × num_layers
  4. Global mean pooling for both GCN and GT streams
  5. Concatenate → MLP → 2-class output

Paper reports Devign F1=60.22%, Acc=58.87% (GNN-GT backbone beats AMPLE by
34-161% accuracy on large graphs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, TransformerConv, SAGPooling, global_mean_pool
)


class ANGLE(nn.Module):
    """
    ANGLE: Attention-based hieNarchical Graph LEarning for vulnerability detection.

    Args:
        vocab_size:  number of code tokens in vocabulary
        embed_dim:   Word2Vec embedding dimension (100)
        hidden:      hidden channel width (paper: 64)
        pool_ratio:  SAGPool keep ratio (paper: 0.5)
        num_layers:  alternating GCN+GT pairs (paper: 3)
        dropout:     dropout rate
        pretrained_emb: optional FloatTensor [vocab_size, embed_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = 100,
        hidden:     int = 64,
        pool_ratio: float = 0.5,
        num_layers: int = 3,
        dropout:    float = 0.1,
        pretrained_emb: torch.Tensor = None,
    ):
        super().__init__()

        # Token embedding layer (initialized from Word2Vec if provided)
        self.embed = nn.Embedding(vocab_size + 3, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            with torch.no_grad():
                # pretrained_emb: [vocab_size, embed_dim], indices 2..vocab_size+1
                self.embed.weight[2:2 + pretrained_emb.size(0)].copy_(pretrained_emb)

        self.input_proj = nn.Linear(embed_dim, hidden)

        # Hierarchical graph refinement
        self.sag_pool = SAGPooling(hidden, ratio=pool_ratio)

        # Alternating GCN + TransformerConv layers
        self.gcns = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(num_layers)])
        self.gts  = nn.ModuleList([
            TransformerConv(hidden, hidden, heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers * 2)
        ])

        self.dropout = nn.Dropout(dropout)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x: [N, max_seq_len] token indices (long) OR [N] node-type ints
        # Handle both: if 1-D, unsqueeze
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [N, 1]

        # Embed tokens, mean-pool over sequence dimension
        x = self.embed(x)      # [N, seq_len, embed_dim]
        x = x.mean(dim=1)      # [N, embed_dim]

        x = F.relu(self.input_proj(x))   # [N, hidden]

        # SAGPooling — hierarchical graph refinement
        if edge_index.size(1) > 0:
            x, edge_index, _, batch, _, _ = self.sag_pool(
                x, edge_index, batch=batch
            )

        # Alternating GCN + Graph Transformer with residual connections
        gnn_x = x
        gt_x  = x
        for i, (gcn, gt) in enumerate(zip(self.gcns, self.gts)):
            # GCN branch
            gnn_res = gnn_x
            gnn_x   = gcn(gnn_x, edge_index)
            gnn_x   = self.layer_norms[2 * i](gnn_x + gnn_res)
            gnn_x   = F.relu(gnn_x)
            gnn_x   = self.dropout(gnn_x)

            # Graph Transformer branch
            gt_res  = gt_x
            gt_x    = gt(gt_x, edge_index)
            gt_x    = self.layer_norms[2 * i + 1](gt_x + gt_res)
            gt_x    = F.relu(gt_x)
            gt_x    = self.dropout(gt_x)

        # Global mean pooling for both streams
        gnn_pooled = global_mean_pool(gnn_x, batch)   # [B, hidden]
        gt_pooled  = global_mean_pool(gt_x,  batch)   # [B, hidden]

        out = torch.cat([gnn_pooled, gt_pooled], dim=-1)  # [B, 2*hidden]
        return self.classifier(out)
