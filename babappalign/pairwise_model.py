# babappalign/pairwise_model.py
# pairwise scoring model for residue–residue compatibility.
# Symmetric by construction (alignment-safe).

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


class PairwiseScorer(nn.Module):
    """
    High-quality symmetric scorer with:
    - LayerNorm
    - GELU activation
    - Residual MLP blocks
    - Gradient checkpointing support
    - Xavier initialization

    Inputs:
        e1: [B, D]
        e2: [B, D]
    Output:
        logits: [B]
    """

    def __init__(self, emb_dim=1280, checkpointing=False):
        super().__init__()
        self.checkpointing = checkpointing

        hidden1 = 1024
        hidden2 = 384
        hidden3 = 96

        # Symmetric feature dimension: |e1-e2| and e1*e2
        feat_dim = 2 * emb_dim

        # Input projection
        self.in_proj = nn.Linear(feat_dim, hidden1)

        # Block 1 (residual)
        self.norm1 = nn.LayerNorm(hidden1)
        self.fc1 = nn.Linear(hidden1, hidden1)

        # Block 2
        self.norm2 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        # Block 3
        self.norm3 = nn.LayerNorm(hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)

        # Output head
        self.out = nn.Linear(hidden3, 1)

        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=0.15)
        self.drop2 = nn.Dropout(p=0.10)

        # -------- Xavier initialization --------
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- Residual block definitions ----
    def _block1(self, x):
        h = self.norm1(x)
        h = self.fc1(self.drop1(self.act(h)))
        return x + h

    def _block2(self, x):
        h = self.norm2(x)
        h = self.fc2(self.drop2(self.act(h)))
        return h

    def _block3(self, x):
        h = self.norm3(x)
        h = self.fc3(self.act(h))
        return h

    # ------------------- Forward pass -------------------
    def forward(self, e1, e2):
        """
        Symmetric residue–residue scoring.
        Ensures score(e1,e2) == score(e2,e1).
        """

        # ✅ Symmetric feature construction
        diff = torch.abs(e1 - e2)
        prod = e1 * e2
        x = torch.cat([diff, prod], dim=1)

        x = self.in_proj(x)

        if self.checkpointing:
            x = checkpoint.checkpoint(self._block1, x, use_reentrant=False)
            x = checkpoint.checkpoint(self._block2, x, use_reentrant=False)
            x = checkpoint.checkpoint(self._block3, x, use_reentrant=False)
        else:
            x = self._block1(x)
            x = self._block2(x)
            x = self._block3(x)

        x = self.out(x)
        return x.squeeze(1)
