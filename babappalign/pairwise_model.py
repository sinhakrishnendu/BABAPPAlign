# babappalign/pairwise_model.py
import torch
import torch.nn as nn

class PairwiseScorer(nn.Module):
    def __init__(self, emb_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logits
        )

    def forward(self, e1, e2):
        x = torch.cat([e1, e2], dim=1)
        return self.net(x).squeeze(1)  # return [B] logits
