#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAScore
============

Mandatory learned residue–residue scoring engine.
This model is REQUIRED for BABAPPAlign to function.
"""

from pathlib import Path
import os
import urllib.request

import numpy as np
import torch
import esm

from babappalign.pairwise_model import PairwiseScorer

# ============================================================
# Canonical model location (DO NOT CHANGE)
# ============================================================

CACHE_BASE = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
MODEL_DIR = CACHE_BASE / "babappalign" / "models"
MODEL_PATH = MODEL_DIR / "babappascore.pt"

MODEL_URL = (
    "https://github.com/sinhakrishnendu/BABAPPAlign/"
    "releases/download/v1.0.0/babappascore.pt"
)

# ============================================================
# ESM2 embedding (fair-esm)
# ============================================================

_esm_model = None
_batch_converter = None


def load_esm2(device):
    global _esm_model, _batch_converter
    if _esm_model is None:
        print("[info] Loading ESM2-650M (fair-esm)")
        _esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        _esm_model.to(device)
        _esm_model.eval()
        _batch_converter = alphabet.get_batch_converter()
    return _esm_model, _batch_converter


def embed_sequence(seq: str, device: torch.device) -> torch.Tensor:
    model, batch_converter = load_esm2(device)
    batch = [("seq", seq)]
    _, _, toks = batch_converter(batch)
    toks = toks.to(device)

    with torch.no_grad():
        out = model(toks, repr_layers=[33])
        emb = out["representations"][33][0, 1:-1]

    return emb.cpu()

# ============================================================
# Model download + load (MANDATORY)
# ============================================================

def ensure_model_present() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("[info] BABAPPAScore model not found, downloading...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Failed to download BABAPPAScore model.\n"
            "This model is required for BABAPPAlign.\n"
            f"Error: {e}"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError("Model download failed.")

    return MODEL_PATH


def load_babappascore_model(device: torch.device):
    model_path = ensure_model_present()
    print(f"[info] Loading BABAPPAScore model: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model = PairwiseScorer()
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model

# ============================================================
# Scoring
# ============================================================

def batched_score(model, A: torch.Tensor, B: torch.Tensor,
                  device: torch.device, batch: int = 4096) -> np.ndarray:

    A = A.to(device)
    B = B.to(device)

    m, _ = A.shape
    n, _ = B.shape
    S = np.zeros((m, n), dtype=float)

    idx = [(i, j) for i in range(m) for j in range(n)]

    with torch.no_grad():
        for k in range(0, len(idx), batch):
            sub = idx[k:k + batch]
            ai = torch.stack([A[i] for i, _ in sub])
            bj = torch.stack([B[j] for _, j in sub])
            out = model(ai, bj).view(-1).cpu().numpy()
            for (i, j), v in zip(sub, out):
                S[i, j] = float(v)

    return S
