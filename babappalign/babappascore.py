#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAScore
============

Deep learned residue–residue scoring interface used by BABAPPAlign.

Features:
---------
✔ Automatic ESM2-650M embedding
✔ Safe model loading (models/babappascore.pt)
✔ Batched scoring for speed
✔ Cosine fallback if model unavailable
✔ CLI: score two sequences or output a score matrix

Usage:
------
babappascore --seqA A.fasta --seqB B.fasta
babappascore --seqA A.fasta --seqB B.fasta --matrix matrix.npy

"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ------------------------------------------------------------
# ESM2 Embedding
# ------------------------------------------------------------

_ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
_tokenizer = None
_esm2_model = None


def load_esm2(device: torch.device):
    """Lazy-load the ESM2-650M embedding model."""
    global _tokenizer, _esm2_model

    if _tokenizer is not None and _esm2_model is not None:
        return _tokenizer, _esm2_model

    print(f"[info] Loading ESM2 model: {_ESM2_MODEL_NAME}")
    _tokenizer = AutoTokenizer.from_pretrained(_ESM2_MODEL_NAME)
    _esm2_model = AutoModel.from_pretrained(_ESM2_MODEL_NAME)
    _esm2_model.to(device)
    _esm2_model.eval()

    return _tokenizer, _esm2_model


def embed_sequence(seq: str, device: torch.device) -> torch.Tensor:
    """Compute per-residue embeddings using ESM2."""
    tokenizer, esm = load_esm2(device)
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = esm(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)  # (L+2, D)

    # remove CLS and EOS tokens
    emb = hidden[1:-1]
    return emb.cpu()


# ------------------------------------------------------------
# Safe model loading
# ------------------------------------------------------------

def safe_load_model(model_path=None, device="cpu"):
    import torch
    from pathlib import Path
    from babappalign.pairwise_model import PairwiseScorer

    if model_path is None:
        model_path = (
            Path.home()
            / ".cache"
            / "babappalign"
            / "models"
            / "babappascore.pt"
        )

    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"BABAPPAScore model not found at {model_path}. "
            "Learned scoring is mandatory."
        )

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt

    model = PairwiseScorer(emb_dim=state_dict[next(iter(state_dict))].shape[1] // 2)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model



# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------

def batched_score(model, A: torch.Tensor, B: torch.Tensor,
                  device: torch.device, batch: int = 4096) -> np.ndarray:
    """
    Compute residue-residue score matrix using model or cosine fallback.

    A: (m, D)
    B: (n, D)

    Returns: numpy matrix (m, n)
    """

    A = A.to(device)
    B = B.to(device)

    m, d = A.shape
    n, _ = B.shape

    # cosine fallback
    if model is None:
        A_n = A / (A.norm(dim=1, keepdim=True) + 1e-12)
        B_n = B / (B.norm(dim=1, keepdim=True) + 1e-12)
        M = (A_n @ B_n.t()).cpu().numpy()
        return M

    # learned model scoring
    S = np.zeros((m, n), dtype=float)
    idx_i = []
    idx_j = []

    for i in range(m):
        for j in range(n):
            idx_i.append(i)
            idx_j.append(j)

    total = len(idx_i)
    ptr = 0

    with torch.no_grad():
        while ptr < total:
            end = min(ptr + batch, total)
            bi = idx_i[ptr:end]
            bj = idx_j[ptr:end]

            ai = A[bi]
            bjv = B[bj]

            try:
                out = model(ai, bjv)
            except Exception:
                # fallback to cosine if call signature mismatched
                ai_n = ai / (ai.norm(dim=1, keepdim=True) + 1e-12)
                bj_n = bjv / (bjv.norm(dim=1, keepdim=True) + 1e-12)
                out = (ai_n * bj_n).sum(dim=1, keepdim=True)

            if isinstance(out, (tuple, list)):
                out = out[0]

            vals = out.detach().cpu().view(-1).numpy()
            for k, v in enumerate(vals):
                S[bi[k], bj[k]] = float(v)

            ptr = end

    return S


# ------------------------------------------------------------
# FASTA loading
# ------------------------------------------------------------

def read_fasta(path: Path) -> Tuple[str, str]:
    seq = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq), path.stem


# ------------------------------------------------------------
# CLI driver
# ------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser(description="BABAPPAScore: deep residue–residue scorer")
    p.add_argument("--seqA", required=True, help="FASTA file A")
    p.add_argument("--seqB", required=True, help="FASTA file B")
    p.add_argument("--matrix", default=None, help="Optional: save score matrix to .npy")
    p.add_argument("--model", default="models/babappascore.pt", help="Path to learned scorer model")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--batch", type=int, default=4096)

    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # load sequences
    seqA, idA = read_fasta(Path(args.seqA))
    seqB, idB = read_fasta(Path(args.seqB))

    # embedding
    print("[info] Embedding sequences with ESM2-650M…")
    embA = embed_sequence(seqA, device)
    embB = embed_sequence(seqB, device)

    # load model
    model = safe_load_model(Path(args.model), device)

    # score matrix
    print("[info] Computing residue–residue score matrix…")
    M = batched_score(model, embA, embB, device, batch=args.batch)

    print(f"[done] Score matrix shape: {M.shape}")

    if args.matrix:
        np.save(args.matrix, M)
        print(f"[saved] Score matrix → {args.matrix}")


if __name__ == "__main__":
    cli()
