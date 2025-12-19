#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAScore
============

Mandatory learned residue–residue scoring using ESM2 embeddings.

- Single backend: transformers + fair-esm
- Lazy model loading
- CUDA → CPU fallback
- XDG-compliant caching
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
_tokenizer = None
_esm_model = None


# ============================================================
# Cache utilities
# ============================================================

def get_cache_dir(subdir: str) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "babappalign" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# ESM embedding
# ============================================================

def load_esm2(device: torch.device):
    global _tokenizer, _esm_model

    if _tokenizer is not None:
        return _tokenizer, _esm_model

    print(f"[info] Loading ESM2 model: {_ESM2_MODEL}")
    _tokenizer = AutoTokenizer.from_pretrained(_ESM2_MODEL)
    _esm_model = AutoModel.from_pretrained(_ESM2_MODEL)
    _esm_model.to(device).eval()

    return _tokenizer, _esm_model


def embed_sequence(seq: str, device: torch.device) -> torch.Tensor:
    tokenizer, model = load_esm2(device)

    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        hidden = out.last_hidden_state.squeeze(0)

    # remove CLS and EOS
    return hidden[1:-1].cpu()


# ============================================================
# Learned scorer model
# ============================================================

def safe_load_model(model_path, device, version="v1.0.4"):
    """
    Load BABAPPAScore model.

    If model_path is None, the pretrained scorer is downloaded once
    from the GitHub release and cached under the XDG cache.
    """

    from babappalign.pairwise_model import PairwiseScorer
    import urllib.request

    # -------------------------------------------------
    # Resolve model path
    # -------------------------------------------------
    if model_path is None:
        cache_dir = get_cache_dir("models")
        path = cache_dir / "babappascore.pt"

        if not path.exists():
            url = (
                "https://github.com/sinhakrishnendu/BABAPPAlign/"
                f"releases/download/{version}/babappascore.pt"
            )
            print("[info] Downloading BABAPPAScore weights...")
            urllib.request.urlretrieve(url, path)
    else:
        path = Path(model_path)

    if not path.exists():
        raise RuntimeError(
            f"BABAPPAScore model not found at {path}. "
            "Learned scoring is mandatory."
        )

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    model = PairwiseScorer()
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
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

    # vectorized index grid
    ii, jj = torch.meshgrid(
        torch.arange(m), torch.arange(n), indexing="ij"
    )
    ii = ii.flatten()
    jj = jj.flatten()

    S = np.zeros((m, n), dtype=float)

    ptr = 0
    total = len(ii)

    with torch.no_grad():
        while ptr < total:
            end = min(ptr + batch, total)
            ai = A[ii[ptr:end]]
            bj = B[jj[ptr:end]]

            out = model(ai, bj)
            vals = out.detach().cpu().numpy()

            for k in range(end - ptr):
                S[ii[ptr + k], jj[ptr + k]] = float(vals[k])

            ptr = end

    return S


# ============================================================
# CLI
# ============================================================

def cli():
    p = argparse.ArgumentParser(description="BABAPPAScore: deep residue scorer")
    p.add_argument("--seqA", required=True)
    p.add_argument("--seqB", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--matrix", default=None)

    args = p.parse_args()

    device = (
        torch.device("cuda")
        if args.device != "cpu" and torch.cuda.is_available()
        else torch.device("cpu")
    )

    def read_fasta(path: Path) -> str:
        return "".join(
            l.strip() for l in open(path) if not l.startswith(">")
        )

    seqA = read_fasta(Path(args.seqA))
    seqB = read_fasta(Path(args.seqB))

    embA = embed_sequence(seqA, device)
    embB = embed_sequence(seqB, device)

    model = safe_load_model(args.model, device)
    M = batched_score(model, embA, embB, device, args.batch)

    print(f"[done] Score matrix shape: {M.shape}")

    if args.matrix:
        np.save(args.matrix, M)


if __name__ == "__main__":
    cli()
