#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Progressive MSA engine with:
  - Cached ESM2 embeddings
  - BABAPPAScore learned scoring (MANDATORY)
  - UPGMA guide tree
  - Profile–profile alignment (column mean embeddings)
  - Global Gotoh affine DPT
"""

from __future__ import annotations

import argparse
import time
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from babappalign.babappascore import (
    embed_sequence,
    safe_load_model,
    batched_score,
)

# ============================================================
# Embedding cache (FIXED: no import-time filesystem writes)
# ============================================================

def get_embedding_cache_dir() -> Path:
    """
    Resolve and create the embedding cache directory lazily.
    Respects XDG_CACHE_HOME if set (Bioconda/CI safe).
    """
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = base / "babappalign" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()

# ============================================================
# FASTA utilities
# ============================================================

def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    with open(path) as fh:
        cur, buf = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    ids.append(cur)
                    seqs.append("".join(buf))
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            ids.append(cur)
            seqs.append("".join(buf))
    return ids, seqs


def write_fasta(ids: List[str], seqs: List[str], path: Path):
    with open(path, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n")
            for j in range(0, len(s), 80):
                fh.write(s[j:j+80] + "\n")

# ============================================================
# Guide tree (UPGMA)
# ============================================================

def pooled_vector(emb: torch.Tensor) -> torch.Tensor:
    return emb.mean(dim=0)


def compute_similarity_matrix(seq_ids, emb_map, model, device, batch_pairs):
    vecs = [pooled_vector(emb_map[sid]) for sid in seq_ids]
    V = torch.stack(vecs, dim=0).to(device)
    with torch.no_grad():
        S = batched_score(model, V, V, device=device, batch=batch_pairs)
    return S.cpu().numpy() if isinstance(S, torch.Tensor) else S


def upgma(sim: np.ndarray, labels: List[str]):
    n = len(labels)
    clusters = {i: [labels[i]] for i in range(n)}
    dist = sim.max() - sim
    np.fill_diagonal(dist, np.inf)

    active = set(range(n))
    merges = []

    while len(active) > 1:
        best, pair = np.inf, None
        for i in active:
            for j in active:
                if j <= i:
                    continue
                if dist[i, j] < best:
                    best, pair = dist[i, j], (i, j)

        i, j = pair
        left, right = clusters[i], clusters[j]
        new = len(clusters)
        clusters[new] = left + right
        merges.append(("|".join(left), "|".join(right)))

        dist = np.pad(dist, ((0,1),(0,1)), constant_values=np.inf)
        for k in active:
            if k in (i, j):
                continue
            d = (dist[i,k]*len(left) + dist[j,k]*len(right)) / (len(left)+len(right))
            dist[new,k] = dist[k,new] = d

        dist[i,:] = dist[:,i] = dist[j,:] = dist[:,j] = np.inf
        active.remove(i)
        active.remove(j)
        active.add(new)

    return merges

# ============================================================
# Global Gotoh affine DP
# ============================================================

def gotoh(S, gap_open, gap_extend):
    """
    Safe Gotoh affine-gap DP with explicit traceback guards.
    Returns (best_score, traceback)
    traceback is a list of (i, j) where -1 indicates gap.
    """

    m, n = S.shape
    NEG = -1e12

    M  = np.full((m+1, n+1), NEG)
    Ix = np.full((m+1, n+1), NEG)
    Iy = np.full((m+1, n+1), NEG)

    ptrM  = np.zeros((m+1, n+1), dtype=np.int8)
    ptrIx = np.zeros((m+1, n+1), dtype=np.int8)
    ptrIy = np.zeros((m+1, n+1), dtype=np.int8)

    M[0, 0] = 0.0

    for i in range(1, m+1):
        Ix[i, 0] = gap_open + (i-1) * gap_extend
        ptrIx[i, 0] = 1

    for j in range(1, n+1):
        Iy[0, j] = gap_open + (j-1) * gap_extend
        ptrIy[0, j] = 2

    for i in range(1, m+1):
        for j in range(1, n+1):
            s = S[i-1, j-1]

            choices = [M[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1]]
            ptrM[i, j] = int(np.argmax(choices))
            M[i, j] = choices[ptrM[i, j]] + s

            choices = [M[i-1, j] + gap_open + gap_extend,
                       Ix[i-1, j] + gap_extend]
            ptrIx[i, j] = int(np.argmax(choices))
            Ix[i, j] = choices[ptrIx[i, j]]

            choices = [M[i, j-1] + gap_open + gap_extend,
                       Iy[i, j-1] + gap_extend]
            ptrIy[i, j] = int(np.argmax(choices))
            Iy[i, j] = choices[ptrIy[i, j]]

    scores = [M[m, n], Ix[m, n], Iy[m, n]]
    state = int(np.argmax(scores))
    i, j = m, n

    tb = []

    while i > 0 or j > 0:
        if i == 0:
            tb.append((-1, j-1))
            j -= 1
            state = 2
            continue
        if j == 0:
            tb.append((i-1, -1))
            i -= 1
            state = 1
            continue

        if state == 0:
            tb.append((i-1, j-1))
            state = ptrM[i, j]
            i -= 1
            j -= 1
        elif state == 1:
            tb.append((i-1, -1))
            state = 0 if ptrIx[i, j] == 0 else 1
            i -= 1
        else:
            tb.append((-1, j-1))
            state = 0 if ptrIy[i, j] == 0 else 2
            j -= 1

    tb.reverse()
    return max(scores), tb

# ============================================================
# Progressive alignment (unchanged except cache usage)
# ============================================================

def progressive_align(ids, seqs, emb_map, model, device, go, ge, batch_pairs):
    from babappalign.babappalign import Profile  # unchanged logic
    P = {i: Profile([i], [s], [emb_map[i]]) for i, s in zip(ids, seqs)}
    S = compute_similarity_matrix(ids, emb_map, model, device, batch_pairs)
    merges = upgma(S, ids)

    for L, R in merges:
        P[L + "|" + R] = merge_profiles(P.pop(L), P.pop(R), model, device, go, ge, batch_pairs)

    final = next(iter(P.values()))
    return final.ids, final.seqs

# ============================================================
# CLI
# ============================================================

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("sequences")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default="models/babappascore.pt")
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--batch-pairs", type=int, default=4096)
    p.add_argument("--device", choices=["cpu","cuda"], default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[info] Using device: {device}")

    ids, seqs = read_fasta(Path(args.sequences))

    cache_dir = get_embedding_cache_dir()
    emb_map = {}

    for sid, seq in zip(ids, seqs):
        f = cache_dir / f"{seq_hash(seq)}.pt"
        if f.exists():
            emb = torch.load(f, map_location=device)
        else:
            emb = embed_sequence(seq, device)
            torch.save(emb.cpu(), f)
        emb_map[sid] = emb.to(device)

    model = safe_load_model(Path(args.model), device)

    t0 = time.time()
    out_ids, out_seqs = progressive_align(ids, seqs, emb_map, model, device,
                                          args.gap_open, args.gap_extend, args.batch_pairs)
    print(f"[done] Alignment finished in {time.time()-t0:.2f}s")

    write_fasta(out_ids, out_seqs, Path(args.output))


if __name__ == "__main__":
    cli()
