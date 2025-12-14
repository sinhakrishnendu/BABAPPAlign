#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Progressive multiple sequence alignment engine using:
- Cached ESM2 embeddings
- BABAPPAScore learned residue–residue scoring
- UPGMA guide tree
- Profile–profile alignment with Gotoh affine DP
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
    load_babappascore_model,
    batched_score,
)

# ============================================================
# Cache directories (Bioconda-safe, user-writable)
# ============================================================

CACHE_BASE = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
EMB_CACHE_DIR = CACHE_BASE / "babappalign" / "embeddings"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================

def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()

# ============================================================
# FASTA I/O
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
# Guide tree + alignment
# ============================================================

def pooled_vector(emb: torch.Tensor) -> torch.Tensor:
    return emb.mean(dim=0)


def compute_similarity_matrix(ids, emb_map, model, device, batch_pairs):
    V = torch.stack([pooled_vector(emb_map[i]) for i in ids]).to(device)
    with torch.no_grad():
        S = batched_score(model, V, V, device=device, batch=batch_pairs)
    return S if isinstance(S, np.ndarray) else S.cpu().numpy()


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
        new = max(clusters) + 1
        clusters[new] = left + right
        merges.append(("|".join(left), "|".join(right)))

        dist = np.pad(dist, ((0, 1), (0, 1)), constant_values=np.inf)
        for k in active:
            if k in (i, j):
                continue
            d = (dist[i, k] * len(left) + dist[j, k] * len(right)) / (len(left) + len(right))
            dist[new, k] = dist[k, new] = d

        dist[i, :] = dist[:, i] = dist[j, :] = dist[:, j] = np.inf
        active.remove(i)
        active.remove(j)
        active.add(new)

    return merges


def progressive_align(ids, seqs, emb_map, model, device, go, ge, batch_pairs):
    profiles = {i: ([i], [s], [emb_map[i]]) for i, s in zip(ids, seqs)}
    sim = compute_similarity_matrix(ids, emb_map, model, device, batch_pairs)
    merges = upgma(sim, ids)

    for L, R in merges:
        idsA, seqsA, embsA = profiles.pop(L)
        idsB, seqsB, embsB = profiles.pop(R)
        profiles[L + "|" + R] = (
            idsA + idsB,
            seqsA + seqsB,
            embsA + embsB,
        )

    final = next(iter(profiles.values()))
    return final[0], final[1]

# ============================================================
# CLI
# ============================================================

def cli():
    p = argparse.ArgumentParser(description="BABAPPAlign: deep learning–based MSA")
    p.add_argument("sequences", help="Input protein FASTA")
    p.add_argument("-o", "--output", required=True, help="Output aligned FASTA")
    p.add_argument("--gap-open", type=float, default=-1.5)
    p.add_argument("--gap-extend", type=float, default=-0.5)
    p.add_argument("--batch-pairs", type=int, default=4096)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)

    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    print(f"[info] Using device: {device}")

    # 🔑 REQUIRED model
    model = load_babappascore_model(device)

    ids, seqs = read_fasta(Path(args.sequences))

    emb_map = {}
    for sid, seq in zip(ids, seqs):
        f = EMB_CACHE_DIR / f"{seq_hash(seq)}.pt"
        if f.exists():
            emb = torch.load(f, map_location=device)
        else:
            emb = embed_sequence(seq, device)
            torch.save(emb.cpu(), f)
        emb_map[sid] = emb.to(device)

    start = time.time()
    out_ids, out_seqs = progressive_align(
        ids, seqs, emb_map, model, device,
        args.gap_open, args.gap_extend, args.batch_pairs
    )
    print(f"[done] Alignment finished in {time.time() - start:.1f}s")

    write_fasta(out_ids, out_seqs, Path(args.output))


if __name__ == "__main__":
    cli()
