#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Embedding-first progressive multiple sequence alignment engine with:

- True affine-gap DP (Gotoh)
- Learned residue–residue scoring
- Distance-based Neighbor Joining guide tree
- Residue-level bootstrap + majority-rule consensus
- Optimized symmetric profile–profile alignment

Guide tree is a computational heuristic, not a phylogeny.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import Counter

import numpy as np
import torch


# ============================================================
# Profile
# ============================================================

class Profile:
    def __init__(self, ids, seqs, idxs=None):
        self.ids = list(ids)
        self.seqs = list(seqs)
        self.length = len(seqs[0])
        self.idxs = idxs if idxs is not None else [
            list(range(self.length)) for _ in seqs
        ]

    def __len__(self):
        return len(self.seqs)


# ============================================================
# FASTA
# ============================================================

def read_fasta(path: Path):
    ids, seqs, cur, buf = [], [], None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                ids.append(cur)
                seqs.append("".join(buf))
            cur = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if cur:
        ids.append(cur)
        seqs.append("".join(buf))
    return ids, seqs


def write_fasta(ids, seqs, path: Path):
    with open(path, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n{s}\n")


# ============================================================
# Embeddings & distances
# ============================================================

def sequence_embedding(res_emb: torch.Tensor) -> torch.Tensor:
    return res_emb.mean(dim=0)


def cosine_distance_matrix(seq_embs: Dict[str, torch.Tensor]):
    ids = list(seq_embs.keys())
    X = torch.stack([seq_embs[i] for i in ids])
    X = X / X.norm(dim=1, keepdim=True)
    return (1.0 - X @ X.T).cpu().numpy(), ids


# ============================================================
# Neighbor Joining (FIXED)
# ============================================================

Tree = Union[str, Tuple["Tree", "Tree"]]


def neighbor_joining(D: np.ndarray, labels: List[str]) -> Tree:
    D = D.copy()
    clusters = {i: labels[i] for i in range(len(labels))}
    active = list(range(len(labels)))

    while len(active) > 2:
        n = len(active)
        r = {i: sum(D[i, j] for j in active) for i in active}

        i, j = min(
            ((i, j) for i in active for j in active if i < j),
            key=lambda x: (n - 2) * D[x[0], x[1]] - r[x[0]] - r[x[1]]
        )

        new = max(clusters) + 1
        clusters[new] = (clusters[i], clusters[j])
        D = np.pad(D, ((0, 1), (0, 1)))

        for k in active:
            if k != i and k != j:
                D[new, k] = D[k, new] = 0.5 * (
                    D[i, k] + D[j, k] - D[i, j]
                )

        active = [x for x in active if x not in (i, j)]
        active.append(new)

    i, j = active
    return (clusters[i], clusters[j])


# ============================================================
# Bootstrap + consensus
# ============================================================

def get_splits(tree: Tree):
    splits = []

    def leaves(n):
        if isinstance(n, str):
            return {n}
        return leaves(n[0]) | leaves(n[1])

    allL = leaves(tree)

    def walk(n):
        if isinstance(n, str):
            return
        L = leaves(n[0])
        if 0 < len(L) < len(allL):
            splits.append(frozenset(L))
        walk(n[0])
        walk(n[1])

    walk(tree)
    return splits


def bootstrap_consensus(emb_map, n_boot):
    counts = Counter()
    ids = list(emb_map.keys())

    for _ in range(n_boot):
        boot = {
            k: v[torch.randint(0, v.shape[0], (v.shape[0],))].mean(0)
            for k, v in emb_map.items()
        }
        D, order = cosine_distance_matrix(boot)
        t = neighbor_joining(D, order)
        counts.update(get_splits(t))

    valid = [s for s, c in counts.items() if c / n_boot >= 0.5]
    valid.sort(key=len, reverse=True)

    def build(S):
        for s in valid:
            if s < S:
                return (build(s), build(S - s))
        return next(iter(S))

    return build(set(ids))


# ============================================================
# Profile–profile scoring (BATCHED)
# ============================================================

def compute_profile_profile_scores(A: Profile, B: Profile,
                                   emb_map, model, device):
    from babappalign.babappascore import batched_score

    EA, EB, idxA, idxB = [], [], [], []

    for i in range(A.length):
        col = []
        for sid, idxs in zip(A.ids, A.idxs):
            pos = idxs[i]
            if isinstance(pos, int) and pos >= 0:
                col.append(emb_map[sid][pos])
        if col:
            EA.append(torch.stack(col).mean(0))
            idxA.append(i)

    for j in range(B.length):
        col = []
        for sid, idxs in zip(B.ids, B.idxs):
            pos = idxs[j]
            if isinstance(pos, int) and pos >= 0:
                col.append(emb_map[sid][pos])
        if col:
            EB.append(torch.stack(col).mean(0))
            idxB.append(j)

    if not EA or not EB:
        return np.zeros((A.length, B.length))

    Aemb = torch.stack(EA).to(device)   # [m, d]
    Bemb = torch.stack(EB).to(device)   # [n, d]

    S = batched_score(model, Aemb, Bemb, device)

    M = np.zeros((A.length, B.length))
    for ii, i in enumerate(idxA):
        for jj, j in enumerate(idxB):
            M[i, j] = S[ii, jj]

    return M



# ============================================================
# TRUE AFFINE GAP PROFILE–PROFILE DP (Gotoh)
# ============================================================

def align_profiles(A: Profile, B: Profile,
                   emb_map, model, device, go, ge):
    S = compute_profile_profile_scores(A, B, emb_map, model, device)

    LA, LB = A.length, B.length
    NEG = -1e12

    M = np.full((LA + 1, LB + 1), NEG)
    X = np.full_like(M, NEG)
    Y = np.full_like(M, NEG)

    TM = np.zeros_like(M, dtype=np.int8)
    TX = np.zeros_like(M, dtype=np.int8)
    TY = np.zeros_like(M, dtype=np.int8)

    M[0, 0] = 0.0

    for i in range(1, LA + 1):
        X[i, 0] = go + (i - 1) * ge
        TX[i, 0] = 1
    for j in range(1, LB + 1):
        Y[0, j] = go + (j - 1) * ge
        TY[0, j] = 2

    for i in range(1, LA + 1):
        for j in range(1, LB + 1):
            prev = [M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1]]
            k = int(np.argmax(prev))
            M[i, j] = prev[k] + S[i-1, j-1]
            TM[i, j] = k

            if M[i-1, j] + go >= X[i-1, j] + ge:
                X[i, j] = M[i-1, j] + go
                TX[i, j] = 0
            else:
                X[i, j] = X[i-1, j] + ge
                TX[i, j] = 1

            if M[i, j-1] + go >= Y[i, j-1] + ge:
                Y[i, j] = M[i, j-1] + go
                TY[i, j] = 0
            else:
                Y[i, j] = Y[i, j-1] + ge
                TY[i, j] = 2

    i, j = LA, LB
    state = int(np.argmax([M[i, j], X[i, j], Y[i, j]]))

    new_ids = A.ids + B.ids
    new_seqs = [[] for _ in new_ids]
    new_idxs = [[] for _ in new_ids]

    while i > 0 or j > 0:
        if state == 0:
            prev = TM[i, j]
            for k in range(len(A)):
                new_seqs[k].append(A.seqs[k][i-1])
                new_idxs[k].append(A.idxs[k][i-1])
            for k in range(len(B)):
                new_seqs[len(A)+k].append(B.seqs[k][j-1])
                new_idxs[len(A)+k].append(B.idxs[k][j-1])
            i -= 1; j -= 1
            state = prev

        elif state == 1:
            prev = TX[i, j]
            for k in range(len(A)):
                new_seqs[k].append(A.seqs[k][i-1])
                new_idxs[k].append(A.idxs[k][i-1])
            for k in range(len(B)):
                new_seqs[len(A)+k].append("-")
                new_idxs[len(A)+k].append(-1)
            i -= 1
            state = prev

        else:
            prev = TY[i, j]
            for k in range(len(A)):
                new_seqs[k].append("-")
                new_idxs[k].append(-1)
            for k in range(len(B)):
                new_seqs[len(A)+k].append(B.seqs[k][j-1])
                new_idxs[len(A)+k].append(B.idxs[k][j-1])
            j -= 1
            state = prev

    new_seqs = ["".join(reversed(s)) for s in new_seqs]
    new_idxs = [list(reversed(x)) for x in new_idxs]

    return Profile(new_ids, new_seqs, new_idxs)


# ============================================================
# Tree-guided alignment
# ============================================================

def align_tree(node, seq_map, emb_map, model, device, go, ge):
    if isinstance(node, str):
        return Profile([node], [seq_map[node]])
    A = align_tree(node[0], seq_map, emb_map, model, device, go, ge)
    B = align_tree(node[1], seq_map, emb_map, model, device, go, ge)
    return align_profiles(A, B, emb_map, model, device, go, ge)


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("fasta")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    # ---- robust device resolution
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from babappalign.babappascore import embed_sequence, safe_load_model

    ids, seqs = read_fasta(Path(args.fasta))
    seq_map = dict(zip(ids, seqs))

    emb_map = {
        sid: embed_sequence(seq, device)
        for sid, seq in zip(ids, seqs)
    }

    model = safe_load_model(args.model, device)

    seq_embs = {sid: sequence_embedding(emb)
                for sid, emb in emb_map.items()}
    D, order = cosine_distance_matrix(seq_embs)
    tree = neighbor_joining(D, order)

    if args.bootstrap > 0:
        tree = bootstrap_consensus(emb_map, args.bootstrap)

    profile = align_tree(
        tree, seq_map, emb_map, model, device,
        args.gap_open, args.gap_extend
    )

    write_fasta(profile.ids, profile.seqs, Path(args.output))


if __name__ == "__main__":
    main()
