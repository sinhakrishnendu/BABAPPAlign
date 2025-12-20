#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Embedding-first progressive multiple sequence alignment engine.

Optimized core:
- Learned residue–residue scoring (mandatory)
- Score-matrix precomputation (no repeated NN calls)
- Column-aware profile scoring
- Explicit gap-open vs gap-extend handling
- Conservative diagonal continuation prior
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


# =========================
# Profile
# =========================

class Profile:
    def __init__(self, ids: List[str], seqs: List[str], idxs=None):
        self.ids = list(ids)
        self.seqs = list(seqs)

        if idxs is None:
            self.idxs = [list(range(len(seqs[0]))) for _ in seqs]
        else:
            self.idxs = idxs

        self.length = len(seqs[0])

    def __len__(self):
        return len(self.seqs)

    def get_column(self, i):
        col = []
        for s, idx in zip(self.seqs, self.idxs):
            if s[i] == "-":
                col.append(None)
            else:
                col.append(idx[i])
        return col


# =========================
# Device
# =========================

def resolve_device(user_choice):
    import torch
    if user_choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# Cache
# =========================

def get_cache_dir(subdir: str) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "babappalign" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


# =========================
# FASTA
# =========================

def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    cur, buf = None, []
    with open(path) as fh:
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


def write_fasta(ids, seqs, path: Path):
    with open(path, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n{s}\n")


# =========================
# NW profile vs sequence (FAST)
# =========================

def nw_align_profile_seq(
    profile,
    sid,
    seq,
    emb_map,
    model,
    device,
    gap_open,
    gap_extend,
):
    from babappalign.babappascore import batched_score

    # Conservative priors
    DIAG_CONTINUATION_BONUS = 0.2
    COLUMN_CONFIDENCE_FLOOR = 0.3

    m, n = profile.length, len(seq)

    # -------------------------
    # PRECOMPUTE SCORE MATRIX
    # -------------------------
    # Flatten all non-gap profile residues
    prof_ids = []
    prof_pos = []
    for pid, idxs in zip(profile.ids, profile.idxs):
        for idx in idxs:
            if idx is not None:
                prof_ids.append(pid)
                prof_pos.append(idx)

    # Map flattened profile positions to row indices
    row_map = {}
    r = 0
    for i in range(m):
        col = profile.get_column(i)
        rows = []
        for idx in col:
            if idx is None:
                rows.append(None)
            else:
                rows.append(r)
                r += 1
        row_map[i] = rows

    # Build embedding matrices
    import torch
    E_prof = torch.stack(
        [emb_map[pid][idx] for pid, idx in zip(prof_ids, prof_pos)]
    )
    E_seq = emb_map[sid]

    # ONE neural call
    SCORE = batched_score(model, E_prof, E_seq, device)

    # -------------------------
    # DP matrices
    # -------------------------
    S = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    T = np.zeros((m + 1, n + 1), dtype=np.int8)

    S[0, 0] = 0.0
    for i in range(1, m + 1):
        S[i, 0] = gap_open + (i - 1) * gap_extend
        T[i, 0] = 1
    for j in range(1, n + 1):
        S[0, j] = gap_open + (j - 1) * gap_extend
        T[0, j] = 2

    # -------------------------
    # DP fill (FAST, no NN)
    # -------------------------
    for i in range(1, m + 1):
        rows = row_map[i - 1]
        valid = [r for r in rows if r is not None]

        for j in range(1, n + 1):
            if valid:
                raw = float(np.mean(SCORE[valid, j - 1]))
                conf = max(len(valid) / len(profile), COLUMN_CONFIDENCE_FLOOR)
                match_score = raw * conf
            else:
                match_score = gap_extend

            match = (
                S[i - 1, j - 1]
                + match_score
                + (DIAG_CONTINUATION_BONUS if T[i - 1, j - 1] == 0 else 0.0)
            )
            delete = S[i - 1, j] + (
                gap_extend if T[i - 1, j] == 1 else gap_open
            )
            insert = S[i, j - 1] + (
                gap_extend if T[i, j - 1] == 2 else gap_open
            )

            best = max(match, delete, insert)
            S[i, j] = best
            T[i, j] = 0 if best == match else (1 if best == delete else 2)

    # -------------------------
    # Traceback (unchanged)
    # -------------------------
    new_seqs = [""] * len(profile)
    new_idxs = [[] for _ in range(len(profile))]
    new_seq, new_idx = [], []

    i, j = m, n
    while i > 0 or j > 0:
        t = T[i, j]
        if t == 0:
            for k, idx in enumerate(profile.get_column(i - 1)):
                if idx is None:
                    new_seqs[k] = "-" + new_seqs[k]
                    new_idxs[k] = [None] + new_idxs[k]
                else:
                    new_seqs[k] = profile.seqs[k][i - 1] + new_seqs[k]
                    new_idxs[k] = [idx] + new_idxs[k]
            new_seq.append(seq[j - 1])
            new_idx.append(j - 1)
            i -= 1
            j -= 1
        elif t == 1:
            for k, idx in enumerate(profile.get_column(i - 1)):
                if idx is None:
                    new_seqs[k] = "-" + new_seqs[k]
                    new_idxs[k] = [None] + new_idxs[k]
                else:
                    new_seqs[k] = profile.seqs[k][i - 1] + new_seqs[k]
                    new_idxs[k] = [idx] + new_idxs[k]
            new_seq.append("-")
            new_idx.append(None)
            i -= 1
        else:
            for k in range(len(profile)):
                new_seqs[k] = "-" + new_seqs[k]
                new_idxs[k] = [None] + new_idxs[k]
            new_seq.append(seq[j - 1])
            new_idx.append(j - 1)
            j -= 1

    new_seqs.append("".join(reversed(new_seq)))
    new_idxs.append(list(reversed(new_idx)))

    return Profile(profile.ids + [sid], new_seqs, new_idxs)


# =========================
# Progressive
# =========================

def progressive_align(ids, seqs, emb_map, model, device, gap_open, gap_extend):
    prof = Profile([ids[0]], [seqs[0]])
    for sid, seq in zip(ids[1:], seqs[1:]):
        prof = nw_align_profile_seq(
            prof, sid, seq, emb_map, model, device, gap_open, gap_extend
        )
    return prof.ids, prof.seqs


# =========================
# CLI
# =========================

def cli():
    p = argparse.ArgumentParser()

    # Positional arguments (backward-compatible)
    p.add_argument("sequences", nargs="?", help="Input FASTA file")
    p.add_argument("output", nargs="?", help="Output FASTA file")

    # Optional flags (override positional)
    p.add_argument("-i", "--input", dest="input_flag",
                   help="Input FASTA file (flag form)")
    p.add_argument("-o", "--output", dest="output_flag",
                   help="Output FASTA file (flag form)")

    p.add_argument("--model", default=None)
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)

    args = p.parse_args()

    # Resolve inputs (flags override positional)
    sequences = args.input_flag or args.sequences
    output = args.output_flag or args.output

    if sequences is None or output is None:
        p.error("Input FASTA and output FASTA must be specified.")

    import torch
    from babappalign.babappascore import embed_sequence, safe_load_model

    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")

    ids, seqs = read_fasta(Path(sequences))

    emb_cache = get_cache_dir("embeddings")
    emb_map = {}
    for sid, seq in zip(ids, seqs):
        f = emb_cache / f"{seq_hash(seq)}.pt"
        if f.exists():
            emb = torch.load(f, map_location=device)
        else:
            emb = embed_sequence(seq, device)
            torch.save(emb.cpu(), f)
        emb_map[sid] = emb.to(device)

    model = safe_load_model(args.model, device)

    out_ids, out_seqs = progressive_align(
        ids, seqs, emb_map, model, device,
        args.gap_open, args.gap_extend
    )

    write_fasta(out_ids, out_seqs, Path(output))



if __name__ == "__main__":
    cli()
