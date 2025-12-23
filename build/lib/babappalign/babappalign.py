#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Embedding-first progressive multiple sequence alignment engine.

Core characteristics:
- Learned residue–residue scoring (mandatory)
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

    def __iter__(self):
        return iter(self.seqs)

    def get_column(self, i):
        col = []
        for s, idx in zip(self.seqs, self.idxs):
            if s[i] == "-":
                col.append(("-", None))
            else:
                col.append((s[i], idx[i]))
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
# NW profile vs sequence
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

    S = np.full((m + 1, n + 1), -1e9, dtype=float)
    T = np.zeros((m + 1, n + 1), dtype=np.int8)
    # T: 0 = diag, 1 = up (gap in seq), 2 = left (gap in profile)

    S[0, 0] = 0.0
    for i in range(1, m + 1):
        S[i, 0] = gap_open + (i - 1) * gap_extend
        T[i, 0] = 1
    for j in range(1, n + 1):
        S[0, j] = gap_open + (j - 1) * gap_extend
        T[0, j] = 2

    # DP fill
    for i in range(1, m + 1):
        col = profile.get_column(i - 1)
        for j in range(1, n + 1):
            scores = []
            nongap = 0

            for (c, idx), pid in zip(col, profile.ids):
                if c == "-":
                    continue
                nongap += 1
                e1 = emb_map[pid][idx]
                e2 = emb_map[sid][j - 1]
                scores.append(
                    batched_score(
                        model,
                        e1.unsqueeze(0),
                        e2.unsqueeze(0),
                        device,
                    )[0, 0]
                )

            if scores:
                raw = float(np.mean(scores))
                conf = max(nongap / len(profile), COLUMN_CONFIDENCE_FLOOR)
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

    # Traceback
    new_seqs = [""] * len(profile)
    new_idxs = [[] for _ in profile]
    new_seq, new_idx = [], []

    i, j = m, n
    while i > 0 or j > 0:
        t = T[i, j]
        if t == 0:
            for k, (c, idx) in enumerate(profile.get_column(i - 1)):
                new_seqs[k] = c + new_seqs[k]
                new_idxs[k] = [idx] + new_idxs[k]
            new_seq.append(seq[j - 1])
            new_idx.append(j - 1)
            i -= 1
            j -= 1
        elif t == 1:
            for k, (c, idx) in enumerate(profile.get_column(i - 1)):
                new_seqs[k] = c + new_seqs[k]
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
            prof,
            sid,
            seq,
            emb_map,
            model,
            device,
            gap_open,
            gap_extend,
        )
    return prof.ids, prof.seqs


# =========================
# CLI
# =========================

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("sequences")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = p.parse_args()

    import torch
    from babappalign.babappascore import embed_sequence, safe_load_model

    device = resolve_device(args.device)
    print(f"[info] Using device: {device}")

    ids, seqs = read_fasta(Path(args.sequences))

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

    # Model resolution is handled lazily inside safe_load_model()
    model = safe_load_model(args.model, device)

    out_ids, out_seqs = progressive_align(
        ids,
        seqs,
        emb_map,
        model,
        device,
        args.gap_open,
        args.gap_extend,
    )

    write_fasta(out_ids, out_seqs, Path(args.output))


if __name__ == "__main__":
    cli()
