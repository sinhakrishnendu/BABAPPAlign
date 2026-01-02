#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign
===========

Embedding-first progressive multiple sequence alignment engine
with true affine-gap dynamic programming (Gotoh).

STRICT learned residue-level scoring is mandatory.
"""

from __future__ import annotations

import argparse
import os
import hashlib
from pathlib import Path

import numpy as np
import torch


# ============================================================
# STRICT SCORER ENFORCEMENT (MANDATORY)
# ============================================================

try:
    from babappalign.babappascore import (
        embed_sequence,
        batched_score,
        safe_load_model,
    )
except Exception as e:
    raise RuntimeError(
        "\n[FATAL] babappascorer.py is REQUIRED but could not be loaded.\n"
        "BABAPPAlign will NOT fall back to any other scorer.\n\n"
        f"Original error:\n{e}\n"
    )


# ============================================================
# Cache helpers
# ============================================================

def get_cache_dir(subdir: str) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "babappalign" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_model_path(model_arg: str) -> Path:
    if model_arg is None:
        raise RuntimeError(
            "\n[FATAL] --model is mandatory.\n"
            "BABAPPAlign does not provide any default or fallback scoring model.\n"
        )

    if os.path.sep in model_arg or model_arg.startswith("."):
        path = Path(model_arg).expanduser().resolve()
    else:
        name = model_arg
        if not name.endswith(".pt"):
            name += ".pt"
        path = get_cache_dir("models") / name

    if not path.is_file():
        raise FileNotFoundError(
            f"\n[FATAL] Scoring model not found:\n"
            f"  {path}\n\n"
            f"Download the model from Zenodo and place it in:\n"
            f"  {get_cache_dir('models')}\n"
        )
    return path


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


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
# Device
# ============================================================

def resolve_device(user_choice):
    if user_choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
# Column embeddings
# ============================================================

def compute_column_embeddings(profile: Profile, emb_map):
    col_embs = []
    conf = []

    for i in range(profile.length):
        embs = []
        nongap = 0
        for sid, seq, idxs in zip(profile.ids, profile.seqs, profile.idxs):
            if seq[i] != "-":
                embs.append(emb_map[sid][idxs[i]])
                nongap += 1

        if embs:
            col_embs.append(torch.stack(embs).mean(dim=0))
            conf.append(max(nongap / len(profile), 0.3))
        else:
            col_embs.append(None)
            conf.append(0.0)

    return col_embs, np.asarray(conf, dtype=float)


# ============================================================
# Profile–sequence score matrix (NEURAL ONLY)
# ============================================================

def compute_profile_seq_score_matrix(profile, sid, emb_map, model, device):
    col_embs, conf = compute_column_embeddings(profile, emb_map)
    valid = [i for i, e in enumerate(col_embs) if e is not None]

    if not valid:
        return np.zeros((profile.length, emb_map[sid].shape[0]), dtype=float)

    A = torch.stack([col_embs[i] for i in valid]).to(device)
    B = emb_map[sid].to(device)

    S = batched_score(model, A, B, device)

    M = np.full((profile.length, B.shape[0]), -0.7, dtype=float)
    for k, i in enumerate(valid):
        M[i, :] = S[k] * conf[i]

    return M


# ============================================================
# TRUE AFFINE GAP DP (Gotoh)
# ============================================================

def nw_align_profile_seq(profile, sid, seq, emb_map, model, device,
                         gap_open, gap_extend):

    M_score = compute_profile_seq_score_matrix(
        profile, sid, emb_map, model, device
    )

    m, n = profile.length, len(seq)
    NEG = -1e12

    M = np.full((m + 1, n + 1), NEG)
    Ix = np.full((m + 1, n + 1), NEG)
    Iy = np.full((m + 1, n + 1), NEG)

    TM = np.zeros((m + 1, n + 1), dtype=np.int8)
    TX = np.zeros((m + 1, n + 1), dtype=np.int8)
    TY = np.zeros((m + 1, n + 1), dtype=np.int8)

    M[0, 0] = 0.0

    for i in range(1, m + 1):
        Ix[i, 0] = gap_open + (i - 1) * gap_extend
        TX[i, 0] = 1

    for j in range(1, n + 1):
        Iy[0, j] = gap_open + (j - 1) * gap_extend
        TY[0, j] = 2

    for i in range(1, m + 1):
        for j in range(1, n + 1):

            prev = [M[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1]]
            k = int(np.argmax(prev))
            M[i, j] = prev[k] + M_score[i-1, j-1]
            TM[i, j] = k

            if M[i-1, j] + gap_open >= Ix[i-1, j] + gap_extend:
                Ix[i, j] = M[i-1, j] + gap_open
                TX[i, j] = 0
            else:
                Ix[i, j] = Ix[i-1, j] + gap_extend
                TX[i, j] = 1

            if M[i, j-1] + gap_open >= Iy[i, j-1] + gap_extend:
                Iy[i, j] = M[i, j-1] + gap_open
                TY[i, j] = 0
            else:
                Iy[i, j] = Iy[i, j-1] + gap_extend
                TY[i, j] = 2

    state = int(np.argmax([M[m, n], Ix[m, n], Iy[m, n]]))

    new_seqs = [[] for _ in profile.seqs]
    new_idxs = [[] for _ in profile.seqs]
    new_seq, new_idx = [], []

    i, j = m, n
    while i > 0 or j > 0:
        if state == 0:
            prev = TM[i, j]
            for k, (s, idxs) in enumerate(zip(profile.seqs, profile.idxs)):
                new_seqs[k].append(s[i-1])
                new_idxs[k].append(idxs[i-1])
            new_seq.append(seq[j-1])
            new_idx.append(j-1)
            i -= 1
            j -= 1
            state = prev
        elif state == 1:
            prev = TX[i, j]
            for k, (s, idxs) in enumerate(zip(profile.seqs, profile.idxs)):
                new_seqs[k].append(s[i-1])
                new_idxs[k].append(idxs[i-1])
            new_seq.append("-")
            new_idx.append(None)
            i -= 1
            state = prev
        else:
            prev = TY[i, j]
            for k in range(len(profile)):
                new_seqs[k].append("-")
                new_idxs[k].append(None)
            new_seq.append(seq[j-1])
            new_idx.append(j-1)
            j -= 1
            state = prev

    new_seqs = ["".join(reversed(s)) for s in new_seqs]
    new_idxs = [list(reversed(x)) for x in new_idxs]

    # CRITICAL FIX: append aligned new sequence
    new_seqs.append("".join(reversed(new_seq)))
    new_idxs.append(list(reversed(new_idx)))

    return Profile(profile.ids + [sid], new_seqs, new_idxs)


# ============================================================
# Progressive
# ============================================================

def progressive_align(ids, seqs, emb_map, model, device, gap_open, gap_extend):
    prof = Profile([ids[0]], [seqs[0]])
    for sid, seq in zip(ids[1:], seqs[1:]):
        prof = nw_align_profile_seq(
            prof, sid, seq, emb_map, model, device,
            gap_open, gap_extend
        )
    return prof.ids, prof.seqs


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("fasta")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", required=True,
                   help="Scoring model name or explicit path (.pt).")
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    if args.model is None:
        raise RuntimeError("[FATAL] --model is mandatory.")

    ids, seqs = read_fasta(Path(args.fasta))

    device = resolve_device(args.device)
    print(f"[BABAPPAlign] Using device: {device}")

    model_path = resolve_model_path(args.model)
    print("[BABAPPAlign] Using scoring model:")
    print(f"  Path    : {model_path}")
    print(f"  SHA-256 : {sha256sum(model_path)}")

    model = safe_load_model(str(model_path), device)

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

    out_ids, out_seqs = progressive_align(
        ids, seqs, emb_map, model, device,
        args.gap_open, args.gap_extend
    )

    write_fasta(out_ids, out_seqs, Path(args.output))


if __name__ == "__main__":
    main()
