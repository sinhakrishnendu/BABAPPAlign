#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BABAPPAlign v1.2.0
==================

Embedding-first progressive multiple sequence alignment engine
with true affine-gap dynamic programming (Gotoh).

Modes:
    protein (default)
    codon   (CDS → translate → align → backmap)

Outputs:
    Protein mode:
        <input>.protein.aln.fasta

    Codon mode:
        <input>.protein.aln.fasta
        <input>.codon.aln.fasta

STRICT learned residue-level scoring is mandatory.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Silence HuggingFace warnings only
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore",
    message="Some weights of .* were not initialized"
)

# ============================================================
# STRICT SCORER ENFORCEMENT
# ============================================================

from babappalign.babappascore import (
    embed_sequence,
    batched_score,
    safe_load_model,
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
    if os.path.sep in model_arg or model_arg.startswith("."):
        path = Path(model_arg).expanduser().resolve()
    else:
        if not model_arg.endswith(".pt"):
            model_arg += ".pt"
        path = get_cache_dir("models") / model_arg

    if not path.is_file():
        raise FileNotFoundError(f"[FATAL] Scoring model not found: {path}")

    return path


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


# ============================================================
# FASTA I/O
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
            fh.write(f">{i}\n")
            for j in range(0, len(s), 60):
                fh.write(s[j:j+60] + "\n")



# ============================================================
# CODON UTILITIES
# ============================================================

STOP_CODONS = {"TAA", "TAG", "TGA"}

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','CAT':'H','CAC':'H',
    'CAA':'Q','CAG':'Q','AAT':'N','AAC':'N',
    'AAA':'K','AAG':'K','GAT':'D','GAC':'D',
    'GAA':'E','GAG':'E','TGT':'C','TGC':'C',
    'TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}


def validate_cds(seq: str, sid: str):
    if len(seq) % 3 != 0:
        raise ValueError(f"[{sid}] CDS length not multiple of 3.")
    if not set(seq).issubset({"A","T","G","C","N"}):
        raise ValueError(f"[{sid}] Invalid nucleotide detected.")
    for i in range(0, len(seq) - 3, 3):
        if seq[i:i+3] in STOP_CODONS:
            raise ValueError(f"[{sid}] Internal stop codon detected.")


def translate_cds(seq: str) -> str:
    return "".join(CODON_TABLE.get(seq[i:i+3], "X")
                   for i in range(0, len(seq), 3))


def backmap_to_codon_alignment(aligned_prot: str, original_cds: str) -> str:
    codon_aln = []
    ptr = 0
    for aa in aligned_prot:
        if aa == "-":
            codon_aln.append("---")
        else:
            codon_aln.append(original_cds[ptr:ptr+3])
            ptr += 3
    return "".join(codon_aln)


# ============================================================
# PROFILE CLASS (ORIGINAL)
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
# DEVICE
# ============================================================

def resolve_device(user_choice):
    if user_choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# COLUMN EMBEDDINGS (ORIGINAL)
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
# PROFILE–SEQUENCE SCORE MATRIX (ORIGINAL)
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
# GOTHOH DP (ORIGINAL, UNCHANGED)
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

    new_seqs.append("".join(reversed(new_seq)))
    new_idxs.append(list(reversed(new_idx)))

    return Profile(profile.ids + [sid], new_seqs, new_idxs)


# ============================================================
# PROGRESSIVE ALIGNMENT (ORIGINAL)
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
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["protein", "codon"], default="protein")
    p.add_argument("--gap-open", type=float, default=-2.5)
    p.add_argument("--gap-extend", type=float, default=-0.7)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    input_path = Path(args.fasta)
    stem = input_path.stem

    protein_out = input_path.with_name(f"{stem}.protein.aln.fasta")
    codon_out = input_path.with_name(f"{stem}.codon.aln.fasta")

    ids, raw_seqs = read_fasta(input_path)

    cds_map: Dict[str, str] = {}
    seqs: List[str] = []

    if args.mode == "protein":
        seqs = raw_seqs
    else:
        print("[BABAPPAlign] Codon mode enabled.")
        for sid, cds in zip(ids, raw_seqs):
            cds = cds.upper().replace("U", "T")
            validate_cds(cds, sid)
            cds_map[sid] = cds
            seqs.append(translate_cds(cds))
        args.gap_open *= 3
        args.gap_extend *= 3

    device = resolve_device(args.device)
    model = safe_load_model(str(resolve_model_path(args.model)), device)

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

    write_fasta(out_ids, out_seqs, protein_out)
    print(f"[BABAPPAlign] Protein alignment written: {protein_out}")

    if args.mode == "codon":
        codon_aligned = [
            backmap_to_codon_alignment(aln, cds_map[sid])
            for sid, aln in zip(out_ids, out_seqs)
        ]
        write_fasta(out_ids, codon_aligned, codon_out)
        print(f"[BABAPPAlign] Codon alignment written: {codon_out}")


if __name__ == "__main__":
    main()
