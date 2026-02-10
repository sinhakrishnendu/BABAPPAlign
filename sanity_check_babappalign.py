#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity checker for BABAPPAlign codon mode.

Compares BABAPPAlign codon alignment
against PAL2NAL output automatically.

Usage:
    python sanity_check_babappalign.py input_cds.fasta --model babappascore
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# FASTA UTILITIES
# ============================================================

def read_fasta(path):
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
    return dict(zip(ids, seqs))


# ============================================================
# RUN COMMAND
# ============================================================

def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cds")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pal2nal", default="pal2nal.pl")
    args = parser.parse_args()

    cds_path = Path(args.cds)
    stem = cds_path.stem

    print("=== SANITY CHECK START ===")

    # --------------------------------------------------------
    # 1. Run BABAPPAlign codon mode
    # --------------------------------------------------------
    print("[1] Running BABAPPAlign codon mode...")
    run_cmd([
        "babappalign",
        str(cds_path),
        "--model", args.model,
        "--mode", "codon"
    ])

    protein_aln = cds_path.with_name(f"{stem}.protein.aln.fasta")
    codon_aln = cds_path.with_name(f"{stem}.codon.aln.fasta")

    if not protein_aln.exists() or not codon_aln.exists():
        print("[FAIL] BABAPPAlign outputs not found.")
        sys.exit(1)

    print("[OK] BABAPPAlign outputs detected.")

    # --------------------------------------------------------
    # 2. Run PAL2NAL
    # --------------------------------------------------------
    print("[2] Running PAL2NAL...")
    pal_out = cds_path.with_name(f"{stem}.pal2nal.codon.fasta")

    run_cmd([
        args.pal2nal,
        str(protein_aln),
        str(cds_path),
        "-output", "fasta"
    ])

    # PAL2NAL prints to stdout — capture manually
    with open(pal_out, "w") as fout:
        subprocess.run(
            [args.pal2nal, str(protein_aln), str(cds_path), "-output", "fasta"],
            stdout=fout,
            check=True
        )

    print("[OK] PAL2NAL completed.")

    # --------------------------------------------------------
    # 3. Load alignments
    # --------------------------------------------------------
    babap = read_fasta(codon_aln)
    pal = read_fasta(pal_out)

    print("[3] Comparing alignments...")

    if set(babap.keys()) != set(pal.keys()):
        print("[FAIL] Sequence IDs mismatch.")
        sys.exit(1)

    total_diff = 0
    identical = True

    for sid in babap:
        seq1 = babap[sid]
        seq2 = pal[sid]

        if len(seq1) != len(seq2):
            print(f"[FAIL] Length mismatch in {sid}")
            identical = False
            continue

        diff = sum(a != b for a, b in zip(seq1, seq2))
        total_diff += diff

        if diff > 0:
            identical = False
            print(f"[WARN] {sid}: {diff} mismatches")

        # frame sanity check
        if len(seq1) % 3 != 0:
            print(f"[FAIL] Frame broken in {sid}")
            identical = False

    # --------------------------------------------------------
    # 4. Report
    # --------------------------------------------------------
    print("\n=== RESULT SUMMARY ===")

    if identical:
        print("✅ Codon alignments IDENTICAL to PAL2NAL.")
    else:
        print("⚠ Differences detected.")
        print(f"Total mismatched positions: {total_diff}")

    print("=== SANITY CHECK END ===")


if __name__ == "__main__":
    main()
