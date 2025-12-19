#!/usr/bin/env python3
"""
Ultra-robust adversarial comparison of BABAPPAlign vs MAFFT
===========================================================

This script evaluates BABAPPAlign against MAFFT across an
extensive battery of adversarial and pathological MSA scenarios.

Outputs:
- compare_runs/ultrarobust_comparison.tsv
- compare_runs/ultrarobust_comparison.log
"""

import subprocess
import random
from pathlib import Path
from itertools import combinations
from datetime import datetime

AA = "ACDEFGHIKLMNPQRSTVWY"
OUT = Path("compare_runs")
OUT.mkdir(exist_ok=True)

TSV = OUT / "ultrarobust_comparison.tsv"
LOG = OUT / "ultrarobust_comparison.log"


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def write_fasta(path, seqs):
    with open(path, "w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")


def read_fasta(path):
    seqs, cur = {}, None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"):
            cur = l[1:]
            seqs[cur] = ""
        else:
            seqs[cur] += l
    return seqs


def gap_fraction(aln):
    total = sum(len(s) for s in aln.values())
    gaps = sum(s.count("-") for s in aln.values())
    return gaps / total


def pairwise_identity(a, b):
    matches, aligned = 0, 0
    for x, y in zip(a, b):
        if x == "-" or y == "-":
            continue
        aligned += 1
        if x == y:
            matches += 1
    return matches / aligned if aligned else 0.0


def mean_pairwise_identity(aln):
    ids = list(aln.keys())
    vals = [
        pairwise_identity(aln[i], aln[j])
        for i, j in combinations(ids, 2)
    ]
    return sum(vals) / len(vals) if vals else 1.0


def classify(delta_gap, delta_id):
    if abs(delta_gap) < 0.01 and abs(delta_id) < 0.01:
        return "IDENTICAL"
    if delta_gap > 0.05:
        return "BABA-more-gappy"
    if delta_gap < -0.05:
        return "BABA-more-conservative"
    return "MAFFT-like"


# -------------------------------------------------
# Adversarial test generator
# -------------------------------------------------

def generate_tests():
    tests = {}

    # 1. Identity
    tests["identity"] = {
        "a": "MKTAYIAKQRQISFVKSHF",
        "b": "MKTAYIAKQRQISFVKSHF",
    }

    # 2. Extreme length imbalance
    tests["extreme_length"] = {
        "short": "MKTAYI",
        "long": "MKTAYIAKQRQISFVKSHFSRQDILDL",
    }

    # 3. All-gap bait
    tests["all_gap_bait"] = {
        "a": "MKTAYIAKQRQISFVKSHF",
        "b": "-------------------",
    }

    # 4. Single residue anchor
    tests["single_anchor"] = {
        "a": "-----------------W",
        "b": "MKTAYIAKQRQISFVKSHFW",
    }

    # 5. Motif displacement
    tests["motif_shift"] = {
        "a": "AAAAAMKTAYBBBBB",
        "b": "BBBBBMKTAYAAAAA",
    }

    # 6. Repeats with phase shift
    tests["repeat_phase"] = {
        "a": "ABABABABABAB",
        "b": "BABABABABABA",
    }

    # 7. Scrambled order
    base = "MKTAYIAKQRQISFVKSHF"
    scrambled = "".join(random.sample(base, len(base)))
    tests["scrambled"] = {
        "a": base,
        "b": scrambled,
    }

    # 8. Gap storm
    tests["gap_storm"] = {
        "a": "MKTAYIAKQRQISFVKSHF",
        "b": "M--T--Y--I--A--K--Q--R--",
        "c": "--MKTAYIAKQRQISFVKSHF--",
    }

    # 9. Near identity with noise
    noisy = list(base)
    for i in random.sample(range(len(noisy)), 3):
        noisy[i] = random.choice(AA)
    tests["near_identity_noise"] = {
        "a": base,
        "b": "".join(noisy),
    }

    # 10. Random null control
    tests["random_null"] = {
        f"s{i}": "".join(random.choice(AA) for _ in range(30))
        for i in range(4)
    }

    return tests


# -------------------------------------------------
# Run comparison
# -------------------------------------------------

tests = generate_tests()

with open(TSV, "w") as tsv, open(LOG, "w") as log:
    tsv.write("test\tnseq\tmethod\taln_len\tgap_fraction\tmean_pairwise_identity\n")
    log.write("Ultra-robust BABAPPAlign vs MAFFT comparison\n")
    log.write(f"Run date: {datetime.now()}\n\n")

    for name, seqs in tests.items():
        log.write(f"=== TEST: {name} ===\n")

        fasta = OUT / f"{name}.fasta"
        mafft_out = OUT / f"{name}.mafft.aln"
        baba_out = OUT / f"{name}.baba.aln"

        write_fasta(fasta, seqs)

        subprocess.run(
            ["mafft", "--auto", fasta],
            stdout=open(mafft_out, "w"),
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["babappalign", fasta, "-o", baba_out, "--device", "cpu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        ma = read_fasta(mafft_out)
        ba = read_fasta(baba_out)

        for label, aln in [("MAFFT", ma), ("BABAPPAlign", ba)]:
            tsv.write(
                f"{name}\t{len(aln)}\t{label}\t"
                f"{len(next(iter(aln.values())))}\t"
                f"{gap_fraction(aln):.4f}\t"
                f"{mean_pairwise_identity(aln):.4f}\n"
            )

        dg = gap_fraction(ba) - gap_fraction(ma)
        di = mean_pairwise_identity(ba) - mean_pairwise_identity(ma)
        tag = classify(dg, di)

        log.write(f"Δgap={dg:+.3f}, Δid={di:+.3f} → {tag}\n\n")

    log.write("=== END ULTRA-ROBUST COMPARISON ===\n")

print("✔ Ultra-robust comparison completed")
print(f"  - {TSV}")
print(f"  - {LOG}")
