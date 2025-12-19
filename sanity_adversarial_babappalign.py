#!/usr/bin/env python3
"""
Adversarial sanity checks for BABAPPAlign.

Goals:
- Stress alignment core correctness
- Catch DP / traceback / profile bugs
- Ensure learned scorer is always exercised
- Mimic real conda-forge usage (CLI only)
"""

import subprocess
import random
import string
from pathlib import Path
import sys

AA = "ACDEFGHIKLMNPQRSTVWY"
OUT = Path("adversarial_runs")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

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


def run_babappalign(name, seqs):
    fasta = OUT / f"{name}.fasta"
    out = OUT / f"{name}.baba.aln"

    write_fasta(fasta, seqs)

    try:
        subprocess.run(
            ["babappalign", fasta, "-o", out, "--device", "cpu"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ CRASH in test: {name}")
        print(e.stderr)
        return None

    return read_fasta(out)


def assert_same_length(aln):
    lengths = {len(s) for s in aln.values()}
    assert len(lengths) == 1, f"Unequal alignment lengths: {lengths}"


def gap_fraction(aln):
    total = sum(len(s) for s in aln.values())
    gaps = sum(s.count("-") for s in aln.values())
    return gaps / total


# ---------------------------------------------------------
# Test cases
# ---------------------------------------------------------

TESTS = {}

# 1. Identity (must be exact)
TESTS["identity"] = {
    "a": "MKTAYIAKQRQISFVKSHF",
    "b": "MKTAYIAKQRQISFVKSHF",
}

# 2. Single mismatch
TESTS["single_mismatch"] = {
    "a": "MKTAYIAKQRQISFVKSHF",
    "b": "MKTAYIAKQRQISFVKSKF",
}

# 3. Leading / trailing gaps
TESTS["terminal_indels"] = {
    "a": "MKTAYIAKQRQISFVKSHF",
    "b": "----YIAKQRQISFVKSHF",
    "c": "MKTAYIAKQRQISFVK----",
}

# 4. Internal gap stress
TESTS["internal_gaps"] = {
    "a": "MKTAYIAKQRQISFVKSHF",
    "b": "MKTA----KQRQISFVKSHF",
    "c": "MKTAYIAK---ISFVKSHF",
}

# 5. Length imbalance (profile vs sequence)
TESTS["length_imbalance"] = {
    "short": "MKTAYI",
    "long":  "MKTAYIAKQRQISFVKSHFSRQDILDL",
}

# 6. Low complexity (alignment collapse risk)
TESTS["low_complexity"] = {
    "a": "AAAAAAAAAAAAAAAAAAAA",
    "b": "AAAAAAAAGAAAAAAAAAAA",
    "c": "AAAAAAAAAAAAAAA",
}

# 7. Random divergent proteins
TESTS["random_divergent"] = {
    f"s{i}": "".join(random.choice(AA) for _ in range(25))
    for i in range(4)
}

# 8. Profile amplification (order sensitivity)
TESTS["profile_growth"] = {
    "s1": "MKTAYIAKQRQISFVKSHF",
    "s2": "MKTAYIAKQ--ISFVKSHF",
    "s3": "MKT--IAKQRQISFVKSHF",
    "s4": "MKTAYIAKQRQISFVKSH-",
}

# 9. All-gap temptation (should not explode)
TESTS["gap_heavy"] = {
    "a": "MKTAYIAKQRQISFVKSHF",
    "b": "-------------------",
}

# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

print("\n=== BABAPPAlign ADVERSARIAL SANITY TESTS ===\n")

FAILED = False

for name, seqs in TESTS.items():
    print(f"▶ TEST: {name}")

    aln = run_babappalign(name, seqs)
    if aln is None:
        FAILED = True
        continue

    try:
        assert_same_length(aln)
    except AssertionError as e:
        print(f"  ❌ FAIL: {e}")
        FAILED = True
        continue

    gf = gap_fraction(aln)
    print(f"  Gap fraction: {gf:.3f}")

    # Strong expectations
    if name == "identity":
        if len(set(aln.values())) != 1:
            print("  ❌ FAIL: identity alignment altered")
            FAILED = True
        else:
            print("  ✅ PASS (identity preserved)")
    else:
        print("  ✅ PASS (structure valid)")

print("\n=== SUMMARY ===")
if FAILED:
    print("❌ Some adversarial tests FAILED")
    sys.exit(1)
else:
    print("✅ All adversarial tests PASSED")
    sys.exit(0)

