#!/usr/bin/env python3

import subprocess
import random
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
random.seed(42)

OUT = Path("sanity_runs")
OUT.mkdir(exist_ok=True)


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


TESTS = {
    "identity": {
        "a": "MKTAYIAKQRQISFVKSHF",
        "b": "MKTAYIAKQRQISFVKSHF",
    },
    "gap_stress": {
        "a": "MKTAYIAKQRQISFVKSHF",
        "b": "MKTA--AKQRQ--FVKSHF",
        "c": "MKTAYIAK---ISFVKSHF",
    },
    "divergent": {
        f"s{i}": "".join(random.choice(AA) for _ in range(20))
        for i in range(4)
    },
}


print("\n=== BABAPPAlign SANITY VALIDATION vs MAFFT ===\n")

for name, seqs in TESTS.items():
    print(f"▶ TEST: {name}")

    fasta = OUT / f"{name}.fasta"
    mafft_out = OUT / f"{name}.mafft.aln"
    baba_out = OUT / f"{name}.baba.aln"

    write_fasta(fasta, seqs)

    with open(mafft_out, "w") as f:
        subprocess.run(["mafft", "--auto", fasta],
                       stdout=f, stderr=subprocess.DEVNULL, check=True)

    subprocess.run(
        ["babappalign", fasta, "-o", baba_out, "--device", "cpu"],
        
        check=True
    )

    ma = read_fasta(mafft_out)
    ba = read_fasta(baba_out)

    gm = gap_fraction(ma)
    gb = gap_fraction(ba)

    print(f"  Gap fraction MAFFT : {gm:.3f}")
    print(f"  Gap fraction BABAPP: {gb:.3f}")

    if name == "identity" and ma != ba:
        print("  ❌ FAIL: identity mismatch")
    else:
        print("  ✅ PASS")

print("\n=== SANITY VALIDATION COMPLETE ===\n")
