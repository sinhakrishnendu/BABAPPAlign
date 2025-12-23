#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_msa_engines.py

Production-grade BAliBASE benchmark runner for BABAPPAlign vs classical MSA tools.

Tools:
  - BABAPPAlign (CUDA)
  - MAFFT
  - MUSCLE
  - T-Coffee
  - ClustalW

Scoring:
  - bali_score_py_v4.py ONLY (Python, tolerant mapping)

Outputs:
  results/
    ├── alignments/<tool>/*.fasta
    ├── logs/<family>_<tool>.log
    ├── scores.tsv
    └── run_metadata.json

Usage example:
  python3 benchmark_msa_engine.py --balibase /path/to/balibase --out resultsdir --timeout 90000
"""

from __future__ import annotations
import subprocess
import argparse
import time
import json
import sys
import os
import shlex
from pathlib import Path
from typing import Dict, Any, List

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def run_cmd(cmd: str, timeout: int | None = None) -> dict:
    start = time.time()
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "runtime_s": time.time() - start
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": 124,
            "stdout": e.stdout or "",
            "stderr": f"TIMEOUT after {timeout}s",
            "runtime_s": time.time() - start
        }

def tool_version(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            shlex.split(cmd),
            stderr=subprocess.STDOUT,
            text=True
        )
        return out.strip().splitlines()[0][:200]
    except Exception:
        return "unknown"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balibase", required=True, help="BAliBASE BB3 root (contains RV11–RV50)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--workers", type=int, default=1, help="reserved (serial for reproducibility)")
    ap.add_argument("--timeout", type=int, default=90000, help="seconds per alignment")
    args = ap.parse_args()

    balibase = Path(args.balibase)
    outroot = Path(args.out)
    align_root = outroot / "alignments"
    log_root = outroot / "logs"

    align_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # locate scorer
    script_dir = Path(__file__).resolve().parent
    scorer = script_dir / "bali_score_py_v4.py"
    if not scorer.exists():
        print("ERROR: bali_score_py_v4.py not found next to script", file=sys.stderr)
        sys.exit(2)

    # tool registry (DEFAULTS ONLY)
    tools = {
        "babappalign": {
            "cmd": "babappalign {infile} -o {outfile} --device cpu"
        },

        "mafft": {
            "cmd": "mafft --auto {infile} > {outfile}"
        },
        "muscle": {
            "cmd": "muscle -in {infile} -out {outfile}"
        },
        "tcoffee": {
            "cmd": "t_coffee {infile} -output fasta_aln -outfile {outfile}"
        },
        "clustalw": {
            "cmd": "clustalw2 -INFILE={infile} -OUTFILE={outfile} -OUTPUT=FASTA"
        }
    }

    # collect BAliBASE families
    families = []
    for rv in sorted(balibase.iterdir()):
        if rv.is_dir() and rv.name.startswith("RV"):
            for xml in sorted(rv.glob("BB*.xml")):
                tfa = xml.with_suffix(".tfa")
                if tfa.exists():
                    families.append((rv.name, tfa, xml))

    if not families:
        print("No BAliBASE families found", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(families)} BAliBASE families")

    # metadata
    metadata = {
        "timestamp": time.ctime(),
        "host": os.uname().nodename,
        "tools": {},
    }

    for name in tools:
        metadata["tools"][name] = tool_version(name + " --version")

    results: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Benchmark loop (serial by design)
    # --------------------------------------------------

    for rv, tfa, xml in families:
        fam = tfa.stem
        print(f"\n=== {rv}/{fam} ===")

        for tool, spec in tools.items():
            print(f"  → {tool}")
            tool_dir = align_root / tool
            tool_dir.mkdir(exist_ok=True)

            aln_out = tool_dir / f"{fam}.fasta"
            log_file = log_root / f"{fam}_{tool}.log"

            # run tool
            cmd = spec["cmd"].format(
                infile=shlex.quote(str(tfa)),
                outfile=shlex.quote(str(aln_out))
            )

            res = run_cmd(cmd, timeout=args.timeout)

            with open(log_file, "w") as log:
                log.write(f"CMD: {cmd}\n\n")
                log.write("=== STDOUT ===\n")
                log.write(res["stdout"])
                log.write("\n=== STDERR ===\n")
                log.write(res["stderr"])

            if res["returncode"] != 0 or not aln_out.exists():
                results.append({
                    "family": fam,
                    "rv": rv,
                    "tool": tool,
                    "status": "FAILED",
                    "runtime_s": res["runtime_s"],
                    "sp": None,
                    "tc": None,
                    "error": "alignment_failed"
                })
                continue

            # score
            score_cmd = f"python3 {scorer} {xml} {aln_out}"
            score = run_cmd(score_cmd, timeout=300)

            with open(log_file, "a") as log:
                log.write("\n=== SCORING ===\n")
                log.write(score_cmd + "\n")
                log.write(score["stdout"])
                log.write(score["stderr"])

            sp = tc = None
            for line in score["stdout"].splitlines():
                if line.startswith("SP score"):
                    sp = float(line.split()[-1])
                if line.startswith("TC score"):
                    tc = float(line.split()[-1])

            results.append({
                "family": fam,
                "rv": rv,
                "tool": tool,
                "status": "OK" if sp is not None else "FAILED",
                "runtime_s": res["runtime_s"],
                "sp": sp,
                "tc": tc,
                "error": None if sp is not None else "scoring_failed"
            })

    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------

    outroot.mkdir(parents=True, exist_ok=True)

    # TSV
    tsv = outroot / "scores.tsv"
    with open(tsv, "w") as fh:
        fh.write("rv\tfamily\ttool\tstatus\truntime_s\tSP\tTC\terror\n")
        for r in results:
            fh.write(
                f"{r['rv']}\t{r['family']}\t{r['tool']}\t{r['status']}\t"
                f"{r['runtime_s']:.2f}\t{r['sp']}\t{r['tc']}\t{r['error']}\n"
            )

    # JSON
    json.dump(
        {"metadata": metadata, "results": results},
        open(outroot / "run_metadata.json", "w"),
        indent=2
    )

    print("\nBenchmark complete")
    print("Results:", tsv)
    print("Metadata:", outroot / "run_metadata.json")

if __name__ == "__main__":
    main()

