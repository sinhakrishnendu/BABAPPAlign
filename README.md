# BABAPPAlign

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17934124.svg)](https://doi.org/10.5281/zenodo.17934124)

## Overview

**BABAPPAlign** is an embedding-first **progressive multiple sequence alignment (MSA) engine** for protein sequences.
It integrates pretrained protein language model embeddings with a **learned neural residue–residue scoring function**
within a **classical, exact affine-gap dynamic programming framework (Gotoh)**.

The method is designed to improve alignment accuracy while maintaining methodological transparency and
full reproducibility. BABAPPAlign is **fully functional on CPU-only systems**; GPU acceleration is optional
and affects performance only, not correctness.

---

## Key features

- Progressive multiple sequence alignment (MSA)
- **Strict learned residue–residue scoring model (BABAPPAScore)**
- Uses pretrained protein language model residue embeddings
- Column-aware profile scoring
- True affine-gap dynamic programming (Gotoh algorithm)
- Exact dynamic programming (no heuristics inside DP)
- Embedding inference performed outside DP
- Fully functional on CPU-only systems
- Optional GPU acceleration for faster embedding and scoring
- Explicit model specification (no silent fallback)
- Reproducible and Bioconda-compliant design

---

## Installation

### Read INSTALL.md for detailed instruction
### Install from PyPI

```bash
pip install babappalign
```
This installs a CPU-compatible version of BABAPPAlign.
No GPU, CUDA, or special hardware is required.

---

## Quick start

```bash
babappalign input.fasta -o output.aln.fasta --model babappascore
```

> **Important:**  
> BABAPPAlign requires an external trained neural scoring model.
> The model is **not downloaded automatically** and must be obtained explicitly (see below).

---

## How BABAPPAlign works

1. **Residue embedding**  
   Each protein sequence is converted into residue-level embeddings using a pretrained protein language model.

2. **Learned residue scoring**  
   Residue compatibility is evaluated using a pretrained neural scoring model (**BABAPPAScore**),
   replacing traditional substitution matrices.

3. **Progressive alignment**  
   Sequences are progressively aligned using **exact affine-gap dynamic programming (Gotoh)**.
   Neural inference is performed outside the DP recursion to preserve correctness.

The progressive ordering is a computational heuristic and is **not interpreted as a phylogeny**.

---

## Model weights (required)

BABAPPAlign requires a trained neural residue-level scoring model (**BABAPPAScore**),
which is distributed separately via Zenodo.

**Concept DOI (all versions):**  
https://doi.org/10.5281/zenodo.18053200  

Version-specific DOIs are provided on Zenodo for exact reproducibility.

### Download and use

```bash
# 1. Download the model (one-time)
mkdir -p ~/.cache/babappalign/models

wget https://zenodo.org/record/18053201/files/babappascore.pt      -O ~/.cache/babappalign/models/babappascore.pt

# 2a. Run BABAPPAlign using the cached model name (recommended)
babappalign input.fasta -o aligned.fasta --model babappascore

# 2b. OR run BABAPPAlign using an explicit model path (equivalent)
babappalign input.fasta -o aligned.fasta \
  --model ~/.cache/babappalign/models/babappascore.pt
```

At runtime, BABAPPAlign prints the resolved model path and a SHA-256 checksum to ensure
transparent and reproducible model usage.

---

## CPU and GPU execution

BABAPPAlign produces identical alignments on CPU and GPU.
GPU acceleration affects performance only.

| Component | CPU | GPU |
|---------|-----|-----|
| Progressive alignment (DP) | Yes | Yes |
| Learned scoring | Yes | Yes |
| Embedding generation | Slower | Faster |

---

## Input requirements

- Protein sequences only  
- FASTA format  
- No strict limits on sequence length or number (runtime depends on hardware)

---

## Command-line interface

```bash
babappalign --help
```

Key options include:

- `-o, --output FILE` : output alignment file  
- `--model MODEL` : scoring model name or path (mandatory)  
- `--gap-open FLOAT` : gap opening penalty  
- `--gap-extend FLOAT` : gap extension penalty  
- `--device {cpu,cuda}` : select execution device  

---

## License

MIT License. See the `LICENSE` file for details.

---

## Citation

Manuscript in preparation.

---

## Author and repository

- **Author:** Krishnendu Sinha  
- **GitHub:** https://github.com/sinhakrishnendu/BABAPPAlign
