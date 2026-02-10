# BABAPPAlign

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17934124.svg)](https://doi.org/10.5281/zenodo.17934124)

## Overview

BABAPPAlign is an embedding-first progressive multiple sequence alignment (MSA) engine
for protein and coding nucleotide sequences.

It integrates pretrained protein language model embeddings with a learned neural
residue–residue scoring function within a classical, exact affine-gap dynamic
programming framework (Gotoh).

Version 1.2.0 introduces native codon alignment mode, allowing direct CDS alignment
without requiring external PAL2NAL.

BABAPPAlign is fully functional on CPU-only systems.
GPU acceleration is optional and affects performance only, not correctness.

---

## Key Features

- Progressive multiple sequence alignment (MSA)
- Strict learned residue–residue scoring model (BABAPPAScore)
- Pretrained protein language model residue embeddings
- Column-aware profile scoring
- True affine-gap dynamic programming (Gotoh algorithm)
- Exact dynamic programming (no heuristics inside DP)
- Neural inference performed outside DP recursion
- Native codon alignment mode (CDS → translate → back-map)
- Automatic frame validation in codon mode
- CPU-only compatible
- Optional GPU acceleration
- Explicit model specification (no silent fallback)
- Reproducible and Zenodo-backed model distribution

---

## Installation

Install from PyPI:

    pip install babappalign

This installs a CPU-compatible version.
No GPU or CUDA is required.

---

## Quick Start

### Protein alignment (default)

    babappalign input.fasta --model babappascore

Output:

    input.protein.aln.fasta

---

### Codon alignment (v1.2.0)

    babappalign cds.fasta --model babappascore --mode codon

Outputs:

    cds.protein.aln.fasta
    cds.codon.aln.fasta

No -o option is required.
Output filenames are generated automatically.

---

## Codon Mode Details

When --mode codon is enabled:

1. CDS sequences are validated:
   - Length divisible by 3
   - No internal stop codons
   - Valid nucleotide alphabet

2. Sequences are translated to protein.

3. Alignment is performed in protein space using the learned neural scoring model.

4. Aligned proteins are back-mapped to codon alignment (PAL2NAL-style logic).

Gap penalties are automatically scaled in codon mode for biological consistency.

No external PAL2NAL dependency is required.

---

## How BABAPPAlign Works

1. Residue Embedding  
   Protein sequences are converted into residue-level embeddings using a pretrained
   protein language model.

2. Learned Residue Scoring  
   Residue compatibility is evaluated using a pretrained neural scoring model
   (BABAPPAScore), replacing traditional substitution matrices.

3. Progressive Alignment  
   Sequences are progressively aligned using exact affine-gap dynamic programming
   (Gotoh). Neural inference is performed outside the DP recursion to preserve
   correctness.

The progressive ordering is a computational heuristic and is not interpreted
as a phylogeny.

---

## Alignment Core Integrity

The alignment engine uses:

- Three-state affine-gap DP (M, Ix, Iy)
- Explicit traceback matrices
- Exact dynamic programming
- No heuristic shortcuts inside recursion

Version 1.2.0 does not modify the alignment core logic.
Scientific reproducibility from earlier versions is preserved.

---

## Model Weights (Required)

BABAPPAlign requires a trained neural residue-level scoring model (BABAPPAScore),
distributed separately via Zenodo.

Concept DOI (all versions):

    https://doi.org/10.5281/zenodo.18053200

Download model:

    mkdir -p ~/.cache/babappalign/models

    wget https://zenodo.org/record/18053201/files/babappascore.pt \
      -O ~/.cache/babappalign/models/babappascore.pt

Run using cached model name:

    babappalign input.fasta --model babappascore

Or using explicit path:

    babappalign input.fasta \
      --model ~/.cache/babappalign/models/babappascore.pt

At runtime, BABAPPAlign prints the resolved model path and checksum
for reproducibility.

---

## CPU and GPU Execution

BABAPPAlign produces identical alignments on CPU and GPU.
GPU acceleration affects performance only.

Component                     CPU     GPU
------------------------------------------------
Progressive alignment (DP)    Yes     Yes
Learned scoring               Yes     Yes
Embedding generation          Slower  Faster

---

## Input Requirements

Protein mode:
- Protein FASTA sequences

Codon mode:
- CDS nucleotide FASTA sequences
- Length divisible by 3
- No internal stop codons

No strict limits on sequence number or length
(runtime depends on hardware).

---

## Command Line Interface

    babappalign --help

Key options:

    --model MODEL           (mandatory)
    --mode {protein,codon}
    --gap-open FLOAT
    --gap-extend FLOAT
    --device {cpu,cuda}

Output filenames are generated automatically.

---

## License

MIT License. See LICENSE file.

---

## Citation

Manuscript in preparation.

---

## Author

Krishnendu Sinha
https://github.com/sinhakrishnendu/BABAPPAlign
