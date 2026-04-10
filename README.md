# BABAPPAlign

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17934124.svg)](https://doi.org/10.5281/zenodo.17934124)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19034335.svg)](https://doi.org/10.5281/zenodo.19034335)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18053201.svg)](https://doi.org/10.5281/zenodo.18053201)

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
- Mandatory `babappascore.pt` model loading (no model override)
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

    babappalign input.fasta

Output:

    input.protein.aln.fasta

---

### Codon alignment (v1.2.0)

    babappalign cds.fasta --mode codon

Outputs:

    cds.protein.aln.fasta
    cds.codon.aln.fasta

No -o option is required.
Output filenames are generated automatically.

---

### Interactive mode (`--i`)

    babappalign --i

Prompts:

    Sequence FASTA file:
    Mode [protein/codon] (default: protein):

The scorer is always the required `babappascore.pt` model.

Without `--i`, BABAPPAlign runs in normal static CLI mode and expects
the FASTA path directly in the command line.

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

BABAPPAlign always loads:

    ~/.cache/babappalign/models/babappascore.pt

If this file is missing, the CLI exits explicitly with a `[FATAL]` error.

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

    --i                   interactive mode
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

If this software contributes to your research, please cite:

Sinha K. **BABAPPAlign: A Multiple Sequence Alignment Engine with a Learned Residue-Level Scoring Function**. *bioRxiv* (2025). DOI: [10.64898/2025.12.26.696577](https://doi.org/10.64898/2025.12.26.696577)

Link: [http://biorxiv.org/content/early/2025/12/29/2025.12.26.696577.abstract](http://biorxiv.org/content/early/2025/12/29/2025.12.26.696577.abstract)

---

## Author

Krishnendu Sinha
https://github.com/sinhakrishnendu/BABAPPAlign
