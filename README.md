# BABAPPAlign

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17934124.svg)](https://doi.org/10.5281/zenodo.17934124)

## Overview

**BABAPPAlign** is an embedding-first **progressive multiple sequence alignment (MSA) engine** for protein sequences.
It integrates pretrained protein language model embeddings with a **learned neural residue–residue scoring function**
within a **classical, exact affine-gap dynamic programming framework**.

The method is designed to improve alignment accuracy while remaining **fully functional on CPU-only systems**.
GPU acceleration is optional and affects performance only, not correctness.

---

## Key features

- Progressive multiple sequence alignment (MSA)
- Learned residue–residue scoring model (BABAPPAScore)
- Uses pretrained ESM2 residue embeddings
- Data-driven guide tree construction using Neighbor Joining (NJ)
- Optional residue-level bootstrap with majority-rule consensus topology
- True affine-gap dynamic programming (Gotoh algorithm)
- Symmetric profile–profile alignment
- Fully functional on CPU-only systems
- Optional GPU acceleration for faster embedding generation and scoring
- Automatic caching of model weights
- Distributed via Bioconda

---

## Installation

### Install from Bioconda (recommended)

```bash
conda install -c bioconda babappalign
```

This installs a CPU-compatible version of BABAPPAlign.
No GPU, CUDA, or special hardware is required.

---

## Quick start

### Basic usage

```bash
babappalign input.fasta -o output.aln.fasta
```

On first use, the pretrained scoring model is downloaded automatically.

---

## How BABAPPAlign works

1. **Residue embedding**  
   Each protein sequence is converted into residue-level embeddings using a pretrained ESM2 model.

2. **Guide tree construction**  
   Sequence-level embeddings are obtained by pooling residue embeddings.
   Pairwise distances are defined using cosine dissimilarity, and a guide tree is inferred using the
   Neighbor Joining (NJ) algorithm.
   Optionally, residue-level bootstrapping can be used to construct a majority-rule consensus tree.

3. **Learned residue scoring**  
   Residue compatibility is evaluated using a pretrained neural scoring model (BABAPPAScore),
   which replaces traditional substitution matrices.

4. **Progressive alignment**  
   Sequences and profiles are progressively aligned following the guide tree using
   exact affine-gap dynamic programming (Gotoh), with symmetric profile–profile alignment.

The guide tree is used as a computational heuristic and is not interpreted as a phylogeny.

---

## Model weights and automatic download

BABAPPAlign relies on a pretrained neural residue–residue scoring model (`babappascore.pt`).
Due to its size, the model weights are not bundled with the software package.

### Automatic model retrieval

When BABAPPAlign is run for the first time, the pretrained scoring model is automatically downloaded
from the official GitHub release corresponding to the installed version.
The model file is cached locally and reused for subsequent runs.

No manual download or configuration is required.

### Cache location

By default, the model is stored under the user cache directory:

```
~/.cache/babappalign/models/babappascore.pt
```

The cache location follows the XDG base directory specification where applicable.

### Offline and custom models

Users may optionally supply a local model file:

```bash
babappalign input.fasta -o output.aln.fasta --model /path/to/babappascore.pt
```

This is useful for offline environments, custom-trained models, or reproducibility experiments.

---

## CPU and GPU execution

BABAPPAlign produces identical alignments on CPU and GPU.
GPU acceleration is used only to improve performance.

| Component | CPU | GPU |
|---------|-----|-----|
| Guide tree construction | Yes | Yes |
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
- `--model FILE` : use a local scoring model  
- `--bootstrap N` : number of bootstrap replicates for guide tree construction  
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
