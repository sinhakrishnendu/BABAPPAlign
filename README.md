[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17934124.svg)](https://doi.org/10.5281/zenodo.17934124)

# BABAPPAlign

**BABAPPAlign** is a deep learning–based **progressive multiple sequence alignment (MSA) engine** for protein sequences.
It integrates pretrained protein language model embeddings with a learned pairwise scoring function to improve alignment
accuracy while remaining **fully usable on CPU-only systems**.

> GPU acceleration is optional and used only for performance, not correctness.

---

## Key Features

- Progressive multiple sequence alignment (MSA)
- Learned pairwise scoring model (BABAPPAScore)
- Uses pretrained **ESM2 residue embeddings**
- Can Runs on CPU-only systems
- Optional GPU acceleration
- Distributed via **Bioconda**

---

## Installation

### Install from Bioconda (recommended)

```bash
conda install -c bioconda babappalign
```

This installs a **CPU-compatible version**. No GPU or CUDA is required.

---

## Quick Start

### Basic usage

```bash
babappalign input.fasta > output.aln.fasta
```

### Explicit output file

```bash
babappalign input.fasta --out output.aln.fasta
```

---

## How BABAPPAlign Works

1. **Embedding generation**  
   Protein sequences are converted into residue-level embeddings using ESM2.

2. **Pairwise scoring**  
   A learned neural network model computes pairwise similarity scores between residues.

3. **Guide tree construction**  
   UPGMA is used to build a guide tree from learned distances.

4. **Progressive alignment**  
   Profiles are aligned using dynamic programming with affine gap penalties.

---

### First run

```bash
babappalign input.fasta
```

### Subsequent runs

```bash
babappalign input.fasta --embedding-dir embeddings/
```

Cached embeddings can be generated once on a GPU machine and reused on any computer.

---

## CPU vs GPU Execution

| Component | CPU | GPU |
|---------|-----|-----|
| MSA logic | ✅ | ✅ |
| Pairwise scoring | ✅ | ✅ |
| Guide tree | ✅ | ✅ |
| Progressive alignment | ✅ | ✅ |
| Embedding generation | ⚠️ Slow | 🚀 Fast |

Alignment accuracy is identical on CPU and GPU.

---

## Input Requirements

- Protein sequences only
- FASTA format
- No strict length or count limit (performance depends on hardware)

---

## Command-Line Options

```bash
babappalign --help
```

Common options:
- `--out FILE` : output alignment file
- `--embedding-dir DIR` : use cached embeddings
- `--cpu-only` : force CPU execution

---

## License

MIT License. See the `LICENSE` file for details.

---

## Citation

Manuscript in preparation.

---

## Author & Repository

- **Author:** Krishnendu Sinha  
- **GitHub:** https://github.com/sinhakrishnendu/BABAPPAlign  

---

**BABAPPAlign** is a portable, CPU-friendly MSA engine with optional GPU acceleration.
