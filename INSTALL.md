# Installation Guide — BABAPPAlign

This document describes the supported and reproducible installation procedure for BABAPPAlign, a deep learning–based multiple sequence alignment engine.

## System requirements

- Linux (x86_64 recommended)
- Python 3.9–3.11 (Python 3.10 recommended)
- Conda (Miniconda or Mambaforge)
- Optional: CUDA-capable GPU for faster inference

## Recommended installation (Conda + pip)

BABAPPAlign uses PyTorch and ESM protein language models. While the core package is distributed via Bioconda, one required dependency (fair-esm) is pip-only and must be installed separately. This hybrid installation pattern is standard for modern ML-based bioinformatics tools.

### Step 1 — Create a clean Conda environment

conda create -n babappalign python=3.10 -y  
conda activate babappalign

Verify:

python --version

### Step 2 — Install BABAPPAlign from Bioconda

conda install -c bioconda babappalign

This installs BABAPPAlign, PyTorch, NumPy/SciPy, Biopython, Transformers, and other standard scientific dependencies.

### Step 3 — Install the ESM dependency (pip-only)

BABAPPAlign requires the ESM protein language model implementation provided by fair-esm. This package is not distributed via Conda and must be installed with pip.

pip install fair-esm

Optional (recommended for reproducibility):

pip install fair-esm==2.0.0

## Model weights (required)

BABAPPAlign uses a pretrained residue-level scoring model that must be downloaded manually.

mkdir -p ~/.cache/babappalign/models  
wget https://zenodo.org/records/18053201/files/babappascorer.pt -O ~/.cache/babappalign/models/babappascorer.pt

The filename must be exactly babappascorer.pt.

## Verify installation

babappalign --help

Optional runtime check:

python - << EOF  
import esm  
import babappalign  
print("BABAPPAlign installation OK")  
EOF

## GPU support (optional)

If a compatible GPU and CUDA-enabled PyTorch are available, BABAPPAlign will automatically use the GPU.

babappalign test.fasta -o out.fasta

Example output:

[BABAPPAlign] Using device: cuda

## Notes on dependency management

### Why fair-esm is installed via pip

- fair-esm is not available in Conda channels
- Conda does not resolve pip-installed packages during dependency solving
- Declaring fair-esm as a Conda dependency would make the package uninstallable

Therefore, fair-esm is treated as a mandatory pip-only runtime dependency. BABAPPAlign will fail with a clear error message if it is missing. This approach is consistent with modern PyTorch- and ESM-based workflows.

### Why the package is not noarch

Although the source code is pure Python, BABAPPAlign depends at runtime on PyTorch (architecture- and ABI-specific), ESM model loading, transformer backends, and optional GPU acceleration. Therefore, the package is intentionally not marked as noarch.

## Minimal installation summary

conda create -n babappalign python=3.10 -y  
conda activate babappalign  
conda install -c bioconda babappalign  
pip install fair-esm

## Troubleshooting

If you see the error “fair-esm is required but not installed”, run:

pip install fair-esm

If you see an error indicating the scoring model is missing, ensure the file exists at:

~/.cache/babappalign/models/babappascorer.pt

## Citation

If you use BABAPPAlign in academic work, please cite the corresponding publication and Zenodo model record.

## Support

GitHub Issues: https://github.com/sinhakrishnendu/BABAPPAlign/issues  
Model archive: https://zenodo.org/records/18053201
