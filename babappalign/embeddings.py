# babappalign/embeddings.py
"""
ESM embedding wrapper with simple caching.
Designed for feature-extraction mode (frozen model).
"""

import torch
import esm
import os
from tqdm import tqdm

def load_model(device="cpu"):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def extract_embeddings_from_fasta(fasta_path, out_path, device="cpu", batch_size=8):
    """
    fasta_path: path to FASTA with raw sequences (ungapped)
    out_path: torch file (.pt) saving list of tensors (per-sequence embeddings)
    device: "mps" or "cuda" or "cpu"
    """
    from babappalign.utils import read_fasta
    seqs = read_fasta(fasta_path)  # list of (id, seq)
    model, batch_converter = load_model(device=device)

    embeddings = []
    # process in batches
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch = seqs[i:i+batch_size]
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33]
        # extract per-sequence per-residue embeddings (remove special tokens)
        for j, (sid, s) in enumerate(batch):
            emb = reps[j, 1:1+len(s)].cpu()  # [L, D]
            embeddings.append((sid, s, emb))
    torch.save(embeddings, out_path)
    print("Saved embeddings to", out_path)
