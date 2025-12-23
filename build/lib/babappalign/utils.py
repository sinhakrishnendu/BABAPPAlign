# babappalign/utils.py
from typing import List, Tuple

def read_fasta(path) -> List[Tuple[str,str]]:
    seqs = []
    with open(path) as f:
        name = None
        seq_lines = []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(seq_lines)))
                name = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if name is not None:
            seqs.append((name, "".join(seq_lines)))
    return seqs

def write_fasta(seq_list, out_path):
    with open(out_path, "w") as f:
        for name, seq in seq_list:
            f.write(f">{name}\n")
            f.write(seq + "\n")
