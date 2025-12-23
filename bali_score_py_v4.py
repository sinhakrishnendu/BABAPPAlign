#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bali_score_py_v4.py
Robust BAliBASE SP/TC scorer with correct XML (<seq-data>) and MSF parsing.
Autodetects formats and uses tolerant content mapping.
"""
from __future__ import annotations
import sys, argparse, collections
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

# ---------- Parsers (FASTA/MSF/XML) ----------
def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    ids, seqs = [], []
    cur_id = None; cur_seq = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id); seqs.append("".join(cur_seq))
                header = line[1:].strip()
                cur_id = header.split()[0]; cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            ids.append(cur_id); seqs.append("".join(cur_seq))
    return ids, seqs

def read_msf(path: Path) -> Tuple[List[str], List[str]]:
    seq_map: Dict[str, List[str]] = collections.OrderedDict()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    names: List[str] = []
    # header parsing
    for ln in lines:
        if ln.strip().startswith("//"):
            break
        l = ln.strip()
        if l.startswith("Name:"):
            parts = l.split()
            if len(parts) >= 2:
                name = parts[1]
                if name not in names:
                    names.append(name); seq_map[name] = []
    # find alignment start
    try:
        start_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("//"))
        align_lines = lines[start_idx+1:]
    except StopIteration:
        align_lines = lines

    for ln in align_lines:
        if not ln:
            continue
        lstripped = ln.lstrip()
        if all((ch.isdigit() or ch.isspace()) for ch in lstripped):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        first = parts[0]
        if ("/" in first) or first.lower().endswith(".msf"):
            continue
        seg = parts[-1]
        if not seg or not all((ch.isalpha() or ch == '-' or ch == '.') for ch in seg):
            continue
        name = parts[0]
        if names:
            if name not in seq_map:
                matched = None
                for nm in names:
                    if nm.startswith(name) or name.startswith(nm):
                        matched = nm; break
                if matched is not None:
                    seq_map[matched].append(seg)
                else:
                    continue
            else:
                seq_map[name].append(seg)
        else:
            seq_map.setdefault(name, []).append(seg)

    ids, seqs = [], []
    if names:
        for nm in names:
            ids.append(nm); seqs.append("".join(seq_map.get(nm, [])))
    else:
        for nm, segs in seq_map.items():
            ids.append(nm); seqs.append("".join(segs))
    return ids, seqs

def parse_bali_xml(path: Path) -> Tuple[List[str], List[str]]:
    """
    Robust BAliBASE XML parser extracting <seq-name> and <seq-data>.
    """
    tree = ET.parse(str(path))
    root = tree.getroot()
    seq_nodes = root.findall(".//sequence")
    if not seq_nodes:
        seq_nodes = root.findall(".//seq")
    ids: List[str] = []
    seqs: List[str] = []
    for s in seq_nodes:
        sid = None
        name_node = s.find("seq-name")
        if name_node is not None and name_node.text and name_node.text.strip():
            sid = name_node.text.strip()
        else:
            sid = s.get("id") or s.get("name") or s.get("code") or s.get("seqid")
        if sid is None:
            sid = f"seq{len(ids)+1}"
        seq_data_node = s.find("seq-data")
        aln = ""
        if seq_data_node is not None:
            if seq_data_node.text and seq_data_node.text.strip():
                aln = seq_data_node.text
            else:
                # join child texts if any
                aln = "".join((child.text or "") for child in seq_data_node)
            aln = "".join(aln.split())
        else:
            if s.text and s.text.strip():
                aln = "".join(s.text.split())
            else:
                aln = ""
        ids.append(sid); seqs.append(aln)
    return ids, seqs

def autodetect_and_parse(path: Path) -> Tuple[List[str], List[str]]:
    suf = path.suffix.lower()
    if suf == ".xml":
        return parse_bali_xml(path)
    if suf in (".msf", ".msf2"):
        return read_msf(path)
    try:
        ids, seqs = read_fasta(path)
        if ids and (any(('-' in s or '.' in s) for s in seqs) or len(set(len(s) for s in seqs)) == 1):
            return ids, seqs
    except Exception:
        pass
    try:
        return read_msf(path)
    except Exception:
        return parse_bali_xml(path)

# ---------- mapping & scoring (same tolerant strategy) ----------
def normalize_ungapped(seq: str) -> str:
    s = seq.upper().replace(".", "")
    return "".join(ch for ch in s if ch.isalpha())

def identity_fraction(a: str, b: str) -> float:
    if len(a)==0 or len(b)==0 or len(a)!=len(b): return 0.0
    return sum(1 for x,y in zip(a,b) if x==y)/len(a)

def map_ref_to_test_by_content(ref_ids, ref_seqs, test_ids, test_seqs, verbose=False, identity_threshold=0.95):
    ref_ungapped = [normalize_ungapped(s) for s in ref_seqs]
    test_ungapped = [normalize_ungapped(s) for s in test_seqs]
    mapping = {i: None for i in range(len(ref_ids))}
    used=set()
    test_index_by_seq={}
    for j, seq in enumerate(test_ungapped):
        test_index_by_seq.setdefault(seq, []).append(j)
    for i,r in enumerate(ref_ungapped):
        if r in test_index_by_seq:
            for c in test_index_by_seq[r]:
                if c not in used:
                    mapping[i]=c; used.add(c); break
            if mapping[i] is None:
                mapping[i]=test_index_by_seq[r][0]; used.add(mapping[i])
            if verbose:
                print(f"[map] exact match ref#{i} ({ref_ids[i]}) -> test#{mapping[i]} ({test_ids[mapping[i]]})", file=sys.stderr)
    for i,r in enumerate(ref_ungapped):
        if mapping[i] is not None: continue
        best_j=None; best_score=0.0
        for j,t in enumerate(test_ungapped):
            if j in used: continue
            if len(r)==0 or len(t)==0 or len(r)!=len(t): continue
            frac = identity_fraction(r,t)
            if frac>best_score:
                best_score=frac; best_j=j
        if best_j is not None and best_score>=identity_threshold:
            mapping[i]=best_j; used.add(best_j)
            if verbose:
                print(f"[map] tolerant ref#{i} -> test#{best_j} identity={best_score:.3f}", file=sys.stderr)
    if any(v is None for v in mapping.values()):
        lower_test=[t.lower() for t in test_ids]
        for i in range(len(ref_ids)):
            if mapping[i] is not None: continue
            rid=ref_ids[i].lower()
            for j,t in enumerate(lower_test):
                if j in used: continue
                if rid in t or t in rid:
                    mapping[i]=j; used.add(j)
                    if verbose:
                        print(f"[map] header fallback ref#{i} ({ref_ids[i]}) -> test#{j} ({test_ids[j]})", file=sys.stderr)
                    break
    if verbose:
        for i,v in mapping.items():
            if v is None:
                print(f"[map] UNMAPPED ref#{i} ({ref_ids[i]})", file=sys.stderr)
    return mapping

def build_residue_index_map(aligned_seqs):
    mapping={}
    for s_idx, aln in enumerate(aligned_seqs):
        mapping[s_idx]={}
        resid=0
        for col_idx,ch in enumerate(aln):
            if ch=='-' or ch=='.': continue
            resid+=1
            mapping[s_idx][resid]=col_idx
    return mapping

def compute_SP_TC_from_aligned(ref_aligned, test_aligned):
    nseq=len(ref_aligned)
    if nseq==0: return 0.0,0.0,0,0
    ref_map=build_residue_index_map(ref_aligned)
    test_map=build_residue_index_map(test_aligned)
    ref_len=len(ref_aligned[0])
    ref_cols=[]
    for c in range(ref_len):
        col_entries=[]
        for s in range(nseq):
            ch=ref_aligned[s][c]
            if ch=='-' or ch=='.': continue
            resid=None
            for rnum,colidx in ref_map[s].items():
                if colidx==c:
                    resid=rnum; break
            if resid is not None:
                col_entries.append((s,resid))
        ref_cols.append(col_entries)
    total_pairs=0; conserved_pairs=0; total_columns=0; conserved_columns=0
    for col in ref_cols:
        m=len(col)
        if m<2: continue
        for i in range(m):
            for j in range(i+1,m):
                s1,r1=col[i]; s2,r2=col[j]
                total_pairs+=1
                c1=test_map.get(s1,{}).get(r1,None)
                c2=test_map.get(s2,{}).get(r2,None)
                if c1 is not None and c2 is not None and c1==c2:
                    conserved_pairs+=1
    for col in ref_cols:
        if len(col)==0: continue
        total_columns+=1
        test_cols=set(); ok=True
        for s,r in col:
            cidx=test_map.get(s,{}).get(r,None)
            if cidx is None: ok=False; break
            test_cols.add(cidx)
            if len(test_cols)>1: ok=False; break
        if not ok or len(test_cols)!=1: continue
        tcol=next(iter(test_cols))
        test_col_pairs=set()
        for s_idx in range(nseq):
            for rnum,colidx in test_map.get(s_idx,{}).items():
                if colidx==tcol:
                    test_col_pairs.add((s_idx,rnum)); break
        ref_col_pairs=set(col)
        if test_col_pairs==ref_col_pairs:
            conserved_columns+=1
    SP = conserved_pairs/total_pairs if total_pairs>0 else 0.0
    TC = conserved_columns/total_columns if total_columns>0 else 0.0
    return SP,TC,total_pairs,total_columns

def prepare_aligned_lists(ref_ids, ref_seqs, test_ids, test_seqs, verbose=False, identity_threshold=0.95):
    mapping = map_ref_to_test_by_content(ref_ids, ref_seqs, test_ids, test_seqs, verbose=verbose, identity_threshold=identity_threshold)
    ref_aligned=[]; test_aligned=[]
    for i,rid in enumerate(ref_ids):
        ref_aln=ref_seqs[i]; ref_aligned.append(ref_aln)
        j=mapping.get(i)
        if j is None:
            test_aligned.append('-'*len(ref_aln))
        else:
            test_aln=test_seqs[j]
            if len(test_aln)<len(ref_aln):
                test_aln = test_aln + '-'*(len(ref_aln)-len(test_aln))
            elif len(test_aln)>len(ref_aln):
                test_aln = test_aln[:len(ref_aln)]
            test_aligned.append(test_aln)
    return ref_aligned, test_aligned, mapping

def main():
    p=argparse.ArgumentParser(description="bali_score_py_v4: robust MSF/XML parser + tolerant mapping")
    p.add_argument("ref")
    p.add_argument("test")
    p.add_argument("-v","--verbose", action="store_true")
    p.add_argument("--identity-threshold", type=float, default=0.95)
    args=p.parse_args()
    refp=Path(args.ref); testp=Path(args.test)
    if not refp.exists():
        print(f"[ERROR] reference not found: {refp}", file=sys.stderr); sys.exit(2)
    if not testp.exists():
        print(f"[ERROR] test alignment not found: {testp}", file=sys.stderr); sys.exit(2)
    try:
        ref_ids, ref_seqs = autodetect_and_parse(refp)
    except Exception as e:
        print(f"[ERROR] failed parsing reference {refp}: {e}", file=sys.stderr); sys.exit(2)
    try:
        test_ids, test_seqs = autodetect_and_parse(testp)
    except Exception as e:
        print(f"[ERROR] failed parsing test {testp}: {e}", file=sys.stderr); sys.exit(2)
    if args.verbose:
        print(f"[info] parsed reference: {len(ref_ids)} sequences", file=sys.stderr)
        print(f"[info] parsed test: {len(test_ids)} sequences", file=sys.stderr)
        print(f"[info] ref ids snippet: {ref_ids[:8]}", file=sys.stderr)
        print(f"[info] test ids snippet: {test_ids[:12]}", file=sys.stderr)
    ref_aligned, test_aligned, mapping = prepare_aligned_lists(ref_ids, ref_seqs, test_ids, test_seqs, verbose=args.verbose, identity_threshold=args.identity_threshold)
    SP,TC,tp,tcnt = compute_SP_TC_from_aligned(ref_aligned, test_aligned)
    print(f"SP score: {SP:.6f}")
    print(f"TC score: {TC:.6f}")
    if args.verbose:
        print(f"[info] total_pairs {tp}, total_columns {tcnt}", file=sys.stderr)
        mapped_count = sum(1 for v in mapping.values() if v is not None)
        print(f"[info] mapping: {mapped_count}/{len(ref_ids)} reference sequences mapped to test sequences", file=sys.stderr)
        for i,j in mapping.items():
            if j is None:
                print(f"  ref#{i} ({ref_ids[i]}) -> UNMAPPED", file=sys.stderr)
            else:
                print(f"  ref#{i} ({ref_ids[i]}) -> test#{j} ({test_ids[j]})", file=sys.stderr)

if __name__=="__main__":
    main()
