#!/usr/bin/env python3

import os

ROOT_DIR = "."                 # start from current directory
OUTPUT_FILE = "combined.txt"   # output file
ENCODING = "utf-8"

with open(OUTPUT_FILE, "w", encoding=ENCODING) as out:
    for root, _, files in os.walk(ROOT_DIR):
        for name in sorted(files):
            # only include Python files
            if not name.endswith(".py"):
                continue

            path = os.path.join(root, name)

            # skip the output file itself (in case it's a .py in future)
            if os.path.abspath(path) == os.path.abspath(OUTPUT_FILE):
                continue

            try:
                with open(path, "r", encoding=ENCODING, errors="ignore") as f:
                    out.write(f"\n\n===== FILE: {path} =====\n\n")
                    out.write(f.read())
            except Exception as e:
                out.write(f"\n\n===== FILE: {path} (ERROR: {e}) =====\n\n")
