#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entrypoints for the BABAPPAlign toolkit.

This module defines:

    babappalign   → runs the progressive MSA engine
    babappascore  → runs the deep scorer / score matrix generator

These entrypoints are registered in setup.py so users can run them directly:

    $ babappalign --sequences in.fasta --output out.fasta
    $ babappascore --seqA A.fasta --seqB B.fasta

"""

import sys
from babappalign.babappalign import cli as babappalign_cli
from babappalign.babappascore import cli as babappascore_cli


def main():
    """
    Dispatcher for `babappalign` command.
    """
    babappalign_cli()


def score():
    """
    Dispatcher for `babappascore` command.
    """
    babappascore_cli()


# Allow running python -m babappalign.cli
if __name__ == "__main__":
    # If invoked directly, behave like babappalign
    main()
