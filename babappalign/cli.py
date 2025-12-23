#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entrypoints for the BABAPPAlign toolkit.

This module defines:

    babappalign   → runs the progressive MSA engine
    babappascore  → runs the deep scorer / score matrix generator
"""

from babappalign.babappalign import main as babappalign_main
from babappalign.babappascore import cli as babappascore_cli


def main():
    """
    Dispatcher for `babappalign` command.
    """
    babappalign_main()


def score():
    """
    Dispatcher for `babappascore` command.
    """
    babappascore_cli()


# Allow running: python -m babappalign.cli
if __name__ == "__main__":
    main()
