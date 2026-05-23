"""
Smoke test for BABAPPAlign.

This test is intentionally minimal.
It verifies that the CLI entry point is installed and runnable.

IMPORTANT:
- Must NOT download models
- Must NOT run alignment
- Must work without GPU
- Must work without internet
"""

import subprocess
import sys


def test_babappalign_help():
    """
    Ensure that the BABAPPAlign CLI help runs successfully.
    """
    result = subprocess.run(
        [sys.executable, "-m", "babappalign.cli", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert "BABAPPAlign" in result.stdout or result.stderr
    assert "{auto,cpu,cuda,mps}" in result.stdout
