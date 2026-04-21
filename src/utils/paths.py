"""Filesystem paths anchored at the repository root (not the notebook cwd)."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return project root: directory containing requirements.txt and notebooks/."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "requirements.txt").is_file() and (p / "notebooks").is_dir():
            return p
    return here
