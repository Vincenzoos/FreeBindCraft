"""Filename and output naming helpers for IPSAE runs."""

from __future__ import annotations

import re
from pathlib import Path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "ipsae_model"


def model_name_from_structure(structure_file: str | Path) -> str:
    return safe_name(Path(structure_file).stem)


def ipsae_output_stem(structure_file: Path, pae_cutoff: float, dist_cutoff: float) -> Path:
    pae_tag = f"{int(pae_cutoff):02d}" if pae_cutoff < 10 else str(int(pae_cutoff))
    dist_tag = f"{int(dist_cutoff):02d}" if dist_cutoff < 10 else str(int(dist_cutoff))
    stem = structure_file.with_suffix("")
    return stem.with_name(f"{stem.name}_{pae_tag}_{dist_tag}")


def ipsae_output_files(structure_file: Path, pae_cutoff: float, dist_cutoff: float) -> dict[str, Path]:
    stem = ipsae_output_stem(structure_file, pae_cutoff, dist_cutoff)
    return {
        "scores": stem.with_suffix(".txt"),
        "byres": Path(f"{stem}_byres.txt"),
        "pymol": stem.with_suffix(".pml"),
    }
