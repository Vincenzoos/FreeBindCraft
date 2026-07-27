"""Bulk AlphaFold Server discovery and ipSAE execution helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from single_model_eval import (
    DEFAULT_EVALS_DIR,
    ROOT,
    IpsaeJob,
    _rel,
    model_name_from_structure,
    resolve_repo_path,
    run_ipsae,
)


def new_bulk_output_dir(base: str = "bulk_ipsae_evals") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = ROOT / f"{base}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = ROOT / f"{base}_{stamp}_{suffix}"
        suffix += 1
    return candidate


def bulk_log_path(output_dir: str | Path) -> Path:
    out_dir = resolve_repo_path(output_dir)
    return out_dir / f"{out_dir.name}_log.json"


def write_bulk_run_log(
    output_dir: str | Path,
    af3_folder: str | Path,
    model_index: int,
    jobs: list[IpsaeJob],
    ok_count: int,
    error_count: int,
) -> Path:
    out_dir = resolve_repo_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    af3_path = resolve_repo_path(af3_folder)
    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "af3_folder": _rel(af3_path),
        "model_index": int(model_index),
        "model_count": len(jobs),
        "successful_models": int(ok_count),
        "failed_models": int(error_count),
        "models": [
            {
                "model": job.label,
                "structure_file": _rel(resolve_repo_path(job.structure_file)),
                "pae_file": _rel(resolve_repo_path(job.pae_file)),
            }
            for job in jobs
        ],
    }
    log_path = bulk_log_path(out_dir)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    return log_path


def discover_af3_models(
    folder: str | Path,
    model_index: int = 0,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> tuple[list[IpsaeJob], pd.DataFrame]:
    """Find AlphaFold Server model_N CIF files and matching full_data_N JSON files below a folder."""
    root = resolve_repo_path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    jobs = []
    rows = []
    model_pattern = f"*_model_{model_index}.cif"
    for structure in sorted(root.rglob(model_pattern)):
        prefix = structure.name[: -len(f"_model_{model_index}.cif")]
        pae = structure.with_name(f"{prefix}_full_data_{model_index}.json")
        model_name = model_name_from_structure(structure)
        found = pae.exists()
        rows.append(
            {
                "Model": model_name,
                "Folder": _rel(structure.parent),
                "Structure": structure.name,
                "PAE": pae.name,
                "Ready": found,
            }
        )
        if found:
            jobs.append(
                IpsaeJob(
                    label=model_name,
                    structure_file=structure,
                    pae_file=pae,
                    pae_cutoff=float(pae_cutoff),
                    dist_cutoff=float(dist_cutoff),
                )
            )

    preview = pd.DataFrame(rows, columns=["Model", "Folder", "Structure", "PAE", "Ready"])
    if preview.empty:
        raise FileNotFoundError(
            "Provided folder does not match the AlphaFold Server output folder structure. "
            f"Expected complex subfolders containing files like fold_<name>_model_{model_index}.cif "
            f"and fold_<name>_full_data_{model_index}.json."
        )
    if not jobs:
        raise FileNotFoundError(
            "Provided folder contains AlphaFold-like model files, but no matching full_data JSON files were found. "
            f"Expected each model_{model_index}.cif to have a matching full_data_{model_index}.json file."
        )
    return jobs, preview


discover_af3_model_jobs = discover_af3_models


def run_bulk_ipsae(
    jobs: list[IpsaeJob],
    overwrite: bool = False,
    collect_outputs: bool = False,
    output_dir: str | Path = DEFAULT_EVALS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames = []
    error_rows = []
    for job in jobs:
        try:
            result_frames.append(
                run_ipsae(
                    job,
                    overwrite=overwrite,
                    collect_outputs=collect_outputs,
                    output_dir=output_dir,
                )
            )
        except Exception as exc:
            error_rows.append(
                {
                    "Model": job.label,
                    "Structure_File": _rel(resolve_repo_path(job.structure_file)),
                    "PAE_File": _rel(resolve_repo_path(job.pae_file)),
                    "Error": str(exc),
                }
            )

    scores = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    errors = pd.DataFrame(error_rows)
    return scores, errors
