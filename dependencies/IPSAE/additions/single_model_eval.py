"""Single-model ipSAE execution helpers for the FreeBindCraft notebook UI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from af3_pairing import validate_structure_pae_pairing
from naming import ipsae_output_files, safe_name
from paths import DEFAULT_EVALS_DIR, IPSAE_SCRIPT, ROOT, rel_repo_path, resolve_repo_path


@dataclass(frozen=True)
class IpsaeJob:
    label: str
    pae_file: Path
    structure_file: Path
    pae_cutoff: float = 10.0
    dist_cutoff: float = 10.0


def collect_ipsae_outputs(
    job: IpsaeJob,
    output_dir: str | Path = DEFAULT_EVALS_DIR,
) -> dict[str, Path]:
    structure_file = resolve_repo_path(job.structure_file)
    source_files = ipsae_output_files(structure_file, job.pae_cutoff, job.dist_cutoff)
    base = safe_name(job.label)
    out_dir = resolve_repo_path(output_dir) / base
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for key, src in source_files.items():
        if not src.exists():
            continue
        dest = out_dir / src.name
        shutil.copy2(src, dest)
        copied[key] = dest
    return copied


def parse_ipsae_scores(score_file: str | Path) -> pd.DataFrame:
    score_file = resolve_repo_path(score_file)
    if not score_file.exists():
        raise FileNotFoundError(score_file)

    lines = [line.strip() for line in score_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty ipSAE score file: {score_file}")
    columns = lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split(None, len(columns) - 1)
        if len(values) != len(columns):
            raise ValueError(f"Could not parse ipSAE score row in {score_file}: {line}")
        rows.append(values)
    scores = pd.DataFrame(rows, columns=columns)
    for column in scores.columns:
        if column not in {"Chn1", "Chn2", "Type", "Model"}:
            scores[column] = pd.to_numeric(scores[column], errors="ignore")
    return scores


def run_ipsae(
    job: IpsaeJob,
    overwrite: bool = False,
    collect_outputs: bool = False,
    output_dir: str | Path = DEFAULT_EVALS_DIR,
) -> pd.DataFrame:
    pae_file = resolve_repo_path(job.pae_file)
    structure_file = resolve_repo_path(job.structure_file)
    if not pae_file.exists():
        raise FileNotFoundError(f"PAE/confidence file not found: {pae_file}")
    if not structure_file.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file}")
    if not IPSAE_SCRIPT.exists():
        raise FileNotFoundError(f"IPSAE script not found: {IPSAE_SCRIPT}")
    validate_structure_pae_pairing(structure_file, pae_file)

    outputs = ipsae_output_files(structure_file, job.pae_cutoff, job.dist_cutoff)
    if outputs["scores"].exists() and not overwrite:
        scores = parse_ipsae_scores(outputs["scores"])
    else:
        cmd = [
            sys.executable,
            str(IPSAE_SCRIPT),
            str(pae_file),
            str(structure_file),
            str(job.pae_cutoff),
            str(job.dist_cutoff),
        ]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"IPSAE failed: {detail}")
        scores = parse_ipsae_scores(outputs["scores"])

    copied = collect_ipsae_outputs(job, output_dir) if collect_outputs else {}
    score_file = copied.get("scores", outputs["scores"])

    if "Model" in scores.columns:
        scores = scores.drop(columns=["Model"])
    scores.insert(0, "Model", job.label)
    scores.insert(1, "PAE_File", rel_repo_path(pae_file))
    scores.insert(2, "Structure_File", rel_repo_path(structure_file))
    scores.insert(3, "Score_File", rel_repo_path(score_file))
    if copied:
        scores.insert(4, "Eval_Output_Dir", rel_repo_path(resolve_repo_path(output_dir)))
    return scores
