"""Single-model ipSAE execution helpers for the FreeBindCraft notebook UI."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ADDITIONS_DIR = Path(__file__).resolve().parent
IPSAE_DIR = ADDITIONS_DIR.parent
ROOT = IPSAE_DIR.parent.parent
IPSAE_SCRIPT = IPSAE_DIR / "ipsae.py"
DEFAULT_EVALS_DIR = ROOT / "ipsae_evals"


@dataclass(frozen=True)
class IpsaeJob:
    label: str
    pae_file: Path
    structure_file: Path
    pae_cutoff: float = 10.0
    dist_cutoff: float = 10.0


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "ipsae_model"


def model_name_from_structure(structure_file: str | Path) -> str:
    return _safe_name(Path(structure_file).stem)


def collect_ipsae_outputs(
    job: IpsaeJob,
    output_dir: str | Path = DEFAULT_EVALS_DIR,
) -> dict[str, Path]:
    structure_file = resolve_repo_path(job.structure_file)
    source_files = ipsae_output_files(structure_file, job.pae_cutoff, job.dist_cutoff)
    base = _safe_name(job.label)
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
    scores.insert(1, "PAE_File", _rel(pae_file))
    scores.insert(2, "Structure_File", _rel(structure_file))
    scores.insert(3, "Score_File", _rel(score_file))
    if copied:
        scores.insert(4, "Eval_Output_Dir", _rel(resolve_repo_path(output_dir)))
    return scores
