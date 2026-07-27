"""Summary helpers for collected ipSAE output folders."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from naming import safe_name
from paths import rel_repo_path, resolve_repo_path
from single_model_eval import parse_ipsae_scores


def best_ipsae_summary(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or "ipSAE" not in scores.columns:
        return scores

    working = scores.copy()
    model_col = "Model" if "Model" in working.columns else "Job"
    working["ipSAE_numeric"] = pd.to_numeric(working["ipSAE"], errors="coerce")
    idx = working.groupby(model_col)["ipSAE_numeric"].idxmax().dropna().astype(int)
    columns = [
        col
        for col in [
            model_col,
            "Structure_File",
            "PAE_File",
            "Chn1",
            "Chn2",
            "Type",
            "ipSAE",
            "ipTM_af",
            "pDockQ",
            "LIS",
            "Score_File",
        ]
        if col in working.columns
    ]
    summary = working.loc[idx, columns + ["ipSAE_numeric"]].sort_values("ipSAE_numeric", ascending=False)
    if model_col == "Job":
        summary = summary.rename(columns={"Job": "Model"})
    return summary.drop(columns=["ipSAE_numeric"])


def score_files_in_folder(folder: str | Path) -> list[Path]:
    root = resolve_repo_path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Metrics folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    files = [path for path in sorted(root.rglob("*.txt")) if not path.name.endswith("_byres.txt")]
    if not files:
        raise FileNotFoundError(f"No ipSAE score .txt files found below {root}")
    return files


def summarize_ipsae_folder(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    rows = []
    root = resolve_repo_path(folder)
    for score_file in score_files_in_folder(root):
        try:
            scores = parse_ipsae_scores(score_file)
            if "Model" in scores.columns:
                scores = scores.drop(columns=["Model"])
            model_name = safe_name(score_file.parent.name)
            if not model_name or model_name == safe_name(root.name):
                model_name = re.sub(r"_\d+_\d+$", "", score_file.stem)
            scores.insert(0, "Model", model_name)
            scores.insert(1, "Score_File", rel_repo_path(score_file))
            frames.append(scores)
        except Exception as exc:
            rows.append({"Score_File": rel_repo_path(score_file), "Error": str(exc)})
    all_scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    errors = pd.DataFrame(rows)
    if errors.empty:
        return best_ipsae_summary(all_scores), all_scores
    if all_scores.empty:
        raise RuntimeError("No ipSAE score files could be parsed.")
    return best_ipsae_summary(all_scores), all_scores


def resolve_summary_csv_path(folder: str | Path, csv_path: str | Path = "ipsae_evals_summary.csv") -> Path:
    root = resolve_repo_path(folder)
    value = str(csv_path).strip() or "ipsae_evals_summary.csv"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return root / path
    return resolve_repo_path(path)
