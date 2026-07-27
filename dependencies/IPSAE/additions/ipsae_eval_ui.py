"""
ipSAE evaluation UI for a single model.

Launch from notebooks/ipsae_eval.ipynb:

    from ipsae_eval_ui import launch_ipsae_eval_ui
    launch_ipsae_eval_ui()

The notebook stays intentionally small; execution and output handling live here.
DunbrackLab/IPSAE itself remains in ../ipsae.py.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display
except ImportError:  # Allows import in non-notebook contexts for tests/parsing.
    widgets = None
    clear_output = None
    display = print


ADDITIONS_DIR = Path(__file__).resolve().parent
IPSAE_DIR = ADDITIONS_DIR.parent
ROOT = IPSAE_DIR.parent.parent
IPSAE_SCRIPT = IPSAE_DIR / "ipsae.py"
DEFAULT_EVALS_DIR = ROOT / "ipsae_evals"

BANNER = (
    "background:#e8d5f5;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
SOFT = "color:#57606a;"
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"
INFO = "color:#0969da;"


@dataclass(frozen=True)
class IpsaeJob:
    label: str
    pae_file: Path
    structure_file: Path
    pae_cutoff: float = 10.0
    dist_cutoff: float = 10.0


def _banner(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def _html(text: str) -> "widgets.HTML":
    return widgets.HTML(text)


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
    return pd.read_csv(score_file, sep=r"\s+", engine="python")


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

    scores.insert(0, "Job", job.label)
    scores.insert(1, "PAE_File", _rel(pae_file))
    scores.insert(2, "Structure_File", _rel(structure_file))
    scores.insert(3, "Score_File", _rel(score_file))
    if copied:
        scores.insert(4, "Eval_Output_Dir", _rel(resolve_repo_path(output_dir)))
    return scores


def launch_ipsae_eval_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required to launch this UI.")

    # Replace any previously rendered copy of this UI when the notebook cell is rerun.
    clear_output(wait=True)

    status = widgets.HTML(value=f'<span style="{SOFT}">Ready. Enter structure + PAE paths, then click Run ipSAE.</span>')
    state = {"running": False}

    pae_cutoff = widgets.FloatText(value=10.0, description="PAE cutoff", layout=widgets.Layout(width="220px"))
    dist_cutoff = widgets.FloatText(value=10.0, description="Dist cutoff", layout=widgets.Layout(width="220px"))
    collect_outputs = widgets.Checkbox(value=False, description="Copy outputs to separate folder")
    output_dir = widgets.Text(
        value="ipsae_evals",
        description="Output dir",
        disabled=True,
        layout=widgets.Layout(width="700px"),
    )

    label = widgets.Text(value="single_model", description="Label", layout=widgets.Layout(width="520px"))
    structure_path = widgets.Text(
        value="",
        description="Structure",
        placeholder="AF2/BindCraft: .pdb | AF3: .cif | Boltz: .pdb or .cif",
        layout=widgets.Layout(width="940px"),
    )
    pae_path = widgets.Text(
        value="",
        description="PAE",
        placeholder="AF2/AF3: .json full-data/confidence file | Boltz: .npz PAE file",
        layout=widgets.Layout(width="940px"),
    )
    run_single = widgets.Button(description="Run ipSAE", button_style="primary", icon="play")

    panel = widgets.VBox(
        [
            _banner("Single Model Evaluation"),
            _html(
                f'<span style="{SOFT}">Provide one matching PAE/confidence file and one structure file. '
                "Use PDB for AF2/BindCraft outputs, CIF for AF3 outputs, and NPZ for Boltz PAE files.</span>"
            ),
            label,
            structure_path,
            pae_path,
            widgets.HBox([pae_cutoff, dist_cutoff]),
            collect_outputs,
            output_dir,
            run_single,
            status,
        ]
    )

    def sync_output_dir_state(_change=None) -> None:
        output_dir.disabled = not collect_outputs.value

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def current_job() -> IpsaeJob:
        structure = structure_path.value.strip()
        pae = pae_path.value.strip()
        if not structure:
            raise ValueError("Structure path is empty.")
        if not pae:
            raise ValueError("PAE path is empty.")
        return IpsaeJob(
            label=label.value.strip() or "single_model",
            pae_file=Path(pae),
            structure_file=Path(structure),
            pae_cutoff=float(pae_cutoff.value),
            dist_cutoff=float(dist_cutoff.value),
        )

    def on_run_single(_button) -> None:
        if state["running"]:
            return
        state["running"] = True
        run_single.disabled = True
        set_status("Running ipSAE...", INFO)
        try:
            job = current_job()
            run_ipsae(
                job,
                overwrite=True,
                collect_outputs=collect_outputs.value,
                output_dir=output_dir.value.strip() or DEFAULT_EVALS_DIR,
            )
            structure = resolve_repo_path(job.structure_file)
            msg = f"SUCCESS: ipSAE completed. Outputs saved to {_rel(structure.parent)}"
            if collect_outputs.value:
                copied_dir = resolve_repo_path(output_dir.value.strip() or DEFAULT_EVALS_DIR) / _safe_name(job.label)
                msg += f"<br/>Copied outputs to {_rel(copied_dir)}"
            set_status(msg, OK)
        except Exception as exc:
            set_status(f"FAILED: {exc}", ERR)
        finally:
            state["running"] = False
            run_single.disabled = False

    collect_outputs.observe(sync_output_dir_state, names="value")
    sync_output_dir_state()
    run_single.on_click(on_run_single)

    display(
        widgets.VBox(
            [
                _banner("ipSAE Evaluation"),
                panel,
                _html(
                    f'<span style="{SOFT}">Using original DunbrackLab script: {_rel(IPSAE_SCRIPT)}. '
                    "By default, outputs stay beside the structure file. Enable the copy option to also collect them under specified output folder.</span>"
                ),
            ]
        )
    )


if __name__ == "__main__":
    print(f"IPSAE script: {IPSAE_SCRIPT}")
    print("Launch from notebooks/ipsae_eval.ipynb via launch_ipsae_eval_ui().")
