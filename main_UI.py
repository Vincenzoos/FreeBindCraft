"""
FreeBindCraft interactive UI (ipywidgets).

Launch from FreeBindCraft_UI.ipynb:

    from main_UI import launch_all_ui
    launch_all_ui()

Best viewed in JupyterLab / classic Notebook in a browser.
Cursor's notebook widget renderer may fail with ipywidgetsKernel errors.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets as widgets
from IPython.display import display

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BINDCRAFT_ROOT = Path(__file__).resolve().parent
INPUTS_DIR = BINDCRAFT_ROOT / "inputs"
SETTINGS_TARGET_DIR = BINDCRAFT_ROOT / "settings_target"
SETTINGS_FILTERS_DIR = BINDCRAFT_ROOT / "settings_filters"
SETTINGS_ADVANCED_DIR = BINDCRAFT_ROOT / "settings_advanced"
OUTPUTS_DIR = BINDCRAFT_ROOT / "outputs"
BINDCRAFT_SCRIPT = BINDCRAFT_ROOT / "bindcraft.py"

BANNER = (
    "background:#e8d5f5;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"

RUNNING_JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _ensure_dirs() -> None:
    for d in (INPUTS_DIR, SETTINGS_TARGET_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _list_json(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.glob("*.json") if p.is_file())


def _list_pdb_options(folder: Path) -> List[Tuple[str, str]]:
    """Return (label, absolute_path) pairs for .pdb/.cif files in folder."""
    if not folder.is_dir():
        return []
    files = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".pdb", ".cif")],
        key=lambda p: p.name.lower(),
    )
    return [(p.name, str(p.resolve())) for p in files]


def _banner(text: str) -> widgets.HTML:
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def _parse_lengths(raw: str) -> List[int]:
    raw = (raw or "").strip()
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return [int(val[0]), int(val[1])]
    except Exception:
        pass
    parts = re.split(r"[,\s]+", raw)
    if len(parts) >= 2:
        return [int(parts[0]), int(parts[1])]
    raise ValueError(f"Could not parse lengths from: {raw!r} (expected e.g. [65, 150])")


def _validate_pdb(pdb_path: Path) -> List[str]:
    warnings: List[str] = []
    try:
        from Bio.PDB import PDBParser

        structure = PDBParser(QUIET=True).get_structure("target", str(pdb_path))
        models = list(structure)
        if not models:
            return ["No models found in PDB."]
        for model in models:
            for chain in model:
                residues = [r for r in chain if r.id[0] == " "]
                if not residues:
                    continue
                prev = residues[0].id[1]
                for res in residues[1:]:
                    curr = res.id[1]
                    if curr - prev > 1:
                        warnings.append(
                            f"Possible chain break in chain {chain.id}: {prev} → {curr}"
                        )
                    prev = curr
                for res in residues:
                    atoms = {a.name.strip() for a in res}
                    if "CA" not in atoms and res.resname.strip() not in ("HOH", "WAT"):
                        warnings.append(
                            f"Residue {chain.id}{res.id[1]} ({res.resname}) missing CA atom."
                        )
    except Exception as e:
        warnings.append(f"PDB parse warning: {e}")
    return warnings


def _save_upload(upload_widget: widgets.FileUpload, dest_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    saved: List[Path] = []
    value = upload_widget.value
    # ipywidgets v8+: tuple of UploadFile-like; v7: dict
    if isinstance(value, dict):
        items = [(k, v) for k, v in value.items()]
    else:
        items = []
        for i, item in enumerate(value):
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else f"upload_{i}")
            items.append((name, item))

    for name, item in items:
        if hasattr(item, "name") and hasattr(item, "content"):
            fname = item.name
            content = item.content
        elif isinstance(item, dict):
            fname = item.get("name", name)
            content = item.get("content", b"")
        else:
            fname = name
            content = getattr(item, "content", b"")
        if isinstance(content, memoryview):
            content = content.tobytes()
        ext = Path(fname).suffix.lower()
        if extensions and ext not in extensions:
            continue
        dest = dest_dir / Path(fname).name
        dest.write_bytes(content)
        saved.append(dest)
    return saved



def _count_accepted_designs(design_path: Path) -> int:
    accepted = Path(design_path) / "Accepted"
    if not accepted.is_dir():
        return 0
    return sum(1 for p in accepted.glob("*.pdb") if p.is_file() and not p.name.startswith("."))


def _newest_freebindcraft_log(design_path: Path) -> Optional[Path]:
    """Most recent freebindcraft_YYYYMMDD_HHMMSS.log in a design output folder."""
    logs = [
        p
        for p in Path(design_path).glob("freebindcraft_*.log")
        if p.is_file() and not p.is_symlink()
    ]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _job_name_from_design_path(design_path) -> str:
    """Job name is the design_path folder name (keeps logs and outputs aligned)."""
    name = Path(str(design_path)).name.strip()
    if not name:
        raise ValueError("design_path must include a folder name")
    return name



def _list_gpu_options() -> List[Tuple[str, str]]:
    """Return (label, CUDA_VISIBLE_DEVICES value) pairs from nvidia-smi."""
    options: List[Tuple[str, str]] = [("Auto (default visible GPUs)", "")]
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return options
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx_s, name, total, used, free = parts[:5]
            label = f"GPU {idx_s}: {name} — {free} MiB free / {total} MiB ({used} used)"
            options.append((label, idx_s))
    except Exception:
        pass
    return options

def _load_target_progress_meta(target_json_name: str) -> tuple[Path, int]:
    """Return (design_path, number_of_final_designs) from a settings_target JSON."""
    path = SETTINGS_TARGET_DIR / target_json_name
    data = json.loads(path.read_text())
    design_path = Path(str(data.get("design_path", OUTPUTS_DIR)))
    n_final = int(data.get("number_of_final_designs", 100))
    return design_path, max(1, n_final)


def _tmux_session_name(job_name: str) -> str:
    """Safe tmux session name derived from job/design folder name."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(job_name)).strip("_")
    if not safe:
        safe = "job"
    return safe[:60]


def _tmux_available() -> bool:
    try:
        return subprocess.run(["tmux", "-V"], capture_output=True).returncode == 0
    except Exception:
        return False


def _tmux_has_session(session: str) -> bool:
    try:
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", f"={session}"],
                capture_output=True,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _tmux_kill_session(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", f"={session}"], capture_output=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _job_still_running(info: Dict[str, Any]) -> bool:
    session = info.get("tmux_session")
    if session:
        return _tmux_has_session(session)
    proc = info.get("proc")
    if proc is not None:
        return proc.poll() is None
    process_pid = info.get("process_pid")
    if process_pid:
        return _pid_alive(int(process_pid))
    return False


def _extract_settings_path_from_cmd(cmdline: str) -> Optional[Path]:
    m = re.search(r"--settings\s+([^\s]+)", cmdline)
    if not m:
        return None
    raw = m.group(1).strip().strip("\"'")
    return Path(raw)


def _discover_running_jobs() -> Dict[str, Dict[str, Any]]:
    """Discover running jobs from tmux + active bindcraft.py processes."""
    discovered: Dict[str, Dict[str, Any]] = {}

    # 1) tmux sessions named as job folders
    try:
        proc = subprocess.run(["tmux", "list-sessions", "-F", "#S"], capture_output=True, text=True)
        if proc.returncode == 0:
            for sess in [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]:
                design_path = OUTPUTS_DIR / sess
                if design_path.is_dir():
                    discovered[sess] = {
                        "tmux_session": sess,
                        "design_path": str(design_path),
                        "n_final": 1,
                    }
    except Exception:
        pass

    # 2) live bindcraft.py processes (covers non-tmux legacy runs)
    try:
        proc = subprocess.run(["pgrep", "-af", "python.*bindcraft.py"], capture_output=True, text=True)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2 or not parts[0].isdigit():
                    continue
                pid = int(parts[0])
                cmd = parts[1]
                settings_path = _extract_settings_path_from_cmd(cmd)
                if settings_path is None:
                    continue
                if not settings_path.is_absolute():
                    settings_path = (BINDCRAFT_ROOT / settings_path).resolve()
                if not settings_path.is_file():
                    continue
                try:
                    data = json.loads(settings_path.read_text())
                    design_path = Path(str(data.get("design_path", OUTPUTS_DIR)))
                    job_name = _job_name_from_design_path(design_path)
                    discovered[job_name] = {
                        "process_pid": pid,
                        "design_path": str(design_path),
                        "n_final": int(data.get("number_of_final_designs", 1)),
                        "settings_path": str(settings_path),
                    }
                except Exception:
                    continue
    except Exception:
        pass

    return discovered


def _resolve_n_final(info: Dict[str, Any], design_path: Path) -> int:
    """Best-effort target accepted-design count for a running/discovered job."""
    try:
        n = int(info.get("n_final") or 0)
    except Exception:
        n = 0
    if n > 1:
        return n
    settings_path = info.get("settings_path")
    if settings_path:
        try:
            data = json.loads(Path(settings_path).read_text())
            return max(1, int(data.get("number_of_final_designs", n or 1)))
        except Exception:
            pass
    try:
        design_resolved = Path(design_path).resolve()
        for jp in SETTINGS_TARGET_DIR.glob("*.json"):
            try:
                data = json.loads(jp.read_text())
                if Path(str(data.get("design_path", ""))).resolve() == design_resolved:
                    return max(1, int(data.get("number_of_final_designs", n or 1)))
            except Exception:
                continue
    except Exception:
        pass
    return max(1, n or 1)


def _refresh_running_job_names() -> List[str]:
    with JOBS_LOCK:
        # prune stale in-memory jobs
        for name, info in list(RUNNING_JOBS.items()):
            if not _job_still_running(info):
                RUNNING_JOBS.pop(name, None)

        # merge discovered runtime jobs
        for name, info in _discover_running_jobs().items():
            if name in RUNNING_JOBS:
                RUNNING_JOBS[name].update({k: v for k, v in info.items() if k not in RUNNING_JOBS[name]})
            else:
                RUNNING_JOBS[name] = info

        # Fill in n_final for discovered tmux jobs that still have the placeholder
        for name, info in list(RUNNING_JOBS.items()):
            design_path = Path(info.get("design_path", OUTPUTS_DIR / name))
            info["n_final"] = _resolve_n_final(info, design_path)
            if not info.get("log"):
                log_hint = _newest_freebindcraft_log(design_path)
                if log_hint:
                    info["log"] = str(log_hint)

        return sorted(RUNNING_JOBS.keys())


def launch_all_ui() -> None:
    """Build and display the FreeBindCraft ipywidgets UI inline in the notebook."""
    _ensure_dirs()
    if not BINDCRAFT_SCRIPT.exists():
        raise FileNotFoundError(f"bindcraft.py not found at {BINDCRAFT_SCRIPT}")

    state: Dict[str, Any] = {
        "job_name": None,
        "target_json": None,
        "run_script": None,
        "progress_job": None,  # job name shown on the progress bar
    }

    welcome = widgets.HTML(
        f"""
        <div style="{BANNER}">
          <h2 style="margin:0 0 8px 0;">Welcome to the FreeBindCraft User Interface</h2>
          <ol style="margin:0;padding-left:20px;">
            <li>Upload a target PDB file.</li>
            <li>Upload or create a settings target JSON file (must contain the PDB path).</li>
            <li>Select settings target JSON, filters, and advanced settings files.</li>
            <li>Job name is automatic from <code>design_path</code> folder name.</li>
            <li>Generate the run script, then Run FreeBindCraft.</li>
          </ol>
        </div>
        """
    )

    # --- Uploads ---
    upload_intro = widgets.HTML("<p>Next, you can upload a PDB or a target JSON file:</p>")
    pdb_banner = _banner(f"Step 1: If not already present, upload a PDB file to {INPUTS_DIR}/")
    pdb_upload = widgets.FileUpload(
        accept=".pdb,.cif", multiple=True, description="Upload PDB", button_style="info"
    )
    pdb_status = widgets.HTML("")

    json_banner = _banner(f"Step 2: If not already present, upload a JSON file to {SETTINGS_TARGET_DIR}/")
    json_upload = widgets.FileUpload(
        accept=".json", multiple=True, description="Upload JSON", button_style="info"
    )
    json_status = widgets.HTML("")
    upload_note = widgets.HTML(
        "<p style='font-size:0.95em;color:#555;'>"
        "Uploaded PDBs are checked for chain breaks, mislabeled atoms and other "
        "inconsistencies. Problems are reported but not auto-fixed.</p>"
    )

    def on_pdb_upload(change):
        if not change["new"]:
            return
        saved = _save_upload(pdb_upload, INPUTS_DIR, (".pdb", ".cif"))
        msgs = []
        for path in saved:
            warns = _validate_pdb(path) if path.suffix.lower() == ".pdb" else []
            if warns:
                msgs.append(
                    f"<span style='{ERR}'>Saved {path.name} with warnings:</span><ul>"
                    + "".join(f"<li>{w}</li>" for w in warns[:8])
                    + "</ul>"
                )
            else:
                msgs.append(f"<span style='{OK}'>Saved {path} (OK)</span>")
        pdb_status.value = "<br>".join(msgs)
        pdb_upload.description = f"Upload PDB ({len(saved)})"
        refresh_pdb_dropdown(select=str(saved[-1].resolve()) if saved else None)

    def on_json_upload(change):
        if not change["new"]:
            return
        saved = _save_upload(json_upload, SETTINGS_TARGET_DIR, (".json",))
        if saved:
            json_status.value = "<br>".join(
                f"<span style='{OK}'>Saved {p}</span>" for p in saved
            )
            json_upload.description = f"Upload JSON ({len(saved)})"
            refresh_target_dropdown()

    pdb_upload.observe(on_pdb_upload, names="value")
    json_upload.observe(on_json_upload, names="value")

    # --- Target JSON editor ---
    editor_banner = _banner("Or you can edit/create a new target JSON file.")
    style = {"description_width": "140px"}
    layout = widgets.Layout(width="70%")
    # PDB list first so File Name / binder_name can default from it
    pdb_name_to_path = {name: p for name, p in _list_pdb_options(INPUTS_DIR)}
    pdb_names = list(pdb_name_to_path.keys()) or ["(no PDB files in inputs/)"]
    initial_pdb = pdb_names[0]

    def _defaults_from_pdb_name(pdb_name: str) -> tuple[str, str, str]:
        """Return (json_filename, binder_name, design_path) from a PDB basename."""
        if not pdb_name or pdb_name.startswith("("):
            stem = "my_target"
            return f"{stem}.json", "My-Binder", str(OUTPUTS_DIR / f"{stem}_freebindcraft")
        stem = Path(pdb_name).stem
        return (
            f"{stem}.json",
            f"{stem}-Binder",
            str(OUTPUTS_DIR / f"{stem}_freebindcraft"),
        )

    init_json, init_binder, init_design = _defaults_from_pdb_name(initial_pdb)

    file_name_w = widgets.Text(value=init_json, description="File Name:", layout=layout)
    design_path_w = widgets.Text(
        value=init_design, description="design_path:", layout=layout, style=style
    )
    binder_name_w = widgets.Text(
        value=init_binder, description="binder_name:", layout=layout, style=style
    )
    starting_pdb_w = widgets.Dropdown(
        options=pdb_names,
        value=initial_pdb,
        description="starting_pdb:",
        layout=layout,
        style=style,
    )
    refresh_pdb_btn = widgets.Button(
        description="Refresh PDB list",
        icon="refresh",
        button_style="warning",
        layout=widgets.Layout(width="180px"),
    )

    def _selected_pdb_path() -> str:
        name = starting_pdb_w.value
        if not name or name.startswith("("):
            return ""
        return pdb_name_to_path.get(name, str(INPUTS_DIR / name))

    def _sync_defaults_from_pdb(change=None):
        """Keep File Name, binder_name, and design_path aligned with starting_pdb."""
        json_name, binder, design = _defaults_from_pdb_name(starting_pdb_w.value)
        file_name_w.value = json_name
        binder_name_w.value = binder
        design_path_w.value = design

    def refresh_pdb_dropdown(select: Optional[str] = None, _=None):
        nonlocal pdb_name_to_path
        pdb_name_to_path = {name: p for name, p in _list_pdb_options(INPUTS_DIR)}
        names = list(pdb_name_to_path.keys())
        if not names:
            starting_pdb_w.options = ["(no PDB files in inputs/)"]
            starting_pdb_w.value = "(no PDB files in inputs/)"
            return
        starting_pdb_w.options = names
        if select:
            sel_name = Path(select).name
            if sel_name in names:
                starting_pdb_w.value = sel_name
                return
            for n, p in pdb_name_to_path.items():
                if p == select:
                    starting_pdb_w.value = n
                    return
        if starting_pdb_w.value not in names:
            starting_pdb_w.value = names[0]

    starting_pdb_w.observe(_sync_defaults_from_pdb, names="value")
    refresh_pdb_btn.on_click(lambda _: refresh_pdb_dropdown())
    pdb_picker_row = widgets.VBox([starting_pdb_w, refresh_pdb_btn])
    chains_w = widgets.Text(value="A", description="chains:", layout=layout, style=style)
    hotspots_w = widgets.Text(
        value="",
        description="hotspots:",
        layout=layout,
        style=style,
        placeholder="e.g. A56,A60-65 (empty = no preference)",
    )
    lengths_w = widgets.Text(value="[65, 150]", description="lengths:", layout=layout, style=style)
    n_designs_w = widgets.IntText(
        value=100, description="num designs:", layout=layout, style=style
    )
    save_json_btn = widgets.Button(
        description="Save Changes",
        button_style="success",
        layout=widgets.Layout(width="70%", height="40px"),
    )
    save_json_status = widgets.HTML("")

    def on_save_json(_):
        try:
            fname = file_name_w.value.strip()
            if not fname.endswith(".json"):
                fname += ".json"
            pdb_sel = _selected_pdb_path()
            if not pdb_sel or not Path(pdb_sel).is_file():
                save_json_status.value = (
                    f"<span style='{ERR}'>Select a valid starting_pdb from the dropdown "
                    f"(upload a PDB to inputs/ first if the list is empty).</span>"
                )
                return
            payload = {
                "design_path": design_path_w.value.strip(),
                "binder_name": binder_name_w.value.strip(),
                "starting_pdb": pdb_sel,
                "chains": chains_w.value.strip(),
                "target_hotspot_residues": hotspots_w.value.strip(),
                "lengths": _parse_lengths(lengths_w.value),
                "number_of_final_designs": int(n_designs_w.value),
            }
            out = SETTINGS_TARGET_DIR / fname
            out.write_text(json.dumps(payload, indent=4) + "\n")
            Path(payload["design_path"]).mkdir(parents=True, exist_ok=True)
            save_json_status.value = f"<span style='{OK}'>Saved {out}</span>"
            refresh_target_dropdown()
            if fname in list(target_dropdown.options):
                target_dropdown.value = fname
        except Exception as e:
            save_json_status.value = f"<span style='{ERR}'>Error: {e}</span>"

    save_json_btn.on_click(on_save_json)

    # --- Target JSON selectors (job name = design_path folder) ---
    job_banner = _banner(
        "Step 3–6: Select your settings files and run FreeBindCraft"
    )
    job_help = widgets.HTML(
        f"<p><b>Job name</b> is taken from the target JSON <code>design_path</code> folder "
        f"(e.g. <code>.../outputs/PD1_freebindcraft</code> → job <code>PD1_freebindcraft</code>). "
        f"Logs and design outputs share that folder under <code>{OUTPUTS_DIR}</code>.</p>"
    )
    job_name_w = widgets.Text(
        value=_job_name_from_design_path(design_path_w.value),
        description="Job name:",
        disabled=True,
        layout=widgets.Layout(width="70%"),
        style={"description_width": "140px"},
    )
    job_status = widgets.HTML(
        f"<span style='{OK}'>Job name locked to design_path folder: "
        f"<b>{_job_name_from_design_path(design_path_w.value)}</b></span>"
    )

    def _sync_job_name_from_design_path(change=None):
        try:
            name = _job_name_from_design_path(design_path_w.value)
        except Exception:
            job_name_w.value = ""
            state["job_name"] = None
            job_status.value = f"<span style='{ERR}'>Invalid design_path — cannot derive job name.</span>"
            return
        job_name_w.value = name
        state["job_name"] = name
        job_dir = Path(design_path_w.value.strip())
        job_dir.mkdir(parents=True, exist_ok=True)
        job_status.value = (
            f"<span style='{OK}'>Job name = design_path folder: <b>{name}</b> "
            f"(outputs/logs → {job_dir})</span>"
        )

    design_path_w.observe(_sync_job_name_from_design_path, names="value")
    _sync_job_name_from_design_path()

    target_opts = _list_json(SETTINGS_TARGET_DIR) or ["(none)"]
    target_dropdown = widgets.Dropdown(
        options=target_opts,
        description="Target JSON:",
        layout=widgets.Layout(width="60%"),
        style={"description_width": "120px"},
    )
    target_sel_status = widgets.HTML("")
    refresh_target_btn = widgets.Button(
        description="Refresh Json Dropdown Menu",
        icon="refresh",
        button_style="warning",
        layout=widgets.Layout(width="280px"),
    )

    def load_target_into_form(fname: str) -> None:
        path = SETTINGS_TARGET_DIR / fname
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text())
            # Set PDB first (may sync defaults), then overwrite from JSON
            desired = str(data.get("starting_pdb", "")).strip()
            refresh_pdb_dropdown(select=desired)
            file_name_w.value = fname
            design_path_w.value = str(data.get("design_path", ""))
            binder_name_w.value = str(data.get("binder_name", ""))
            chains_w.value = str(data.get("chains", "A"))
            hotspots_w.value = str(data.get("target_hotspot_residues", ""))
            lengths_w.value = str(data.get("lengths", [65, 150]))
            n_designs_w.value = int(data.get("number_of_final_designs", 100))
        except Exception:
            pass

    def refresh_target_dropdown(_=None):
        opts = _list_json(SETTINGS_TARGET_DIR)
        target_dropdown.options = opts if opts else ["(none)"]
        if opts and target_dropdown.value not in opts:
            target_dropdown.value = opts[0]
        if target_dropdown.value and target_dropdown.value != "(none)":
            target_sel_status.value = f"Selected Target JSON: <b>{target_dropdown.value}</b>"
            state["target_json"] = target_dropdown.value
            load_target_into_form(target_dropdown.value)

    def on_target_change(change):
        if change["new"] and change["new"] != "(none)":
            state["target_json"] = change["new"]
            target_sel_status.value = f"Selected Target JSON: <b>{change['new']}</b>"
            load_target_into_form(change["new"])

    target_dropdown.observe(on_target_change, names="value")
    refresh_target_btn.on_click(refresh_target_dropdown)
    refresh_target_dropdown()

    filter_opts = _list_json(SETTINGS_FILTERS_DIR)
    advanced_opts = _list_json(SETTINGS_ADVANCED_DIR)
    filters_dropdown = widgets.Dropdown(
        options=filter_opts,
        value="default_filters.json" if "default_filters.json" in filter_opts else None,
        description="Filters:",
        layout=widgets.Layout(width="60%"),
        style={"description_width": "120px"},
    )
    advanced_dropdown = widgets.Dropdown(
        options=advanced_opts,
        value="default_4stage_multimer.json"
        if "default_4stage_multimer.json" in advanced_opts
        else None,
        description="Advanced:",
        layout=widgets.Layout(width="60%"),
        style={"description_width": "120px"},
    )

    options_banner = _banner("FreeBindCraft options")
    gpu_opts = _list_gpu_options()
    # Prefer freest GPU if available (skip Auto entry)
    default_gpu = ""
    try:
        best = None
        for label, val in gpu_opts:
            if not val:
                continue
            # parse "… — X MiB free"
            if " MiB free" in label:
                free = int(label.split(" — ")[1].split(" MiB free")[0].replace(",", ""))
                if best is None or free > best[0]:
                    best = (free, val)
        if best is not None:
            default_gpu = best[1]
    except Exception:
        default_gpu = gpu_opts[1][1] if len(gpu_opts) > 1 else ""

    gpu_dropdown = widgets.Dropdown(
        options=gpu_opts,
        value=default_gpu if any(v == default_gpu for _, v in gpu_opts) else gpu_opts[0][1],
        description="GPU:",
        layout=widgets.Layout(width="85%"),
        style={"description_width": "120px"},
    )
    refresh_gpu_btn = widgets.Button(
        description="Refresh GPU list",
        icon="refresh",
        button_style="info",
        layout=widgets.Layout(width="180px"),
    )
    gpu_status = widgets.HTML(
        "<span style='color:#555;'>Sets <code>CUDA_VISIBLE_DEVICES</code> for the FreeBindCraft job.</span>"
    )

    def refresh_gpu_dropdown(_=None):
        opts = _list_gpu_options()
        cur = gpu_dropdown.value
        gpu_dropdown.options = opts
        values = [v for _, v in opts]
        gpu_dropdown.value = cur if cur in values else (opts[0][1] if opts else "")

    refresh_gpu_btn.on_click(refresh_gpu_dropdown)

    no_pyrosetta_w = widgets.Checkbox(value=True, description="--no-pyrosetta (OpenMM bypass)")
    rank_by_w = widgets.Dropdown(
        options=["i_pTM", "ipSAE"],
        value="i_pTM",
        description="Rank by:",
        style={"description_width": "120px"},
    )
    verbose_w = widgets.Checkbox(value=False, description="Verbose")
    no_plots_w = widgets.Checkbox(value=False, description="Disable plots")
    no_anims_w = widgets.Checkbox(value=False, description="Disable animations")

    select_help = widgets.HTML(
        "<p>Select a <code>settings_filters</code> file and a "
        "<code>settings_advanced</code> file, then generate the run script.</p>"
    )
    generate_btn = widgets.Button(
        description="Generate BindCraft Run Script with Settings",
        button_style="success",
        layout=widgets.Layout(width="70%", height="42px"),
    )
    script_preview = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            max_height="220px",
            overflow="auto",
            width="90%",
        )
    )

    def build_command() -> List[str]:
        if not state.get("target_json") or state["target_json"] == "(none)":
            raise ValueError("Select a target JSON file.")
        if not filters_dropdown.value or not advanced_dropdown.value:
            raise ValueError("Select filters and advanced settings files.")
        design_path, _n_final = _load_target_progress_meta(state["target_json"])
        state["job_name"] = _job_name_from_design_path(design_path)
        job_name_w.value = state["job_name"]
        # Keep editor design_path aligned with selected JSON
        design_path_w.value = str(design_path)
        cmd = [
            "python",
            "-u",
            str(BINDCRAFT_SCRIPT),
            "--settings",
            str(SETTINGS_TARGET_DIR / state["target_json"]),
            "--filters",
            str(SETTINGS_FILTERS_DIR / filters_dropdown.value),
            "--advanced",
            str(SETTINGS_ADVANCED_DIR / advanced_dropdown.value),
            "--rank-by",
            rank_by_w.value,
        ]
        if no_pyrosetta_w.value:
            cmd.append("--no-pyrosetta")
        if verbose_w.value:
            cmd.append("--verbose")
        if no_plots_w.value:
            cmd.append("--no-plots")
        if no_anims_w.value:
            cmd.append("--no-animations")
        return cmd

    def on_generate(_):
        script_preview.clear_output()
        try:
            cmd = build_command()
            state["run_script"] = cmd
            with script_preview:
                print("Generated command:\n")
                gpu = gpu_dropdown.value
                if gpu:
                    print(f"CUDA_VISIBLE_DEVICES={gpu}")
                else:
                    print("CUDA_VISIBLE_DEVICES=(unset — use default visible GPUs)")
                print(" \\\n  ".join(cmd))
                print("\nReady. Press Run FreeBindCraft to start.")
        except Exception as e:
            with script_preview:
                print(f"Error: {e}")

    generate_btn.on_click(on_generate)

    # --- Run / Abort ---
    run_help = widgets.HTML(
        "<p>Press <b>Run FreeBindCraft</b> to start the job in a detached <code>tmux</code> session "
        "(survives notebook kernel death). Reattach with <code>tmux attach -t &lt;job&gt;</code>.</p>"
    )
    run_btn = widgets.Button(
        description="Run FreeBindCraft",
        button_style="primary",
        layout=widgets.Layout(width="70%", height="48px"),
    )
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=1,
        description="Accepted:",
        bar_style="info",
        orientation="horizontal",
        layout=widgets.Layout(width="70%", height="28px"),
        style={"description_width": "80px", "bar_color": "#4c8bf5"},
    )
    progress_status = widgets.HTML(
        "<span style='color:#666;'>Progress will appear here when a job is running.</span>"
    )
    run_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ccc",
            padding="8px",
            max_height="360px",
            overflow="auto",
            width="90%",
        )
    )
    abort_help = widgets.HTML(
        "<p>Select a running job below to watch its <b>progress bar</b>, then click "
        "<b>Abort a running job</b> if you need to stop it. "
        "Use <b>Refresh Jobs Dropdown</b> if it is missing.</p>"
    )
    abort_dropdown = widgets.Dropdown(
        options=[],
        description="Select a job:",
        layout=widgets.Layout(width="60%"),
        style={"description_width": "120px"},
    )
    abort_btn = widgets.Button(
        description="Abort a running job",
        button_style="warning",
        layout=widgets.Layout(width="280px"),
    )
    refresh_jobs_btn = widgets.Button(
        description="Refresh Jobs Dropdown",
        icon="refresh",
        button_style="warning",
        layout=widgets.Layout(width="280px"),
    )
    abort_status = widgets.HTML("")

    def refresh_jobs_dropdown(_=None):
        names = _refresh_running_job_names()
        prev = abort_dropdown.value
        watched = state.get("progress_job")
        abort_dropdown.options = names
        if names:
            if prev in names:
                abort_dropdown.value = prev
            elif watched in names:
                abort_dropdown.value = watched
            else:
                abort_dropdown.value = names[0]
            state["progress_job"] = abort_dropdown.value
            _show_progress_for_job(abort_dropdown.value, force=True)
        else:
            state["progress_job"] = None
            progress_status.value = (
                "<span style='color:#666;'>Select a running job below to watch progress.</span>"
            )

    def _update_progress_ui(
        *,
        accepted: int,
        target: int,
        trajectories: int,
        stage: str,
        running: bool,
        done: bool = False,
        failed: bool = False,
        extra: str = "",
        job_name: Optional[str] = None,
    ) -> None:
        progress_bar.max = max(1, int(target))
        progress_bar.value = min(int(accepted), progress_bar.max)
        if failed:
            progress_bar.bar_style = "danger"
        elif done or accepted >= target:
            progress_bar.bar_style = "success"
        elif running:
            progress_bar.bar_style = "info"
        else:
            progress_bar.bar_style = ""
        stage_txt = stage or ("finished" if done else "idle")
        extra_txt = f" · {extra}" if extra else ""
        job_txt = f"<code>{job_name}</code> · " if job_name else ""
        progress_status.value = (
            f"{job_txt}<b>{accepted}/{target}</b> accepted designs"
            f" · trajectories: <b>{trajectories}</b>"
            f" · current: <b>{stage_txt}</b>{extra_txt}"
        )

    def _parse_log_line(stripped: str, trajectories: int, stage: str):
        if stripped.startswith("Starting trajectory:"):
            return trajectories + 1, "trajectory started"
        if stripped.startswith("Stage "):
            return trajectories, stripped
        if "MPNN" in stripped and (
            "redesign" in stripped.lower()
            or "optim" in stripped.lower()
            or "sequence" in stripped.lower()
        ):
            return trajectories, stripped[:80]
        if stripped.startswith("Found ") and "MPNN designs passing" in stripped:
            return trajectories, stripped
        if "Target number" in stripped and "designs reached" in stripped:
            return trajectories, "target reached — ranking"
        if stripped.startswith("Running binder design"):
            return trajectories, "initializing"
        return trajectories, stage

    def _read_log_progress(log_path: Optional[Path]) -> Tuple[int, str]:
        trajectories = 0
        stage = "running"
        if not log_path or not Path(log_path).is_file():
            return 0, "no log yet"
        try:
            with open(log_path, "r", errors="replace") as logf:
                for line in logf:
                    trajectories, stage = _parse_log_line(
                        line.strip(), trajectories, stage
                    )
        except Exception:
            pass
        return trajectories, stage

    def _show_progress_for_job(job_name: Optional[str], *, force: bool = False) -> None:
        """Refresh the progress bar for the job selected in the dropdown."""
        if not job_name:
            progress_status.value = (
                "<span style='color:#666;'>Select a running job below to watch progress.</span>"
            )
            return
        if not force and state.get("progress_job") != job_name:
            return
        with JOBS_LOCK:
            info = dict(RUNNING_JOBS.get(job_name) or {})
        design_path = Path(info.get("design_path") or (OUTPUTS_DIR / job_name))
        n_final = _resolve_n_final(info, design_path)
        log_raw = info.get("log")
        log_path = Path(log_raw) if log_raw else _newest_freebindcraft_log(design_path)
        trajectories, stage = _read_log_progress(log_path)
        if info:
            alive = _job_still_running(info)
        else:
            alive = _tmux_has_session(_tmux_session_name(job_name))
        accepted = _count_accepted_designs(design_path)
        done = (not alive) and accepted >= n_final
        failed = bool(info.get("aborted")) and not alive
        if not alive and not done and not failed:
            stage = stage if stage not in ("running", "no log yet") else "tmux session ended"
        elif alive and stage in ("running", "no log yet"):
            stage = "running" if trajectories else "starting"
        _update_progress_ui(
            accepted=accepted,
            target=n_final,
            trajectories=trajectories,
            stage=stage,
            running=alive,
            done=done,
            failed=failed,
            job_name=job_name,
        )

    def on_abort_job_selected(change=None):
        if isinstance(change, dict) and change.get("name") != "value":
            return
        name = abort_dropdown.value
        state["progress_job"] = name
        _show_progress_for_job(name, force=True)

    def progress_poll_loop():
        """Keep the progress bar in sync with the job selected in the dropdown."""
        while True:
            try:
                name = state.get("progress_job") or abort_dropdown.value
                if name and state.get("progress_job") == name:
                    _show_progress_for_job(name)
            except Exception:
                pass
            time.sleep(1.0)

    def monitor_tmux_job(
        job_name: str,
        session: str,
        log_path: Path,
        exit_path: Path,
        design_path: Path,
        n_final: int,
    ) -> None:
        """Tail log + poll Accepted/ while tmux session lives (survives kernel? no — monitor dies with kernel, job continues)."""
        trajectories = 0
        stage = "starting in tmux"
        pos = 0
        try:
            with run_output:
                print(f"[{job_name}] attached to tmux session '{session}'")
                print(f"Reattach anytime: tmux attach -t {session}")
                print(f"Log: {log_path}\n")

            while True:
                if log_path.is_file():
                    with open(log_path, "r", errors="replace") as logf:
                        logf.seek(pos)
                        chunk = logf.read()
                        pos = logf.tell()
                    if chunk:
                        with run_output:
                            print(chunk, end="")
                        for line in chunk.splitlines():
                            trajectories, stage = _parse_log_line(
                                line.strip(), trajectories, stage
                            )

                accepted = _count_accepted_designs(design_path)
                alive = _tmux_has_session(session)
                # Progress bar follows the dropdown selection
                if state.get("progress_job") == job_name:
                    _update_progress_ui(
                        accepted=accepted,
                        target=n_final,
                        trajectories=trajectories,
                        stage=stage if alive else stage,
                        running=alive,
                        job_name=job_name,
                    )
                if not alive:
                    break
                time.sleep(1.0)

            # Session ended — read exit code if present
            time.sleep(0.5)
            if log_path.is_file():
                with open(log_path, "r", errors="replace") as logf:
                    logf.seek(pos)
                    chunk = logf.read()
                    if chunk:
                        with run_output:
                            print(chunk, end="")
                        for line in chunk.splitlines():
                            trajectories, stage = _parse_log_line(
                                line.strip(), trajectories, stage
                            )

            exit_code = None
            if exit_path.is_file():
                try:
                    exit_code = int(exit_path.read_text().strip().splitlines()[-1])
                except Exception:
                    exit_code = None

            accepted = _count_accepted_designs(design_path)
            failed = exit_code not in (0, None)
            # If no exit file, treat as unknown/success if accepted reached target
            if exit_code is None:
                failed = False
                stage = "tmux session ended"
            else:
                stage = "finished" if exit_code == 0 else f"exited ({exit_code})"

            if state.get("progress_job") == job_name:
                _update_progress_ui(
                    accepted=accepted,
                    target=n_final,
                    trajectories=trajectories,
                    stage=stage,
                    running=False,
                    done=not failed,
                    failed=failed,
                    job_name=job_name,
                )
            with run_output:
                print(f"\n[{job_name}] tmux session ended (exit={exit_code})")
                print(f"Accepted designs: {accepted}/{n_final} in {design_path / 'Accepted'}")
                print(f"(Job was detached — kernel death would not stop it.)")
        finally:
            with JOBS_LOCK:
                RUNNING_JOBS.pop(job_name, None)
            refresh_jobs_dropdown()

    def on_run(_):
        run_output.clear_output()
        try:
            if not _tmux_available():
                raise RuntimeError(
                    "tmux not found. Install tmux so jobs can survive notebook kernel death."
                )

            cmd = state.get("run_script") or build_command()
            state["run_script"] = cmd
            design_path, n_final = _load_target_progress_meta(state["target_json"])
            job_name = _job_name_from_design_path(design_path)
            state["job_name"] = job_name
            job_name_w.value = job_name
            session = _tmux_session_name(job_name)

            with JOBS_LOCK:
                existing = RUNNING_JOBS.get(job_name)
                if existing and _job_still_running(existing):
                    with run_output:
                        print(
                            f"Job '{job_name}' is already running "
                            f"(tmux session '{existing.get('tmux_session', session)}')."
                        )
                    return
            if _tmux_has_session(session):
                with run_output:
                    print(
                        f"tmux session '{session}' already exists. "
                        f"Attach with: tmux attach -t {session}\n"
                        f"Or abort it from the UI / run: tmux kill-session -t {session}"
                    )
                log_hint = _newest_freebindcraft_log(design_path)
                with JOBS_LOCK:
                    RUNNING_JOBS[job_name] = {
                        "tmux_session": session,
                        "log": str(log_hint) if log_hint else "",
                        "design_path": str(design_path),
                        "n_final": n_final,
                    }
                refresh_jobs_dropdown()
                abort_dropdown.value = job_name
                state["progress_job"] = job_name
                _show_progress_for_job(job_name, force=True)
                return

            design_path.mkdir(parents=True, exist_ok=True)
            state["progress_job"] = job_name
            _update_progress_ui(
                accepted=_count_accepted_designs(design_path),
                target=n_final,
                trajectories=0,
                stage="launching tmux",
                running=True,
                job_name=job_name,
            )

            job_dir = design_path
            job_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = job_dir / f"freebindcraft_{stamp}.log"
            exit_path = job_dir / f"freebindcraft_{stamp}.exit"
            gpu_id = (gpu_dropdown.value or "").strip()
            cuda_export = f"export CUDA_VISIBLE_DEVICES={gpu_id}\n" if gpu_id else ""

            script_path = job_dir / "run_command.sh"
            script_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                "cd "
                + json.dumps(str(BINDCRAFT_ROOT))
                + "\n"
                + cuda_export
                + "export PYTHONUNBUFFERED=1\n"
                + f"LOG_FILE={json.dumps(str(log_path))}\n"
                + f"EXIT_FILE={json.dumps(str(exit_path))}\n"
                + "echo \"[tmux] starting FreeBindCraft in session\" | tee -a \"$LOG_FILE\"\n"
                + "set +e\n"
                + " ".join(json.dumps(c) for c in cmd)
                + " 2>&1 | tee -a \"$LOG_FILE\"\n"
                + "ec=${PIPESTATUS[0]}\n"
                + "echo \"$ec\" > \"$EXIT_FILE\"\n"
                + "echo \"[tmux] finished with exit code $ec\" | tee -a \"$LOG_FILE\"\n"
                + "exit \"$ec\"\n"
            )
            script_path.chmod(script_path.stat().st_mode | 0o755)

            # Detached tmux session — survives notebook kernel death
            launch = subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session,
                    "bash",
                    str(script_path),
                ],
                capture_output=True,
                text=True,
            )
            if launch.returncode != 0:
                raise RuntimeError(
                    f"Failed to start tmux session '{session}': {launch.stderr or launch.stdout}"
                )

            with JOBS_LOCK:
                RUNNING_JOBS[job_name] = {
                    "tmux_session": session,
                    "log": str(log_path),
                    "exit_file": str(exit_path),
                    "design_path": str(design_path),
                    "n_final": n_final,
                }
            refresh_jobs_dropdown()
            abort_dropdown.value = job_name
            state["progress_job"] = job_name

            with run_output:
                print(f"Started tmux session: {session}")
                if gpu_id:
                    print(f"CUDA_VISIBLE_DEVICES={gpu_id}")
                else:
                    print("CUDA_VISIBLE_DEVICES=(unset)")
                print(" ".join(cmd))
                print(f"Design path: {design_path}")
                print(f"Target accepted designs: {n_final}")
                print(f"Log: {log_path}")
                print(f"Attach: tmux attach -t {session}")
                print(f"Kill:   tmux kill-session -t {session}\n")

            threading.Thread(
                target=monitor_tmux_job,
                args=(job_name, session, log_path, exit_path, design_path, n_final),
                daemon=True,
            ).start()
        except Exception as e:
            _update_progress_ui(
                accepted=0,
                target=max(1, progress_bar.max),
                trajectories=0,
                stage="error",
                running=False,
                failed=True,
                extra=str(e),
            )
            with run_output:
                print(f"Error starting job: {e}")

    def on_abort(_):
        name = abort_dropdown.value
        if not name:
            abort_status.value = f"<span style='{ERR}'>No job selected.</span>"
            return
        with JOBS_LOCK:
            info = RUNNING_JOBS.get(name)
        if not info:
            # refresh discovery and retry
            refresh_jobs_dropdown()
            with JOBS_LOCK:
                info = RUNNING_JOBS.get(name)
            if not info:
                # Try killing by conventional session name anyway
                session = _tmux_session_name(name)
                if _tmux_has_session(session):
                    _tmux_kill_session(session)
                    abort_status.value = (
                        f"<span style='{OK}'>Killed tmux session '{session}'.</span>"
                    )
                else:
                    abort_status.value = (
                        f"<span style='{ERR}'>Job not found / already finished.</span>"
                    )
                refresh_jobs_dropdown()
                return

        session = info.get("tmux_session") or _tmux_session_name(name)
        try:
            if session and _tmux_has_session(session):
                _tmux_kill_session(session)
            proc = info.get("proc")
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            process_pid = info.get("process_pid")
            if process_pid and _pid_alive(int(process_pid)):
                try:
                    os.kill(int(process_pid), signal.SIGTERM)
                except Exception:
                    pass
            abort_status.value = (
                f"<span style='{OK}'>Aborted job '{name}' "
                f"(tmux session '{session}').</span>"
            )
            design_path = Path(info.get("design_path", OUTPUTS_DIR / name))
            n_final = _resolve_n_final(info, design_path)
            log_raw = info.get("log")
            log_path = Path(log_raw) if log_raw else _newest_freebindcraft_log(design_path)
            trajectories, _stage = _read_log_progress(log_path)
            state["progress_job"] = name
            _update_progress_ui(
                accepted=_count_accepted_designs(design_path),
                target=n_final,
                trajectories=trajectories,
                stage="aborted",
                running=False,
                failed=True,
                job_name=name,
            )
            with run_output:
                print(f"\n[{name}] aborted by user (tmux kill-session -t {session}).")
        except Exception as e:
            abort_status.value = f"<span style='{ERR}'>Abort failed: {e}</span>"
        finally:
            with JOBS_LOCK:
                RUNNING_JOBS.pop(name, None)
            refresh_jobs_dropdown()

    run_btn.on_click(on_run)
    abort_btn.on_click(on_abort)
    refresh_jobs_btn.on_click(refresh_jobs_dropdown)
    abort_dropdown.observe(on_abort_job_selected, names="value")
    refresh_jobs_dropdown()
    threading.Thread(target=progress_poll_loop, daemon=True).start()

    ui = widgets.VBox(
        [
            welcome,
            upload_intro,
            pdb_banner,
            pdb_upload,
            pdb_status,
            json_banner,
            json_upload,
            json_status,
            upload_note,
            editor_banner,
            file_name_w,
            design_path_w,
            binder_name_w,
            pdb_picker_row,
            chains_w,
            hotspots_w,
            lengths_w,
            n_designs_w,
            save_json_btn,
            save_json_status,
            job_banner,
            job_help,
            job_name_w,
            job_status,
            widgets.HTML("<p>Select a target JSON file (refresh after upload/create):</p>"),
            target_dropdown,
            target_sel_status,
            refresh_target_btn,
            options_banner,
            gpu_dropdown,
            widgets.HBox([refresh_gpu_btn]),
            gpu_status,
            no_pyrosetta_w,
            rank_by_w,
            widgets.HBox([verbose_w, no_plots_w, no_anims_w]),
            select_help,
            filters_dropdown,
            advanced_dropdown,
            generate_btn,
            script_preview,
            run_help,
            run_btn,
            run_output,
            abort_help,
            abort_dropdown,
            progress_bar,
            progress_status,
            abort_btn,
            refresh_jobs_btn,
            abort_status,
        ],
        layout=widgets.Layout(width="100%", padding="8px"),
    )
    display(ui)


if __name__ == "__main__":
    print("Open FreeBindCraft_UI.ipynb in JupyterLab and run launch_all_ui().")
