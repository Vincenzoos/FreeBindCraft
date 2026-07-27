"""
ipSAE evaluation UI for single-model and bulk model scoring.

Launch from notebooks/ipsae_eval.ipynb:

    from ipsae_eval_ui import launch_ipsae_eval_ui
    launch_ipsae_eval_ui()

The notebook stays intentionally small; execution and output handling live here.
DunbrackLab/IPSAE itself remains in ../ipsae.py.
"""

from __future__ import annotations

import threading
from pathlib import Path

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display, update_display
except ImportError:  # Allows import in non-notebook contexts for tests/parsing.
    widgets = None
    clear_output = None
    display = print
    update_display = None
    HTML = None

from bulk_eval import discover_af3_models, new_bulk_output_dir, run_bulk_ipsae, write_bulk_run_log
from single_model_eval import (
    DEFAULT_EVALS_DIR,
    IPSAE_SCRIPT,
    IpsaeJob,
    _rel,
    _safe_name,
    model_name_from_structure,
    resolve_repo_path,
    run_ipsae,
)

BANNER = (
    "background:#e8d5f5;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
SOFT = "color:#57606a;"
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"
INFO = "color:#0969da;"
WARNING_CARD = (
    "background:#fff8c5;border:1px solid #d4a72c;color:#5c4400;"
    "padding:12px 16px;border-radius:6px;margin:8px 0;font-family:sans-serif;"
)
EXAMPLE_CARD = (
    "background:#f6f8fa;border:1px solid #d0d7de;color:#24292f;"
    "padding:12px 16px;border-radius:6px;margin:8px 0;font-family:monospace;"
)
AF3_SERVER_OUTPUT_URL = "https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/alphafold-server-your-gateway-to-alphafold-3/interpreting-results-from-alphafold-server/"

RESULT_MSG_ID = "ipsae-eval-result-msg"
RESULT_TABLE_ID = "ipsae-eval-result-table"
RESULT_EXTRA_ID = "ipsae-eval-result-extra"


def _banner(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def _html(text: str) -> "widgets.HTML":
    return widgets.HTML(text)


def _warning(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{WARNING_CARD}"><b>Warning:</b> {text}</div>')


def _example(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<pre style="{EXAMPLE_CARD}">{text}</pre>')


def _publish_cell(display_id: str, obj, initialized: set[str]) -> None:
    if display_id in initialized:
        update_display(obj, display_id=display_id)
    else:
        display(obj, display_id=display_id)
        initialized.add(display_id)


def launch_ipsae_eval_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required to launch this UI.")

    # Replace any previously rendered copy of this UI when the notebook cell is rerun.
    clear_output(wait=True)

    status = widgets.HTML(value=f'<span style="{SOFT}">Ready. Choose single or bulk evaluation, then run ipSAE.</span>')
    state = {"running": False, "bulk_jobs": [], "bulk_source": None}
    cell_initialized: set[str] = set()

    pae_cutoff = widgets.FloatText(value=10.0, description="PAE cutoff", layout=widgets.Layout(width="220px"))
    dist_cutoff = widgets.FloatText(value=10.0, description="Dist cutoff", layout=widgets.Layout(width="220px"))
    collect_outputs = widgets.Checkbox(value=False, description="Copy outputs to separate folder")
    output_dir = widgets.Text(
        value="ipsae_evals",
        description="Output dir",
        disabled=True,
        layout=widgets.Layout(width="700px"),
    )

    label = widgets.Text(value="", description="Label", placeholder="defaults to structure filename", layout=widgets.Layout(width="520px"))
    structure_path = widgets.Text(
        value="",
        description="Structure",
        placeholder="AF2 .pdb | AF3 .cif | Boltz .pdb or .cif",
        layout=widgets.Layout(width="940px"),
    )
    pae_path = widgets.Text(
        value="",
        description="PAE",
        placeholder="AF2/AF3: .json full-data/confidence file | Boltz: .npz PAE file",
        layout=widgets.Layout(width="940px"),
    )
    run_single = widgets.Button(description="Run ipSAE", button_style="primary", icon="play", disabled=True)
    discover_bulk = widgets.Button(description="Find models", button_style="info", icon="search", disabled=True)
    run_bulk = widgets.Button(description="Run ipSAE", button_style="primary", icon="play", disabled=True)
    bulk_folder = widgets.Text(
        value="",
        description="AF3 folder",
        placeholder="Cameron IFIT5 binder AlphaFold3 models",
        layout=widgets.Layout(width="940px"),
    )
    bulk_model_index = widgets.IntText(value=0, description="Model", layout=widgets.Layout(width="180px"))

    settings_panel = widgets.VBox([widgets.HBox([pae_cutoff, dist_cutoff])])

    single_panel = widgets.VBox(
        [
            _html(
                f'<span style="{SOFT}">Provide one matching PAE/confidence file and one structure file. '
                "This evaluates AF2, AF3, or Boltz predictions with saved PAE/confidence files only.</span>"
            ),
            label,
            structure_path,
            pae_path,
            collect_outputs,
            output_dir,
            run_single,
        ]
    )

    bulk_panel = widgets.VBox(
        [
            _html(
                f'<span style="{SOFT}">Enter an AlphaFold Server output folder. The UI will discover matching '
                "model_N CIF files and full_data_N JSON files across all complex subfolders before running.</span>"
            ),
            _warning("Boltz folder structure is not supported in bulk mode yet. Use Single Model for Boltz, or provide an AlphaFold Server-style export folder here."),
            _example(
                "Expected AlphaFold Server export layout\n"
                "AF3_outputs/\n"
                "  fold_binder_001/\n"
                "    fold_binder_001_model_0.cif\n"
                "    fold_binder_001_full_data_0.json\n"
                "    fold_binder_001_summary_confidences_0.json\n"
                "  fold_binder_002/\n"
                "    fold_binder_002_model_0.cif\n"
                "    fold_binder_002_full_data_0.json"
            ),
            _html(
                f'<span style="{SOFT}">AlphaFold Server ranks structures from 0 to 4, with model 0 as the highest-confidence prediction. '
                f'<a href="{AF3_SERVER_OUTPUT_URL}" target="_blank">Official AlphaFold Server output reference</a>.</span>'
            ),
            bulk_folder,
            bulk_model_index,
            discover_bulk,
            run_bulk,
        ]
    )

    tabs = widgets.Tab(children=[single_panel, bulk_panel])
    tabs.set_title(0, "Single Model")
    tabs.set_title(1, "Bulk Evaluation")

    def sync_output_dir_state(_change=None) -> None:
        output_dir.disabled = not collect_outputs.value

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def show_cell_result(
        message: str,
        style: str = SOFT,
        table=None,
        extra_title: str | None = None,
        extra_table=None,
    ) -> None:
        _publish_cell(
            RESULT_MSG_ID,
            HTML(f'<div style="{style};margin:8px 0;font-family:sans-serif">{message}</div>'),
            cell_initialized,
        )
        if table is not None:
            _publish_cell(RESULT_TABLE_ID, table, cell_initialized)
        if extra_title is not None and extra_table is not None:
            _publish_cell(RESULT_EXTRA_ID, HTML(f"<b>{extra_title}</b>"), cell_initialized)
            _publish_cell(f"{RESULT_EXTRA_ID}-table", extra_table, cell_initialized)

    def current_job() -> IpsaeJob:
        structure = structure_path.value.strip()
        pae = pae_path.value.strip()
        if not structure:
            raise ValueError("Structure path is empty.")
        if not pae:
            raise ValueError("PAE path is empty.")
        return IpsaeJob(
            label=label.value.strip() or model_name_from_structure(structure),
            pae_file=Path(pae),
            structure_file=Path(structure),
            pae_cutoff=float(pae_cutoff.value),
            dist_cutoff=float(dist_cutoff.value),
        )

    def bulk_input_key() -> tuple[str, int]:
        return (bulk_folder.value.strip(), int(bulk_model_index.value))

    def sync_single_controls(disabled: bool = False) -> None:
        has_required_paths = bool(structure_path.value.strip()) and bool(pae_path.value.strip())
        run_single.disabled = disabled or not has_required_paths

    def sync_bulk_controls(disabled: bool = False) -> None:
        has_folder = bool(bulk_folder.value.strip())
        discover_bulk.disabled = disabled or not has_folder
        run_bulk.disabled = disabled or not state["bulk_jobs"] or state["bulk_source"] != bulk_input_key()

    def set_buttons_disabled(disabled: bool) -> None:
        sync_single_controls(disabled)
        sync_bulk_controls(disabled)

    def on_bulk_input_changed(_change=None) -> None:
        state["bulk_jobs"] = []
        state["bulk_source"] = None
        sync_bulk_controls(state["running"])

    def on_run_single(_button) -> None:
        if state["running"]:
            return
        try:
            job = current_job()
        except Exception as exc:
            set_status(f"FAILED: {exc}", ERR)
            show_cell_result(f"FAILED: {exc}", ERR)
            return

        collect = collect_outputs.value
        eval_output_dir = output_dir.value.strip() or DEFAULT_EVALS_DIR
        state["running"] = True
        set_buttons_disabled(True)
        set_status("Running ipSAE...", INFO)
        show_cell_result("Running ipSAE...", INFO)

        try:
            scores = run_ipsae(
                job,
                overwrite=True,
                collect_outputs=collect,
                output_dir=eval_output_dir,
            )
            structure = resolve_repo_path(job.structure_file)
            msg = f"SUCCESS: ipSAE completed. Outputs saved to {_rel(structure.parent)}"
            if collect:
                copied_dir = resolve_repo_path(eval_output_dir) / _safe_name(job.label)
                msg += f"<br/>Copied outputs to {_rel(copied_dir)}"
            set_status(msg, OK)
            show_cell_result(msg, OK, table=scores)
        except Exception as exc:
            set_status(f"FAILED: {exc}", ERR)
            show_cell_result(f"FAILED: {exc}", ERR)
        finally:
            state["running"] = False
            set_buttons_disabled(False)

    def on_discover_bulk(_button) -> None:
        if state["running"]:
            return
        try:
            jobs, preview = discover_af3_models(
                bulk_folder.value.strip(),
                model_index=int(bulk_model_index.value),
                pae_cutoff=float(pae_cutoff.value),
                dist_cutoff=float(dist_cutoff.value),
            )
            state["bulk_jobs"] = jobs
            state["bulk_source"] = bulk_input_key()
            sync_bulk_controls(False)
            ready_count = int(preview["Ready"].sum()) if "Ready" in preview else len(jobs)
            set_status(
                f"Found {ready_count}/{len(preview)} ready AlphaFold model(s). Review the table, then run ipSAE.",
                OK if ready_count == len(preview) else INFO,
            )
            show_cell_result(
                f"Found {ready_count}/{len(preview)} ready AlphaFold model(s). Review the table below, then run ipSAE.",
                OK if ready_count == len(preview) else INFO,
                table=preview,
            )
        except Exception as exc:
            state["bulk_jobs"] = []
            state["bulk_source"] = None
            sync_bulk_controls(False)
            set_status(f"FAILED: {exc}", ERR)

    def on_run_bulk(_button) -> None:
        if state["running"]:
            return
        if not state["bulk_jobs"] or state["bulk_source"] != bulk_input_key():
            set_status("No models found for the current folder/model selection. Click Find models first.", ERR)
            sync_bulk_controls(False)
            return

        jobs = list(state["bulk_jobs"])
        af3_folder = bulk_folder.value.strip()
        model_index = int(bulk_model_index.value)
        state["running"] = True
        set_buttons_disabled(True)
        set_status("Running bulk ipSAE...", INFO)
        show_cell_result("Running bulk ipSAE...", INFO)

        def work() -> None:
            try:
                bulk_out_dir = new_bulk_output_dir()
                scores, errors = run_bulk_ipsae(
                    jobs,
                    overwrite=True,
                    collect_outputs=True,
                    output_dir=bulk_out_dir,
                )
                ok_count = len(jobs) - len(errors)
                log_path = write_bulk_run_log(
                    bulk_out_dir,
                    af3_folder,
                    model_index,
                    jobs,
                    ok_count,
                    len(errors),
                )
                msg = f"SUCCESS: bulk ipSAE finished for {ok_count}/{len(jobs)} model(s)."
                msg += f"<br/>Logged AF3 folder/model selection to {_rel(log_path)}"
                if ok_count:
                    msg += f"<br/>Copied ipSAE outputs to {_rel(bulk_out_dir)}"
                    job = jobs[0]
                    pae_tag = str(int(job.pae_cutoff))
                    dist_tag = str(int(job.dist_cutoff))
                    suffix = f"{pae_tag}_{dist_tag}"
                    msg += (
                        f'<br/><span style="{SOFT}">Each model gets 3 score files from one run: '
                        f"<code>*_{suffix}.txt</code> (summary), "
                        f"<code>*_{suffix}_byres.txt</code> (per-residue), "
                        f"<code>*_{suffix}.pml</code> (PyMOL). "
                        f"Table contains score files per model for all models in the folder, not a separate model in each row.</span>"
                    )
                if not errors.empty:
                    msg += f"<br/>{len(errors)} model(s) failed; see errors below."
                style = OK if errors.empty else INFO
                show_cell_result(
                    msg,
                    style,
                    table=scores if not scores.empty else None,
                    extra_title="Errors" if not errors.empty else None,
                    extra_table=errors if not errors.empty else None,
                )
            except Exception as exc:
                show_cell_result(f"FAILED: {exc}", ERR)
            finally:
                state["running"] = False
                set_buttons_disabled(False)
                set_status(
                    "Bulk ipSAE finished. See results below the UI.",
                    OK,
                )

        threading.Thread(target=work, daemon=True).start()

    collect_outputs.observe(sync_output_dir_state, names="value")
    structure_path.observe(lambda change: sync_single_controls(state["running"]), names="value")
    pae_path.observe(lambda change: sync_single_controls(state["running"]), names="value")
    bulk_folder.observe(on_bulk_input_changed, names="value")
    bulk_model_index.observe(on_bulk_input_changed, names="value")
    sync_output_dir_state()
    sync_single_controls(False)
    sync_bulk_controls(False)
    run_single.on_click(on_run_single)
    discover_bulk.on_click(on_discover_bulk)
    run_bulk.on_click(on_run_bulk)

    display(
        widgets.VBox(
            [
                _banner("ipSAE Evaluation"),
                _html(
                    f'<span style="{SOFT}">Using original DunbrackLab script: {_rel(IPSAE_SCRIPT)}. '
                    "BindCraft/FreeBindCraft ranked binder PDBs do not include PAE JSON files, so use AF3/Boltz/AF2 confidence outputs here and compare against final_design_stats.csv as needed.</span>"
                ),
                settings_panel,
                tabs,
                status,
            ]
        )
    )
    show_cell_result(
        "Ready. Run ipSAE to show results here in the notebook cell output.",
        SOFT,
    )


if __name__ == "__main__":
    print(f"IPSAE script: {IPSAE_SCRIPT}")
    print("Launch from notebooks/ipsae_eval.ipynb via launch_ipsae_eval_ui().")
