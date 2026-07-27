"""
ipSAE comparison CSV UI for summarizing collected metric outputs.

Launch from notebooks/ipsae_eval.ipynb:

    from ipsae_comparison_ui import launch_ipsae_comparison_ui
    launch_ipsae_comparison_ui()
"""

from __future__ import annotations

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display, update_display
except ImportError:  # Allows import in non-notebook contexts for tests/parsing.
    widgets = None
    clear_output = None
    display = print
    update_display = None
    HTML = None

from ipsae_evals_summary import resolve_summary_csv_path, summarize_ipsae_folder
from single_model_eval import _rel

BANNER = (
    "background:#e8d5f5;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
SOFT = "color:#57606a;"
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"
INFO = "color:#0969da;"

RESULT_MSG_ID = "ipsae-comparison-result-msg"
RESULT_TABLE_ID = "ipsae-comparison-result-table"


def _banner(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def _html(text: str) -> "widgets.HTML":
    return widgets.HTML(text)


def _panel(children) -> "widgets.VBox":
    return widgets.VBox(
        children,
        layout=widgets.Layout(
            width="100%",
            border="1px solid #d0d7de",
            padding="12px 16px",
            margin="0 0 8px 0",
        ),
    )


def _publish_cell(display_id: str, obj, initialized: set[str]) -> None:
    if display_id in initialized:
        update_display(obj, display_id=display_id)
    else:
        display(obj, display_id=display_id)
        initialized.add(display_id)


def launch_ipsae_comparison_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required to launch this UI.")

    clear_output(wait=True)

    state = {"running": False}
    cell_initialized: set[str] = set()

    summary_folder = widgets.Text(
        value="ipsae_evals",
        description="Metrics folder",
        placeholder="ipsae_evals or bulk_ipsae_evals_YYYYMMDD_HHMMSS",
        layout=widgets.Layout(width="940px"),
    )
    summary_csv = widgets.Text(
        value="ipsae_evals_summary.csv",
        description="Summary CSV",
        layout=widgets.Layout(width="940px"),
    )
    run_summary = widgets.Button(description="Summarize metrics", button_style="success", icon="bar-chart")
    summary_status = widgets.HTML(value=f'<span style="{SOFT}">Ready. Choose a metrics folder, then summarize.</span>')

    def set_summary_status(message: str, style: str = SOFT) -> None:
        summary_status.value = f'<div style="{style}">{message}</div>'

    def show_summary_result(message: str, style: str = SOFT, table=None, section_title: str | None = None) -> None:
        _publish_cell(
            RESULT_MSG_ID,
            HTML(f'<div style="{style};margin:8px 0;font-family:sans-serif">{message}</div>'),
            cell_initialized,
        )
        if section_title is not None:
            _publish_cell(RESULT_TABLE_ID, HTML(f"<b>{section_title}</b>"), cell_initialized)
        if table is not None:
            _publish_cell(f"{RESULT_TABLE_ID}-data", table, cell_initialized)

    def on_run_summary(_button) -> None:
        if state["running"]:
            return
        folder = summary_folder.value.strip() or "ipsae_evals"
        csv_name = summary_csv.value.strip() or "ipsae_evals_summary.csv"
        state["running"] = True
        run_summary.disabled = True
        set_summary_status("Summarizing ipSAE metrics...", INFO)
        show_summary_result("Summarizing ipSAE metrics...", INFO)

        try:
            summary, all_scores = summarize_ipsae_folder(folder)
            out_path = resolve_summary_csv_path(folder, csv_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out_path, index=False)
            msg = (
                f"SUCCESS: summarized {len(all_scores)} score file(s) from "
                f"{summary['Model'].nunique() if 'Model' in summary else len(summary)} model(s). "
                f"Saved to {_rel(out_path)}"
            )
            set_summary_status(msg, OK)
            show_summary_result(msg, OK, table=summary, section_title="models evaluation ranked by ipSAE score")
        except Exception as exc:
            set_summary_status(f"FAILED: {exc}", ERR)
            show_summary_result(f"FAILED: {exc}", ERR)
        finally:
            state["running"] = False
            run_summary.disabled = False

    run_summary.on_click(on_run_summary)

    comparison_panel = _panel(
        [
            _html(
                f'<span style="{SOFT}">Select a folder that already contains collected ipSAE metric outputs, such as '
                "ipsae_evals or a bulk_ipsae_evals timestamp folder. This analysis is separate from running ipSAE.</span>"
            ),
            summary_folder,
            summary_csv,
            run_summary,
        ]
    )

    display(
        widgets.VBox(
            [
                _banner("ipSAE Comparison CSV"),
                comparison_panel,
                summary_status,
            ]
        )
    )
    show_summary_result(
        "Ready. Choose a metrics folder, then summarize to show results here.",
        SOFT,
    )


if __name__ == "__main__":
    print("Launch from notebooks/ipsae_eval.ipynb via launch_ipsae_comparison_ui().")
