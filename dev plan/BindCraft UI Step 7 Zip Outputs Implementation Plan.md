# Step 7 Zip Outputs Implementation Plan

## Objective

Expose a **Step 7: Zip outputs for download** section in the BindCraft UI launched from `notebooks/BindCraft_UI.ipynb`, and list that step in the welcome header banner.

## Design

Keep `notebooks/BindCraft_UI.ipynb` as a thin launcher (deps → `from main_UI import launch_all_ui` → `launch_all_ui()`). All widgets and zip logic live in `main_UI.py`.

Optional notebook-only tweak: mention zip/download in the Cell 0 markdown intro so the notebook description matches the banner.

## 1. Helpers in `main_UI.py`

Near other path helpers (`OUTPUTS_DIR`, `BINDCRAFT_ROOT`):

1. `_list_output_zip_options() -> List[Tuple[str, str]]`
   - Always include `("Entire outputs/ folder", "__all__")`.
   - Append one option per non-hidden subdirectory of `outputs/` (label = folder name, value = folder name).
2. `_zip_outputs(selection: str, zip_name: str = "outputs.zip") -> Path`
   - Normalize name: strip, ensure `.zip` suffix, write only the basename under `BINDCRAFT_ROOT` (no path traversal).
   - `__all__` → zip `OUTPUTS_DIR` with archive root `outputs/`; else zip `OUTPUTS_DIR / selection` with that folder as archive root.
   - Use `zipfile.ZIP_DEFLATED`; skip directories; add files via `rglob`.
   - Return the destination path.

Ensure `import zipfile` is present.

## 2. Update welcome banner steps

In `launch_all_ui()`, edit the `welcome` HTML list so it ends with:

- **Steps 3–6:** select settings / generate script / run BindCraft (existing text).
- **Step 7:** Zip outputs for download.

Do not invent new Step 1–2 wording; keep the existing Step 1 and Step 2 lines.

## 3. Build Step 7 UI section

After the abort / progress widgets (before assembling the root `VBox`), add:

| Widget | Role |
|--------|------|
| `zip_banner` | `_banner("Step 7: Zip outputs for download")` |
| `zip_help` | Short HTML: choose all of `outputs/` or one design folder; archive lands in project root |
| `zip_dropdown` | Options from `_list_output_zip_options()` |
| `zip_name_w` | Text; default `outputs.zip` |
| `refresh_zip_btn` | Re-scan `outputs/` folders; preserve selection if still valid |
| `zip_btn` | Create zip |
| `zip_status` | Success path + size MB, or error |

Handlers:

1. **Refresh** — rebuild options; if current value missing, fall back to `__all__`.
2. **Source change** — `__all__` → `outputs.zip`; else → `{folder}.zip`.
3. **Create zip** — call `_zip_outputs(zip_dropdown.value, zip_name_w.value)`; show `Created {path} ({MB} MB)` or failure.

## 4. Wire into root layout

Append to the main `VBox` children (after abort/progress block):

`zip_banner`, `zip_help`, `zip_dropdown`, `zip_name_w`, `HBox([refresh_zip_btn])`, `zip_btn`, `zip_status`.

## 5. Reproduce / verify

1. Open `notebooks/BindCraft_UI.ipynb`, run all cells.
2. Confirm banner lists Step 7.
3. Scroll to Step 7; **Refresh folders** after a design exists under `outputs/`.
4. Zip one design folder → `{name}.zip` in project root; zip entire outputs → `outputs.zip`.
5. Confirm status shows path and size; open the zip and check archive roots.
6. Re-run UI with empty `outputs/` — dropdown still has “Entire outputs/ folder” and zip creates an empty-or-minimal archive without crashing.

## Out of scope

- Changing BindCraft’s internal `zip_animations` / `zip_plots` advanced settings.
- Serving the zip via HTTP download widget (file is written to disk for the user to fetch from the project root).
