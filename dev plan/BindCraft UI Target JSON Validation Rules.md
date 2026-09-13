# Target JSON Editor — Validation Rules

## Objective

Document reproducible validation rules, input widget types, and invalid-highlight behaviour for the **target JSON editor** in the BindCraft UI (`main_UI.py`, launched from `notebooks/BindCraft_UI.ipynb`).

This matches the live tips list under “Or you can edit/create a new target JSON file.” and the validators used before **Save Changes**.

## Design

- Keep `notebooks/BindCraft_UI.ipynb` as a thin launcher.
- All field widgets, validators, CSS highlight, and Save enable/disable logic live in `main_UI.py`.
- Constants and helpers (near `_parse_lengths`):

| Constant / helper | Role |
|-------------------|------|
| `_NAME_MAX_LEN` (150) | Max length for File Name stem, `binder_name`, and `design_path` folder name |
| `_DESIGN_PATH_MAX_LEN` (512) | Max length for full `design_path` string |
| `_CHAINS_MAX_LEN` (64) | Max length for `chains` |
| `_HOTSPOTS_MAX_LEN` (500) | Max length for `hotspots` |
| `_FILE_STEM_RE` | File Name stem: `^[A-Za-z][A-Za-z0-9_-]*$` |
| `_SAFE_NAME_RE` | binder / folder: `^[A-Za-z0-9][A-Za-z0-9_-]*$` |
| `_CHAINS_RE` | `^[A-Za-z0-9](,[A-Za-z0-9])*$` |
| `_HOTSPOT_TOKEN_RE` | ColabDesign `prep_pos` token shapes |
| `_FIELD_INVALID_CLASS` (`bc-field-invalid`) | CSS class toggled on invalid widgets |
| `_set_field_invalid(widget, invalid)` | `add_class` / `remove_class`; clears wrapper `layout.border` |
| `_refresh_target_form_validity()` | Re-validate all fields; highlight; set `save_json_btn.disabled` |

On save, `chains` and `hotspots` are normalized with `_normalize_csv_tokens` (`A, C` → `A,C`) because ColabDesign does not strip spaces around commas.

## Fields, input types, and validation rules

| UI label | JSON key | Widget type | Required | Validation rules |
|----------|----------|-------------|----------|------------------|
| **File Name** | *(UI only — output basename under `settings_target/`)* | `widgets.Text` | Yes | Basename only (no `/` `\`); optional `.json` suffix (added on save if missing); stem must start with a **letter** (not digit, `-`, or `_`); only letters/digits/`_`/`-`; max **150** chars (stem) |
| **design_path** | `design_path` | `widgets.Text` | Yes | Writable output directory path; max **512** chars; no `*?<>\|"` or null; folder name (job name) must match binder-style safe name, max **150** |
| **binder_name** | `binder_name` | `widgets.Text` | Yes | Prefix for output PDBs (`{binder_name}_l…_s….pdb`); start with letter or digit; only letters/digits/`_`/`-`; no spaces/path separators; max **150** |
| **starting_pdb** | `starting_pdb` | `widgets.Dropdown` | Yes | Must select an existing `.pdb` / `.cif` from `inputs/` (absolute path written into JSON) |
| **chains** | `chains` | `widgets.Text` | Yes | Comma-separated PDB chain IDs, e.g. `A` or `A,C`; one alphanumeric ID per entry; max **64** chars |
| **hotspots** | `target_hotspot_residues` | `widgets.Text` | No | Empty = no preference (AF2 picks site). Else comma-separated tokens: `A56`, `A60-65`, `1,2-10`, whole chain `A`; range min ≤ max; max **500** chars |
| **lengths** | `lengths` | `widgets.Text` | Yes | Two positive integers (binder size min/max), e.g. `[65, 150]` or `65,150`; order may be either way (BindCraft uses `min`/`max`) |
| **num designs** | `number_of_final_designs` | `widgets.BoundedIntText` (`min=1`, `max=100`, `step=1`) | Yes | Integer **1–100** (Accepted designs to reach); values loaded from JSON are clamped into range |

### Placeholders (empty text helpers)

| Field | Placeholder |
|-------|-------------|
| File Name | `e.g. PDL1.json` |
| design_path | `e.g. /path/to/outputs/PDL1_bindcraft` |
| binder_name | `e.g. PDL1-Binder` |
| chains | `e.g. A or A,C` |
| hotspots | `e.g. A56,A60-65 (empty = no preference)` |
| lengths | `e.g. [65, 150] (min, max binder size)` |
| num designs | *(spinner; no placeholder — always has a numeric value)* |

## Invalid highlight behaviour

1. **Target the text box, not the label wrapper**  
   Do **not** set `widget.layout.border` on the whole ipywidgets control (that draws a box around label + input).  
   Instead inject `_FIELD_VALIDATION_CSS` and toggle class `bc-field-invalid` so only `input` / `select` get:

   - `border: 2px solid #cf222e`
   - `box-shadow: 0 0 0 1px #cf222e`

2. **Per-widget `Layout` instances**  
   Each field uses `_field_layout()` (new `Layout` every time). Sharing one `Layout` across fields causes border/class side effects to clobber each other.

3. **Live updates**  
   Observe `value` on all editor fields (including `starting_pdb`). On change, `_refresh_target_form_validity()`:

   - runs validators with `require=True` for required fields;
   - calls `_set_field_invalid(w, err is not None)` per field;
   - marks `starting_pdb` invalid if the selected file is missing;
   - sets `save_json_btn.disabled = True` when any error remains.

4. **Save Changes**  
   - Starts disabled until the form is fully valid.  
   - Remains disabled while any field fails validation.  
   - On click, re-validates; refuses to write JSON if still invalid.

5. **UI tip copy**  
   The editor tips HTML lists the same rules (driven by the length constants) and states: invalid text boxes are highlighted in red; Save stays disabled until valid.

## Reproduce / verify

1. Open `notebooks/BindCraft_UI.ipynb`, run all cells (or re-run the `launch_all_ui()` cell after code changes).
2. Confirm the tips list shows constraints for each field.
3. **Highlight on input only:** set File Name to `1bad.json` or `asd(((`. Expect a red border on the text box only (not around the “File Name:” label). Save disabled.
4. **File Name letter start:** `_foo.json`, `-foo.json`, `9foo.json` → invalid; `IFIT5_cropped.json` → valid.
5. **Max lengths:** paste >150 chars into File Name stem or binder_name → invalid; >512 into design_path → invalid; long chains/hotspots past 64/500 → invalid.
6. **chains / hotspots format:** `A,C` OK; `A,,B` invalid; `A56,A60-65` OK; `A60-50` (range inverted) invalid; empty hotspots OK.
7. **lengths:** `[65, 150]` and `150,65` OK; `[0, 10]` invalid.
8. **num designs:** spinner only allows 1–100; Save enabled when the rest of the form is valid.
9. Fix all fields → red highlights clear → **Save Changes** enabled → JSON written under `settings_target/` with normalized `chains` / `hotspots`.

## Out of scope

- Validating that hotspot residues / chain IDs exist inside the selected PDB (runtime / ColabDesign concern).
- Editing `settings_filters` / `settings_advanced` JSON in this form.
- Changing BindCraft core `bindcraft.py` argument parsing.
