# FreeBindCraft Additions for IPSAE

This folder contains FreeBindCraft-specific code added on top of the vendored
DunbrackLab/IPSAE repository.

Original upstream IPSAE files remain in `dependencies/IPSAE/`:

- `ipsae.py`
- `af2rechain.py`
- `README.md`
- `LICENSE`
- `Example/`

FreeBindCraft-added files live here in `dependencies/IPSAE/additions/`:

- `ipsae_eval_ui.py`: ipywidgets UI wiring for single-model and bulk evaluation.
- `ipsae_comparison_ui.py`: ipywidgets UI for summarizing collected outputs into a comparison CSV.
- `single_model_eval.py`: single-model execution, output collection, and score parsing.
- `bulk_eval.py`: AlphaFold Server folder discovery, batch execution, and run logging.
- `ipsae_evals_summary.py`: collected-output discovery and best-ipSAE summary CSV generation.
- `README.md`: this note.

The UI calls the original upstream `../ipsae.py` script; it does not replace or
modify the DunbrackLab implementation.

## Bulk evaluation input format

The notebook Bulk Evaluation tab currently supports AlphaFold Server-style AF3
exports only. Boltz folder discovery is not supported yet; use Single Model for
Boltz outputs.

Point the tab at a parent folder containing one subfolder per complex, for
example:

```text
AF3_outputs/
  fold_binder_001/
    fold_binder_001_model_0.cif
    fold_binder_001_full_data_0.json
    fold_binder_001_summary_confidences_0.json
  fold_binder_002/
    fold_binder_002_model_0.cif
    fold_binder_002_full_data_0.json
```

Click `Find models` to preview discovered models before running. By default the
UI fetches each matching `*_model_0.cif` file and pairs it with the corresponding
`*_full_data_0.json` file in the same complex folder. AlphaFold Server ranks
structures from 0 to 4, where model 0 is the highest-confidence prediction.

Each bulk ipSAE run creates a fresh timestamped folder such as
`bulk_ipsae_evals_YYYYMMDD_HHMMSS`. Inside that folder, each model gets its own
subfolder named from the structure filename, containing the copied ipSAE output
files. The root of the timestamped folder also contains a log named from the
folder, such as `bulk_ipsae_evals_YYYYMMDD_HHMMSS_log.json`, which records the
AF3 folder, selected model index, discovered models, and success/failure counts
for that run.

Official AlphaFold Server output reference:
https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/alphafold-server-your-gateway-to-alphafold-3/interpreting-results-from-alphafold-server/

## Summary comparison

Summary comparison is separate from running ipSAE. Use the **Step 5: ipSAE Comparison CSV**
cell in `notebooks/ipsae_eval.ipynb` to choose a folder that already contains collected
ipSAE metric outputs, such as `ipsae_evals/` or a `bulk_ipsae_evals_YYYYMMDD_HHMMSS/`
folder. The summary ranks models by the best ipSAE row per model and can save the comparison
table to CSV. When the CSV field is just a filename, it is saved inside the selected
metrics folder.
