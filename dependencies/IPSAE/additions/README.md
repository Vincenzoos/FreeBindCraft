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

- `ipsae_eval_ui.py`: ipywidgets UI, editable folder discovery, batch execution,
  single-model evaluation, and comparison summaries.
- `README.md`: this note.

The UI calls the original upstream `../ipsae.py` script; it does not replace or
modify the DunbrackLab implementation.
