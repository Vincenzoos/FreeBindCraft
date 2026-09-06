# BindCraft UI GPU Monitoring Implementation Plan

## Objective

Reproduce the following GPU features in `notebooks/BindCraft_UI.ipynb`:

- GPU selection with current free and total memory.
- Optional GPU memory tracking with `gpu_memory_stats.csv` output per BindCraft job.
- `nvidia-smi` details and a list of active BindCraft jobs using GPU memory.

## Design

Keep `notebooks/BindCraft_UI.ipynb` as a thin launcher. The notebook should install the widget dependencies, import `launch_all_ui()` from `main_UI.py`, and call it. Widget construction, subprocess calls, job launching, and monitoring should remain in the Python modules.

## 1. GPU selection and memory display

Update `main_UI.py` around `_list_gpu_options()`:

1. Run `nvidia-smi` with a CSV query for GPU index, name, total memory, used memory, and free memory.
2. Convert each GPU into a dropdown option such as:

   `GPU 0: NVIDIA GeForce RTX 4080 SUPER — 16078 MiB free / 16376 MiB`

3. Include an `Auto (default visible GPUs)` option for machines where GPU discovery is unavailable.
4. Select the GPU with the most free memory by default when discovery succeeds.
5. Add a refresh button that re-queries GPU information and preserves the current selection when possible.
6. When starting a job, export the selected GPU index through `CUDA_VISIBLE_DEVICES` in the generated run script before invoking `bindcraft.py`.

The memory values are a point-in-time snapshot and should be refreshed explicitly through the refresh button or when the selection changes.

## 2. Optional GPU memory tracking

### Backend tracker

Add GPU memory helpers to `functions/generic_utils.py`:

1. Define the `gpu_memory_stats.csv` columns for:
   - design and pipeline stage;
   - actual memory before, peak, and after the stage;
   - reserved memory before, peak, and after the stage;
   - GPU capacity and peak percentages;
   - status and error information.
2. Use JAX memory statistics when available.
3. Query `nvidia-smi` for process memory and GPU capacity as a fallback or complementary measurement.
4. Implement a `GpuMemoryTracker` context manager that:
   - samples memory at a short interval, such as 0.2 seconds;
   - records before and after samples;
   - calculates peak values;
   - writes one CSV row when the context exits;
   - records exceptions as failed stages without suppressing them.
5. Create the CSV header only when tracking is enabled and avoid overwriting an existing file.

### BindCraft integration

Update `bindcraft.py`:

1. Add a `--gpu-memory-tracking` boolean CLI flag.
2. Set the CSV path to `<design_path>/gpu_memory_stats.csv` only when the flag is enabled.
3. Wrap the GPU-heavy pipeline stages with `GpuMemoryTracker`:
   - `trajectory_backbone_generation`;
   - `mpnn_generation`;
   - `complex_prediction`;
   - `binder_only_prediction`.
4. Mark complex predictions that fail filters with a suitable status such as `filter_failed`.
5. Leave normal BindCraft behavior unchanged when the flag is omitted.

### UI integration

Update `main_UI.py`:

1. Add a default-off checkbox labeled `Track GPU memory usage`.
2. Add help text explaining that enabling it creates `gpu_memory_stats.csv` in the design folder.
3. Append `--gpu-memory-tracking` to the generated BindCraft command only when checked.
4. Show the tracking state in the generated command preview.

## 3. `nvidia-smi` details panel

Add an `_nvidia_smi_output()` helper in `main_UI.py`:

1. Run plain `nvidia-smi`.
2. Combine standard output and standard error.
3. Display the complete report in a read-only `Textarea`.
4. Provide friendly fallback messages when `nvidia-smi` is missing or fails.
5. Explain the meaning of memory usage, GPU utilization, power, performance state, and the Processes section.

## 4. Active BindCraft jobs panel

Add a `_bindcraft_gpu_processes_output()` helper in `main_UI.py`:

1. Query all visible GPU indices with `nvidia-smi`.
2. Query compute applications for each GPU using PID and used GPU memory.
3. Resolve process details with `ps`, including user, process name, and command line.
4. Filter results to processes running `bindcraft.py`.
5. Resolve the process working directory through `/proc/<pid>/cwd`.
6. Format rows containing:
   - GPU;
   - PID;
   - user;
   - process;
   - working directory;
   - GPU memory.
7. Display a clear empty-state message when no BindCraft process is using GPU memory.
8. Display a clear unavailable-state message when `nvidia-smi`, `ps`, or process metadata cannot be read.

Refresh both monitoring panels when the GPU selection changes and when the GPU refresh button is pressed.

## 5. Notebook integration

Keep `notebooks/BindCraft_UI.ipynb` structured as follows:

1. Markdown introduction.
2. Dependency installation for `ipywidgets`, `jupyterlab_widgets`, and `widgetsnbextension`.
3. Repository path setup followed by:

   ```python
   from main_UI import launch_all_ui
   launch_all_ui()
   ```

4. Optional output-archiving instructions.

Do not duplicate the UI implementation inside notebook cells.

## 6. Verification plan

### Automated checks

- Run `python -m py_compile main_UI.py bindcraft.py functions/generic_utils.py`.
- Validate that the notebook is valid nbformat JSON.
- Unit-test GPU CSV parsing with normal, empty, malformed, and failed `nvidia-smi` output.
- Verify command generation with tracking disabled and enabled.
- Verify that the selected GPU is exported into the generated run script.
- Verify that tracking-disabled jobs do not create `gpu_memory_stats.csv`.
- Verify that tracking-enabled jobs create the header and append stage rows.
- Verify that tracker exceptions are recorded with a failure status.

### Manual Jupyter test

1. Open `notebooks/BindCraft_UI.ipynb` in JupyterLab.
2. Confirm the GPU dropdown displays name, free memory, and total memory.
3. Confirm the checkbox is off by default.
4. Select a GPU and generate a command; verify the GPU environment value.
5. Enable tracking and run a short BindCraft job.
6. Confirm `gpu_memory_stats.csv` appears in the selected design folder.
7. Refresh the GPU panel while the job is running.
8. Confirm the `nvidia-smi` report and active BindCraft process table update.
9. Confirm the UI handles an empty process list and unavailable GPU tooling gracefully.

## Relevant implementation touchpoints

- `notebooks/BindCraft_UI.ipynb` — launcher notebook.
- `main_UI.py:187` — GPU discovery and memory labels.
- `main_UI.py:838` — GPU dropdown and tracking checkbox.
- `main_UI.py:924` — command construction.
- `main_UI.py:1325` — CUDA environment export and job script generation.
- `main_UI.py:1532` — monitoring widgets in the rendered UI.
- `bindcraft.py:20` — tracking CLI flag.
- `bindcraft.py:52` — conditional CSV path.
- `bindcraft.py:114` — tracked pipeline stages.
- `functions/generic_utils.py:19` — CSV schema and memory tracker.

