# BindCraft/FreeBindCraft UI CPU Monitoring Implementation Plan

## Objective

Add CPU monitoring to the interactive UI with the same workflow as GPU
monitoring:

- an optional CPU memory tracker that writes `cpu_memory_stats.csv` per job;
- an htop-style system snapshot showing load, CPU, RAM, swap, and processes;
- an active-job table showing CPU percentage and resident memory (RSS);
- repository-aware filtering so the table only shows jobs from the current
  repository.

The repository-aware behavior is required for both repository variants:

| Repository opened in Jupyter | Jobs allowed in the active CPU table |
| --- | --- |
| BindCraft | BindCraft jobs from this checkout only |
| FreeBindCraft | FreeBindCraft jobs from this checkout only |

An unrelated Python process, a job from the other repository, and a process
running from another checkout must never appear in the table.

## Reproducibility and scope

Keep `notebooks/BindCraft_UI.ipynb` as a thin launcher. The implementation
belongs in `main_UI.py`, `bindcraft.py`, and `functions/generic_utils.py`.
Do not scrape htop's interactive terminal screen. htop's `RES`, `MEM%`, and
system memory values are based on Linux procfs and ps data; read those sources
directly to produce stable notebook text and background samples.

The implementation assumes the existing Linux process tools used by the GPU
panel are available:

- `/proc` for process and system memory metadata;
- `ps` for process CPU%, MEM%, RSS, elapsed time, and command line;
- `nvidia-smi` only for the existing GPU panel.

The CPU panel should continue to show a friendly unavailable message when
procfs or `ps` cannot be read.

## 1. Detect the current repository identity

Replace BindCraft-specific process naming with one repository identity derived
from the checkout containing `main_UI.py`.

Add a single helper near the path constants, for example:

```python
def _detect_project_identity() -> tuple[Path, Path, str, str]:
    """Return (root, entrypoint, display_name, process_token)."""
```

The helper must:

1. Set `root` to `Path(__file__).resolve().parent`.
2. Prefer the entrypoint that exists in that root:
   - `bindcraft.py` -> display name `BindCraft`, process token `bindcraft`;
   - `freebindcraft.py` -> display name `FreeBindCraft`, process token
     `freebindcraft`.
3. If both entrypoints exist, select the one matching the root directory name
   case-insensitively; if there is still no unique match, fail with a clear
   error instead of silently mixing repositories.
4. Store the result in neutral constants such as:

   ```python
   PROJECT_ROOT
   PROJECT_SCRIPT
   PROJECT_DISPLAY_NAME
   PROJECT_PROCESS_TOKEN
   ```

5. Use these constants for command generation, process labels, process
   filtering, and user-facing empty/error messages. Do not hard-code
   `BINDCRAFT_SCRIPT`, `bindcraft.py`, or `BindCraft` in the generic monitoring
   code.

The existing BindCraft checkout should resolve to:

```text
PROJECT_ROOT=/storage/viet/BindCraft
PROJECT_SCRIPT=/storage/viet/BindCraft/bindcraft.py
PROJECT_DISPLAY_NAME=BindCraft
PROJECT_PROCESS_TOKEN=bindcraft
```

A FreeBindCraft checkout with `freebindcraft.py` should resolve to the
corresponding FreeBindCraft values without another implementation branch.

## 2. Implement one repository-scoped process predicate

Rename the current `_is_this_bindcraft_process()` helper to a neutral name,
such as `_is_this_project_process(pid, cmdline="")`.

The predicate must return true only when all of the following are satisfied:

1. The command line identifies the current entrypoint. If the command contains
   an absolute script path, resolve it and require it to equal
   `PROJECT_SCRIPT`; if it contains only a bare script name, require that name
   to match the selected entrypoint and validate it against the process cwd in
   the next step. Do not accept a generic process named `python`.
2. The process working directory is inside `PROJECT_ROOT`, resolved through
   `/proc/<pid>/cwd`. A bare `bindcraft.py` or `freebindcraft.py` command is
   accepted only when this cwd check succeeds.
3. The process is still readable and has not exited between the `ps` query and
   the `/proc` lookup.

Use `Path.resolve()` and `Path.relative_to()` for the checkout boundary. Do
not use a substring test on the repository name alone: that could match
`/tmp/OtherBindCraft` or a similarly named project.

Example acceptance matrix:

| Command / working directory | Expected |
| --- | --- |
| `python /current/BindCraft/bindcraft.py`, cwd `/current/BindCraft` | include |
| `python /current/FreeBindCraft/freebindcraft.py`, cwd `/current/FreeBindCraft` while UI is BindCraft | exclude |
| `python /other/BindCraft/bindcraft.py`, cwd `/other/BindCraft` | exclude |
| unrelated `python worker.py`, cwd `/current/BindCraft` | exclude |
| shell/tmux wrapper without the selected entrypoint | exclude |

Reuse this predicate from both `_bindcraft_gpu_processes_output()` (renamed
to `_project_gpu_processes_output()`) and
`_bindcraft_cpu_processes_output()` (renamed to
`_project_cpu_processes_output()`). This keeps GPU and CPU tables consistent.

## 3. CPU system snapshot

Add a neutral `_project_system_snapshot_output()` or `_htop_output()` helper
in `main_UI.py`.

### System values

Read:

- `/proc/meminfo`: `MemTotal`, `MemAvailable`, `SwapTotal`, `SwapFree`;
- `os.getloadavg()` for the 1, 5, and 15 minute load averages;
- `os.cpu_count()` for the visible CPU count;
- `ps -eo pid=,user=,stat=,pcpu=,pmem=,rss=,etime=,time=,args= --sort=-pcpu`
  for the process list.

Format the output like an htop snapshot:

```text
htop-style CPU and memory snapshot (refresh to update)
CPU cores: ...    Load average: ...
Mem: ... used / ... total
Swp: ... used / ... total
Tasks: ...

    PID USER             STAT   CPU%   MEM%        RES      TIME+ COMMAND
```

Use the same units as htop where practical (`K`, `M`, `G`). Limit the display
to a deterministic number of highest-CPU rows, such as 30, and truncate long
commands so the read-only notebook textarea remains usable.

The system snapshot is intentionally machine-wide, like `nvidia-smi`. Only
the separate active-job table is repository-filtered.

## 4. Active CPU/memory jobs panel

Implement `_project_cpu_processes_output()` using:

```text
ps -eo pid=,user=,comm=,pcpu=,pmem=,rss=,etime=,time=,args=
```

For every row:

1. Require `PROJECT_PROCESS_TOKEN` or the selected script basename in the
   command line.
2. Apply `_is_this_project_process()`.
3. Read `/proc/<pid>/cwd` for the working directory.
4. Convert RSS from KiB to the compact display unit.
5. Format columns containing:
   - PID;
   - user;
   - process;
   - CPU%;
   - MEM%;
   - RSS;
   - elapsed time;
   - CWD.

Use dynamic text based on `PROJECT_DISPLAY_NAME`:

```text
Active BindCraft jobs using CPU and memory
```

or, in the other checkout:

```text
Active FreeBindCraft jobs using CPU and memory
```

The help text and empty state must also be dynamic, for example:

```text
No active FreeBindCraft jobs are currently using CPU resources.
```

Do not display all Python processes and do not infer repository ownership from
CPU usage, memory usage, or the process name `python`.

## 5. Per-stage CPU memory tracking

### Backend tracker

Add CPU columns and `create_cpu_memory_csv()` to
`functions/generic_utils.py`. Use the same schema in both repository variants;
the existing `bindcraft_` column prefix is retained for CSV compatibility and
does not control process filtering:

```text
design_name
stage
bindcraft_rss_before_mib
bindcraft_rss_peak_mib
bindcraft_rss_after_mib
system_memory_total_mib
bindcraft_rss_peak_percent
status
error
```

Implement `CpuMemoryTracker` as a context manager that:

1. Samples the current process `VmRSS` from `/proc/<pid>/status` every 0.2
   seconds.
2. Samples `MemTotal` from `/proc/meminfo`.
3. Records before, peak, and after RSS values.
4. Writes one CSV row on normal or exceptional exit.
5. Records exception status without suppressing the original exception.
6. Does not create a CSV file when tracking is disabled.

### Pipeline integration

In the repository’s entrypoint:

1. Add `--cpu-memory-tracking`.
2. Create `<design_path>/cpu_memory_stats.csv` only when enabled.
3. Wrap the same four stages already covered by GPU tracking:
   - trajectory backbone generation;
   - MPNN generation;
   - complex prediction;
   - binder-only prediction.
4. Mark filter-failed complex predictions consistently in both CPU and GPU
   CSVs.

## 6. UI widgets and refresh behavior

In `launch_all_ui()`:

1. Keep the default-off `Track CPU memory usage` checkbox.
2. Append `--cpu-memory-tracking` only when selected.
3. Add a read-only textarea titled `htop details for the current machine`.
4. Add a read-only textarea titled
   `Active <PROJECT_DISPLAY_NAME> jobs using CPU and memory`.
5. Add a `Refresh CPU stats` button.
6. Refresh both CPU panels when:
   - `Refresh CPU stats` is clicked;
   - the GPU refresh button is clicked;
   - the GPU selection changes.
7. Keep all process labels, help text, empty states, and error states dynamic
   from `PROJECT_DISPLAY_NAME`.

## 7. Verification plan

### Automated checks

Run from the repository root:

```bash
python -m py_compile main_UI.py bindcraft.py functions/generic_utils.py
python -m json.tool notebooks/BindCraft_UI.ipynb >/dev/null
git diff --check
```

Add tests or a test harness covering:

1. Repository identity detection for BindCraft and FreeBindCraft layouts.
2. Both entrypoints present, missing entrypoint, and ambiguous layout errors.
3. Positive and negative `_is_this_project_process()` cases from the matrix in
   Section 2.
4. Mocked `ps` output containing:
   - current-repository BindCraft/FreeBindCraft jobs;
   - the other repository’s jobs;
   - another checkout of the same repository;
   - unrelated Python processes.
5. CPU output containing only current-repository jobs.
6. Dynamic headings and empty-state messages for both repository variants.
7. CPU CSV header creation, stage rows, disabled tracking, and exception
   status.
8. Command generation with CPU tracking disabled and enabled.

The key assertion for the active CPU table is:

```python
assert current_repo_job in output
assert other_repo_job not in output
assert other_checkout_job not in output
assert unrelated_python_job not in output
```

### Manual Jupyter checks

Perform the following in both a BindCraft checkout and a FreeBindCraft
checkout:

1. Open `notebooks/BindCraft_UI.ipynb` and run the launcher cell.
2. Confirm the heading says `Active <repo> jobs using CPU and memory`.
3. Confirm the htop-style snapshot shows load, RAM, swap, CPU%, and RES.
4. Start one job from the current checkout and confirm exactly that job
   appears in the active CPU table.
5. Start or identify a job from the other repository and confirm it does not
   appear.
6. Refresh CPU stats while jobs are running and confirm CPU%, MEM%, RSS, and
   elapsed time change.
7. Enable CPU tracking, run a short job, and confirm
   `cpu_memory_stats.csv` is created with one row per tracked stage.
8. Repeat with tracking disabled and confirm no CPU CSV is created by the new
   run.
9. Stop jobs and confirm the table shows the dynamic empty-state message.
10. Temporarily make `ps` unavailable or unreadable and confirm the UI shows a
    friendly unavailable-state message instead of raising from the notebook.

## Relevant implementation touchpoints

- `notebooks/BindCraft_UI.ipynb` — thin launcher only.
- `main_UI.py` — repository identity, process filtering, CPU snapshot, CPU
  process table, widgets, and refresh callbacks.
- `bindcraft.py` or `freebindcraft.py` — CPU tracking CLI flag, CSV path, and
  stage wrappers.
- `functions/generic_utils.py` — CPU CSV schema, procfs sampler, and
  `CpuMemoryTracker`.
- `dev plan/BindCraft UI GPU Monitoring Implementation Plan.md` — existing GPU
  behavior that should reuse the repository identity and process predicate.
