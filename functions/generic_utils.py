####################################
################## General functions
####################################
### Import dependencies
import os
import json
import csv
import re
import shutil
import zipfile
import random
import math
import stat
import subprocess
import threading
import pandas as pd
import numpy as np
import time
from .logging_utils import vprint

try:
    import jax
except ImportError:
    # The tracker can still use nvidia-smi when JAX is not installed.  The
    # normal BindCraft GPU availability check reports the missing backend.
    jax = None


# GPU memory tracking is deliberately opt-in.  Keep the schema in one place so
# the launcher and the tracker always agree on the CSV layout.
GPU_MEMORY_STATS_COLUMNS = [
    'timestamp',
    'design',
    'stage',
    'actual_memory_before_mib',
    'actual_memory_peak_mib',
    'actual_memory_after_mib',
    'reserved_memory_before_mib',
    'reserved_memory_peak_mib',
    'reserved_memory_after_mib',
    'gpu_capacity_mib',
    'actual_peak_percent',
    'reserved_peak_percent',
    'status',
    'error',
]

_GPU_MEMORY_CSV_LOCK = threading.Lock()


def _memory_value_to_mib(value):
    """Convert a byte or MiB value to a finite float, or return ``None``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def _jax_gpu_memory_sample():
    """Return aggregate JAX GPU allocator values in MiB when available."""
    actual = reserved = actual_peak = reserved_peak = None
    if jax is None:
        return {}
    try:
        devices = [device for device in jax.devices() if device.platform == 'gpu']
        if not devices:
            return {}

        actual_values = []
        reserved_values = []
        actual_peak_values = []
        reserved_peak_values = []
        for device in devices:
            stats = device.memory_stats()
            if not stats:
                continue
            # These keys are provided by CUDA/ROCm JAX backends.  A few JAX
            # versions expose only the current allocator values.
            for key, target in (
                ('bytes_in_use', actual_values),
                ('bytes_reserved', reserved_values),
                ('peak_bytes_in_use', actual_peak_values),
                ('peak_bytes_reserved', reserved_peak_values),
            ):
                value = _memory_value_to_mib(stats.get(key))
                if value is not None:
                    target.append(value / (1024 * 1024))

        if actual_values:
            actual = sum(actual_values)
        if reserved_values:
            reserved = sum(reserved_values)
        if actual_peak_values:
            actual_peak = sum(actual_peak_values)
        if reserved_peak_values:
            reserved_peak = sum(reserved_peak_values)
    except Exception:
        # Memory introspection must never interrupt a BindCraft stage.
        return {}

    return {
        'actual': actual,
        'reserved': reserved,
        'actual_peak': actual_peak,
        'reserved_peak': reserved_peak,
    }


def _parse_nvidia_memory(value):
    """Parse a numeric nvidia-smi memory field, tolerating ``MiB`` suffixes."""
    if value is None:
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value))
    if not match:
        return None
    return _memory_value_to_mib(match.group(0))


def _nvidia_smi_memory_sample():
    """Read GPU capacity and this process' GPU allocation from nvidia-smi."""
    capacity = None
    process_memory = None

    try:
        capacity_command = [
            'nvidia-smi',
            '--query-gpu=memory.total',
            '--format=csv,noheader,nounits',
        ]
        selected_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        if selected_gpus and selected_gpus != '-1':
            capacity_command.extend(['-i', selected_gpus])
        capacity_proc = subprocess.run(
            capacity_command,
            capture_output=True,
            text=True,
            check=False,
        )
        if capacity_proc.returncode == 0:
            values = [
                parsed
                for parsed in (_parse_nvidia_memory(line) for line in capacity_proc.stdout.splitlines())
                if parsed is not None
            ]
            if values:
                capacity = sum(values)
    except Exception:
        pass

    try:
        process_proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process_proc.returncode == 0:
            pid = str(os.getpid())
            values = []
            for row in csv.reader(process_proc.stdout.splitlines()):
                if len(row) < 2 or row[0].strip() != pid:
                    continue
                parsed = _parse_nvidia_memory(row[1])
                if parsed is not None:
                    values.append(parsed)
            if values:
                process_memory = sum(values)
            elif process_proc.stdout.strip():
                # A successful query with no matching process is a valid zero
                # reading; an empty query is left as unavailable.
                process_memory = 0.0
    except Exception:
        pass

    return {'capacity': capacity, 'process': process_memory}


def _sample_gpu_memory():
    """Collect a single best-effort GPU memory sample in MiB."""
    jax_sample = _jax_gpu_memory_sample()
    nvidia_sample = _nvidia_smi_memory_sample()

    jax_actual = jax_sample.get('actual')
    jax_reserved = jax_sample.get('reserved')
    process_memory = nvidia_sample.get('process')

    # JAX reports allocator memory while nvidia-smi includes the complete
    # process allocation.  Taking the larger value avoids understating usage
    # when either source reports a narrower view of the allocation.
    actual_values = [value for value in (jax_actual, process_memory) if value is not None]
    reserved_values = [value for value in (jax_reserved, process_memory) if value is not None]
    actual = max(actual_values) if actual_values else None
    reserved = max(reserved_values) if reserved_values else None

    return {
        'actual': actual,
        'reserved': reserved,
        'actual_peak': jax_sample.get('actual_peak'),
        'reserved_peak': jax_sample.get('reserved_peak'),
        'capacity': nvidia_sample.get('capacity'),
    }


def create_gpu_memory_stats_csv(csv_path):
    """Create the GPU tracking header without replacing an existing file."""
    if not csv_path:
        return
    csv_path = os.fspath(csv_path)
    parent = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(parent, exist_ok=True)
    with _GPU_MEMORY_CSV_LOCK:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            return
        with open(csv_path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=GPU_MEMORY_STATS_COLUMNS)
            writer.writeheader()


class GpuMemoryTracker:
    """Sample GPU memory during one BindCraft pipeline stage.

    The context manager is intentionally best effort: unavailable JAX or
    nvidia-smi statistics are recorded as blank cells, and tracking failures do
    not change normal BindCraft execution.  Exceptions raised by the wrapped
    stage are recorded and re-raised by returning ``False`` from ``__exit__``.
    """

    def __init__(
        self,
        csv_path=None,
        design=None,
        stage=None,
        interval=0.2,
        *,
        design_name=None,
        pipeline_stage=None,
    ):
        # Accept descriptive keyword aliases as well as the compact internal
        # names, which keeps the helper convenient for external callers.
        if design is None:
            design = design_name
        if stage is None:
            stage = pipeline_stage
        self.csv_path = csv_path
        self.design = str(design or '')
        self.stage = str(stage or '')
        self.interval = max(0.05, float(interval))
        self.status = 'success'
        self.error = ''
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def set_status(self, status, error=''):
        """Set a non-exception outcome such as ``filter_failed``."""
        self.status = str(status)
        self.error = str(error or '')

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self._samples.append(_sample_gpu_memory())
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    @staticmethod
    def _first_value(samples, key):
        for sample in samples:
            if sample.get(key) is not None:
                return sample[key]
        return None

    @staticmethod
    def _max_value(samples, key):
        values = [sample.get(key) for sample in samples if sample.get(key) is not None]
        return max(values) if values else None

    @staticmethod
    def _last_value(samples, key):
        for sample in reversed(samples):
            if sample.get(key) is not None:
                return sample[key]
        return None

    @staticmethod
    def _round(value):
        return round(value, 3) if value is not None else ''

    def _write_row(self, samples):
        if not self.csv_path:
            return
        if not samples:
            samples = [{}]
        capacity = self._max_value(samples, 'capacity')
        actual_values = [
            value
            for key in ('actual', 'actual_peak')
            for value in (self._max_value(samples, key),)
            if value is not None
        ]
        reserved_values = [
            value
            for key in ('reserved', 'reserved_peak')
            for value in (self._max_value(samples, key),)
            if value is not None
        ]
        actual_peak = max(actual_values) if actual_values else None
        reserved_peak = max(reserved_values) if reserved_values else None
        row = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'design': self.design,
            'stage': self.stage,
            'actual_memory_before_mib': self._round(self._first_value(samples, 'actual')),
            'actual_memory_peak_mib': self._round(actual_peak),
            'actual_memory_after_mib': self._round(self._last_value(samples, 'actual')),
            'reserved_memory_before_mib': self._round(self._first_value(samples, 'reserved')),
            'reserved_memory_peak_mib': self._round(reserved_peak),
            'reserved_memory_after_mib': self._round(self._last_value(samples, 'reserved')),
            'gpu_capacity_mib': self._round(capacity),
            'actual_peak_percent': self._round((actual_peak / capacity * 100) if actual_peak is not None and capacity else None),
            'reserved_peak_percent': self._round((reserved_peak / capacity * 100) if reserved_peak is not None and capacity else None),
            'status': self.status,
            'error': self.error,
        }
        try:
            create_gpu_memory_stats_csv(self.csv_path)
            with _GPU_MEMORY_CSV_LOCK:
                with open(self.csv_path, 'a', newline='') as handle:
                    csv.DictWriter(handle, fieldnames=GPU_MEMORY_STATS_COLUMNS).writerow(row)
        except Exception as exc:
            print(f'Warning: unable to write GPU memory statistics: {exc}')

    def __enter__(self):
        if self.csv_path:
            try:
                self._samples.append(_sample_gpu_memory())
            except Exception:
                self._samples.append({})
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.status = 'failed'
            self.error = f'{exc_type.__name__}: {exc_value}'
        if self.csv_path:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, self.interval * 2))
            try:
                self._samples.append(_sample_gpu_memory())
            except Exception:
                self._samples.append({})
            self._write_row(self._samples)
        return False


# CPU tracking intentionally retains the historical ``bindcraft_`` CSV
# prefix so statistics from the BindCraft and FreeBindCraft variants share a
# compatible schema.  The prefix is a column-format convention only; process
# ownership is handled by the UI's repository-aware predicate.
CPU_MEMORY_STATS_COLUMNS = [
    'design_name',
    'stage',
    'bindcraft_rss_before_mib',
    'bindcraft_rss_peak_mib',
    'bindcraft_rss_after_mib',
    'system_memory_total_mib',
    'bindcraft_rss_peak_percent',
    'status',
    'error',
]

_CPU_MEMORY_CSV_LOCK = threading.Lock()


def _read_proc_rss_mib(pid=None):
    """Read ``VmRSS`` for a process from procfs, returning MiB or ``None``."""
    pid = os.getpid() if pid is None else int(pid)
    try:
        with open(f'/proc/{pid}/status', 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    match = re.search(r'\d+', line)
                    if match:
                        return float(match.group(0)) / 1024.0
    except (OSError, ValueError):
        return None
    return None


def _read_system_memory_total_mib():
    """Read ``MemTotal`` from procfs, returning MiB or ``None``."""
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('MemTotal:'):
                    match = re.search(r'\d+', line)
                    if match:
                        return float(match.group(0)) / 1024.0
    except (OSError, ValueError):
        return None
    return None


def _sample_cpu_memory(pid=None):
    return {
        'rss': _read_proc_rss_mib(pid),
        'system_total': _read_system_memory_total_mib(),
    }


def create_cpu_memory_csv(csv_path):
    """Create the CPU tracking header without replacing an existing file."""
    if not csv_path:
        return
    csv_path = os.fspath(csv_path)
    parent = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(parent, exist_ok=True)
    with _CPU_MEMORY_CSV_LOCK:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            return
        with open(csv_path, 'a', newline='') as handle:
            csv.DictWriter(handle, fieldnames=CPU_MEMORY_STATS_COLUMNS).writeheader()


class CpuMemoryTracker:
    """Sample this process' RSS during one pipeline stage."""

    def __init__(
        self,
        csv_path=None,
        design_name=None,
        stage=None,
        interval=0.2,
        *,
        design=None,
        pipeline_stage=None,
    ):
        if design_name is None:
            design_name = design
        if stage is None:
            stage = pipeline_stage
        self.csv_path = csv_path
        self.design_name = str(design_name or '')
        self.stage = str(stage or '')
        self.interval = max(0.05, float(interval))
        self.pid = os.getpid()
        self.status = 'success'
        self.error = ''
        self._samples = []
        self._stop_event = threading.Event()
        self._thread = None

    def set_status(self, status, error=''):
        """Set a non-exception outcome such as ``filter_failed``."""
        self.status = str(status)
        self.error = str(error or '')

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self._samples.append(_sample_cpu_memory(self.pid))
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    @staticmethod
    def _first_value(samples, key):
        for sample in samples:
            if sample.get(key) is not None:
                return sample[key]
        return None

    @staticmethod
    def _max_value(samples, key):
        values = [sample.get(key) for sample in samples if sample.get(key) is not None]
        return max(values) if values else None

    @staticmethod
    def _last_value(samples, key):
        for sample in reversed(samples):
            if sample.get(key) is not None:
                return sample[key]
        return None

    @staticmethod
    def _round(value):
        return round(value, 3) if value is not None else ''

    def _write_row(self, samples):
        if not self.csv_path:
            return
        if not samples:
            samples = [{}]
        rss_peak = self._max_value(samples, 'rss')
        total = self._first_value(samples, 'system_total')
        row = {
            'design_name': self.design_name,
            'stage': self.stage,
            'bindcraft_rss_before_mib': self._round(self._first_value(samples, 'rss')),
            'bindcraft_rss_peak_mib': self._round(rss_peak),
            'bindcraft_rss_after_mib': self._round(self._last_value(samples, 'rss')),
            'system_memory_total_mib': self._round(total),
            'bindcraft_rss_peak_percent': self._round(
                (rss_peak / total * 100) if rss_peak is not None and total else None
            ),
            'status': self.status,
            'error': self.error,
        }
        try:
            create_cpu_memory_csv(self.csv_path)
            with _CPU_MEMORY_CSV_LOCK:
                with open(self.csv_path, 'a', newline='') as handle:
                    csv.DictWriter(handle, fieldnames=CPU_MEMORY_STATS_COLUMNS).writerow(row)
        except Exception as exc:
            print(f'Warning: unable to write CPU memory statistics: {exc}')

    def __enter__(self):
        if self.csv_path:
            try:
                self._samples.append(_sample_cpu_memory(self.pid))
            except Exception:
                self._samples.append({})
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.status = 'failed'
            self.error = f'{exc_type.__name__}: {exc_value}'
        if self.csv_path:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, self.interval * 2))
            try:
                self._samples.append(_sample_cpu_memory(self.pid))
            except Exception:
                self._samples.append({})
            self._write_row(self._samples)
        return False

# Define labels for dataframes
def generate_dataframe_labels():
    # labels for trajectory
    trajectory_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'ipSAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes',
                        'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues',
                        'n_InterfaceHbonds', 'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
                        'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD', 'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    # labels for mpnn designs
    core_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'ipSAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                    'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                    'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%', 
                    'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD', 'Binder_pLDDT', 'Binder_pTM', 'Binder_pAE', 'Binder_RMSD']

    design_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'MPNN_score', 'MPNN_seq_recovery']

    for label in core_labels:
        design_labels += ['Average_' + label] + [f'{i}_{label}' for i in range(1, 6)]

    design_labels += ['DesignTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    final_labels = ['Rank'] + design_labels

    return trajectory_labels, design_labels, final_labels

# Create base directions of the project
def generate_directories(design_path):
    t0 = time.time()
    design_path_names = ["Accepted", "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle", "Trajectory",
                        "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing", "Trajectory/LowConfidence", "Trajectory/Animation",
                        "Trajectory/Pickle", "MPNN", "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed", "Rejected"]
    design_paths = {}

    # make directories and set design_paths[FOLDER_NAME] variable
    for name in design_path_names:
        path = os.path.join(design_path, name)
        os.makedirs(path, exist_ok=True)
        design_paths[name] = path

    vprint(f"[GenUtils] Directories prepared in {time.time()-t0:.2f}s")
    return design_paths

# generate CSV file for tracking designs not passing filters
def generate_filter_pass_csv(failure_csv, filter_json):
    t0 = time.time()
    if not os.path.exists(failure_csv):
        with open(filter_json, 'r') as file:
            data = json.load(file)
        
        # Create a list of modified keys
        names = ['Trajectory_logits_pLDDT', 'Trajectory_softmax_pLDDT', 'Trajectory_one-hot_pLDDT', 'Trajectory_final_pLDDT', 'Trajectory_Contacts', 'Trajectory_Clashes', 'Trajectory_WrongHotspot']
        special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
        tracked_filters = set()

        for key in data.keys():
            processed_name = key  # Use the full key by default

            # Check if the key starts with any special prefixes
            for prefix in special_prefixes:
                if key.startswith(prefix):
                    # Strip the prefix and use the remaining part
                    processed_name = key.split('_', 1)[1]
                    break

            # Handle 'InterfaceAAs' with appending amino acids
            if 'InterfaceAAs' in processed_name:
                # Generate 20 variations of 'InterfaceAAs' with amino acids appended
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                for aa in amino_acids:
                    variant_name = f"InterfaceAAs_{aa}"
                    if variant_name not in tracked_filters:
                        names.append(variant_name)
                        tracked_filters.add(variant_name)
            elif processed_name not in tracked_filters:
                # Add processed name if it hasn't been added before
                names.append(processed_name)
                tracked_filters.add(processed_name)

        # make dataframe with 0s
        df = pd.DataFrame(columns=names)
        df.loc[0] = [0] * len(names)

        df.to_csv(failure_csv, index=False)
        vprint(f"[GenUtils] Initialized failure CSV in {time.time()-t0:.2f}s")

# update failure rates from trajectories and early predictions
def update_failures(failure_csv, failure_column_or_dict):
    t0 = time.time()
    failure_df = pd.read_csv(failure_csv)
    
    def strip_model_prefix(name):
        # Strips the model-specific prefix if it exists
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name
    
    # update dictionary coming from complex prediction
    if isinstance(failure_column_or_dict, dict):
        # Update using a dictionary of failures
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        # Update a single column from trajectory generation
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1
    
    failure_df.to_csv(failure_csv, index=False)
    vprint(f"[GenUtils] Updated failure CSV in {time.time()-t0:.2f}s")

# Check if number of trajectories generated
def check_n_trajectories(design_paths, advanced_settings):
    n_trajectories = [f for f in os.listdir(design_paths["Trajectory/Relaxed"]) if f.endswith('.pdb') and not f.startswith('.')]

    if advanced_settings["max_trajectories"] is not False and len(n_trajectories) >= advanced_settings["max_trajectories"]:
        print(f"Target number of {str(len(n_trajectories))} trajectories reached, stopping execution...")
        return True
    else:
        return False

# Check if we have required number of accepted targets, rank them, and analyse sequence and structure properties
def check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels, rank_by='Average_i_pTM'):
    t0 = time.time()
    accepted_binders = [f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb') and not f.startswith('.')]

    if len(accepted_binders) >= target_settings["number_of_final_designs"]:
        print(f"Target number {str(len(accepted_binders))} of designs reached! Reranking by {rank_by}...")

        # clear the Ranked folder in case we added new designs in the meantime so we rerank them all
        for f in os.listdir(design_paths["Accepted/Ranked"]):
            os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

        # load dataframe of designed binders
        design_df = pd.read_csv(mpnn_csv)
        design_df = design_df.sort_values(rank_by, ascending=False)
        
        # create final csv dataframe to copy matched rows, initialize with the column labels
        final_df = pd.DataFrame(columns=final_labels)

        # check the ranking of the designs and copy them with new ranked IDs to the folder
        rank = 1
        for _, row in design_df.iterrows():
            for binder in accepted_binders:
                target_settings["binder_name"], model = binder.rsplit('_model', 1)
                if target_settings["binder_name"] == row['Design']:
                    # rank and copy into ranked folder
                    row_data = {'Rank': rank, **{label: row[label] for label in design_labels}}
                    final_df = pd.concat([final_df, pd.DataFrame([row_data])], ignore_index=True)
                    old_path = os.path.join(design_paths["Accepted"], binder)
                    new_path = os.path.join(design_paths["Accepted/Ranked"], f"{rank}_{target_settings['binder_name']}_model{model.rsplit('.', 1)[0]}.pdb")
                    shutil.copyfile(old_path, new_path)

                    rank += 1
                    break

        # save the final_df to final_csv
        final_df.to_csv(final_csv, index=False)
        vprint(f"[GenUtils] Reranking and final CSV write in {time.time()-t0:.2f}s")

        # zip large folders to save space
        if advanced_settings["zip_animations"]:
            zip_and_empty_folder(design_paths["Trajectory/Animation"], '.html')

        if advanced_settings["zip_plots"]:
            zip_and_empty_folder(design_paths["Trajectory/Plots"], '.png')

        return True

    else:
        vprint(f"[GenUtils] Accepted count below target; check in {time.time()-t0:.2f}s")
        return False

# Load required helicity value
def load_helicity(advanced_settings):
    if advanced_settings["random_helicity"] is True:
        # will sample a random bias towards helicity
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # using a preset helicity bias
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # no bias towards helicity
        helicity_value = 0
    return helicity_value

# Report JAX-capable devices
def check_jax_gpu():
    if jax is None:
        print("JAX is not installed, terminating.")
        exit()
    devices = jax.devices()

    has_gpu = any(device.platform == 'gpu' for device in devices)

    if not has_gpu:
        print("No GPU device found, terminating.")
        exit()
    else:
        print("Available GPUs:")
        for i, device in enumerate(devices):
            print(f"{device.device_kind}{i + 1}: {device.platform}")

# check all input files being passed
def perform_input_check(args):
    # Get the directory of the current script
    binder_script_path = os.path.dirname(os.path.abspath(__file__))

    # Ensure settings file is provided
    if not args.settings:
        print("Error: --settings is required.")
        exit()

    # Set default filters.json path if not provided
    if not args.filters:
        args.filters = os.path.join(binder_script_path, 'settings_filters', 'default_filters.json')

    # Set a random advanced json settings file if not provided
    if not args.advanced:
        args.advanced = os.path.join(binder_script_path, 'settings_advanced', 'default_4stage_multimer.json')

    return args.settings, args.filters, args.advanced

# check specific advanced settings
def perform_advanced_settings_check(advanced_settings, bindcraft_folder):
    # set paths to model weights and executables
    if bindcraft_folder == "colab":
        advanced_settings["af_params_dir"] = '/content/bindcraft/params/'
        advanced_settings["dssp_path"] = '/content/bindcraft/functions/dssp'
        advanced_settings["dalphaball_path"] = '/content/bindcraft/functions/DAlphaBall.gcc'
    else:
        # Set paths individually if they are not already set
        if not advanced_settings["af_params_dir"]:
            advanced_settings["af_params_dir"] = bindcraft_folder
        if not advanced_settings["dssp_path"]:
            advanced_settings["dssp_path"] = os.path.join(bindcraft_folder, 'functions', 'dssp')
        if not advanced_settings["dalphaball_path"]:
            advanced_settings["dalphaball_path"] = os.path.join(bindcraft_folder, 'functions', 'DAlphaBall.gcc')

    # check formatting of omit_AAs setting
        omit_aas = advanced_settings["omit_AAs"]
    if advanced_settings["omit_AAs"] in [None, False, '']:
        advanced_settings["omit_AAs"] = None
    elif isinstance(advanced_settings["omit_AAs"], str):
        advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

    # Ensure default toggles for plots/animations if missing
    if "save_design_trajectory_plots" not in advanced_settings:
        advanced_settings["save_design_trajectory_plots"] = True
    if "save_design_animations" not in advanced_settings:
        advanced_settings["save_design_animations"] = True

    # Ensure required executables are present and executable (chmod +x if needed)
    _ensure_required_executables(advanced_settings, bindcraft_folder)

    return advanced_settings

def _ensure_required_executables(advanced_settings, bindcraft_folder):
    """
    Ensure bundled helper binaries have the executable bit set.
    We avoid raising if files are missing; callers decide presence.
    """
    try:
        # DSSP (bundled path)
        dssp_path = advanced_settings.get("dssp_path") or os.path.join(bindcraft_folder, 'functions', 'dssp')
        if isinstance(dssp_path, str) and os.path.isfile(dssp_path):
            st = os.stat(dssp_path)
            if not (st.st_mode & stat.S_IXUSR):
                try:
                    os.chmod(dssp_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception:
                    pass

        # DAlphaBall (only relevant if PyRosetta is used)
        dalphaball_path = advanced_settings.get("dalphaball_path") or os.path.join(bindcraft_folder, 'functions', 'DAlphaBall.gcc')
        if isinstance(dalphaball_path, str) and os.path.isfile(dalphaball_path):
            st = os.stat(dalphaball_path)
            if not (st.st_mode & stat.S_IXUSR):
                try:
                    os.chmod(dalphaball_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except Exception:
                    pass
    except Exception:
        # Never crash on permission fix attempts
        pass

# Load settings from JSONs
def load_json_settings(settings_json, filters_json, advanced_json):
    # load settings from json files
    with open(settings_json, 'r') as file:
        target_settings = json.load(file)

    with open(advanced_json, 'r') as file:
        advanced_settings = json.load(file)

    with open(filters_json, 'r') as file:
        filters = json.load(file)

    return target_settings, advanced_settings, filters

# AF2 model settings, make sure non-overlapping models with template option are being used for design and re-prediction
def load_af2_models(af_multimer_setting):
    if af_multimer_setting:
        design_models = [0,1,2,3,4]
        prediction_models = [0,1]
        multimer_validation = False
    else:
        design_models = [0,1]
        prediction_models = [0,1,2,3,4]
        multimer_validation = True

    return design_models, prediction_models, multimer_validation

# migrate existing CSV to include new columns (backwards compatibility)
def migrate_csv_columns(csv_path, expected_columns):
    """
    Migrate existing CSV to include new columns added in later versions.
    
    Inserts missing columns at their correct positions with None values
    for existing rows, ensuring backwards compatibility when resuming jobs.
    
    Args:
        csv_path: Path to the CSV file
        expected_columns: List of column names in the expected order
    
    Returns:
        True if migration was performed, False if no migration needed
    """
    if not os.path.exists(csv_path):
        return False
    
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        # Empty file, will be recreated with correct headers
        return False
    except Exception as e:
        print(f"Warning: Could not read {csv_path} for migration: {e}")
        return False
    
    if df.empty and len(df.columns) == 0:
        return False
    
    existing_columns = list(df.columns)
    
    # Check if migration is needed
    if existing_columns == expected_columns:
        return False
    
    # Check if we actually need to add any new columns
    new_cols = set(expected_columns) - set(existing_columns)
    needs_reorder = existing_columns != expected_columns
    
    if not new_cols and not needs_reorder:
        return False
    
    # Build column data dictionary efficiently (avoids fragmentation warnings)
    col_data = {}
    for col in expected_columns:
        if col in existing_columns:
            col_data[col] = df[col].values
        else:
            # New column - fill with None for existing rows
            col_data[col] = [None] * len(df)
    
    # Create DataFrame in one shot from dictionary (efficient, no warnings)
    migrated_df = pd.DataFrame(col_data, columns=expected_columns)
    
    if new_cols:
        print(f"Migrating {os.path.basename(csv_path)}: adding columns {new_cols}")
        migrated_df.to_csv(csv_path, index=False)
        return True
    
    # Columns might just be reordered (shouldn't happen, but handle gracefully)
    if needs_reorder:
        print(f"Reordering columns in {os.path.basename(csv_path)}")
        migrated_df.to_csv(csv_path, index=False)
        return True
    
    return False

# create csv for insertion of data
def create_dataframe(csv_file, columns):
    t0 = time.time()
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
        vprint(f"[GenUtils] Created CSV {csv_file} in {time.time()-t0:.2f}s")

# insert row of statistics into csv
def insert_data(csv_file, data_array):
    t0 = time.time()
    df = pd.DataFrame([data_array])
    df.to_csv(csv_file, mode='a', header=False, index=False)
    vprint(f"[GenUtils] Appended row to {csv_file} in {time.time()-t0:.2f}s")

# save generated sequence
def save_fasta(design_name, sequence, design_paths):
    fasta_path = os.path.join(design_paths["MPNN/Sequences"], design_name+".fasta")
    with open(fasta_path,"w") as fasta:
        line = f'>{design_name}\n{sequence}'
        fasta.write(line+"\n")

# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [
            line for line in f_in
            if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK', 'CONECT', 'SSBOND'))
        ]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)

def zip_and_empty_folder(folder_path, extension):
    folder_basename = os.path.basename(folder_path)
    zip_filename = os.path.join(os.path.dirname(folder_path), folder_basename + '.zip')

    # Open the zip file in 'a' mode to append if it exists, otherwise create a new one
    with zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                # Create an absolute path
                file_path = os.path.join(folder_path, file)
                # Add file to zip file, replacing it if it already exists
                zipf.write(file_path, arcname=file)
                # Remove the file after adding it to the zip
                os.remove(file_path)
    print(f"Files in folder '{folder_path}' have been zipped and removed.")

# calculate averages for statistics
def calculate_averages(statistics, handle_aa=False):
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(1, 6):  # assumes models are numbered 1 through 5
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it separately, do so
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid, initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages

# filter designs based on feature thresholds
def check_filters(mpnn_data, design_labels, filters):
    # check mpnn_data against labels
    mpnn_dict = {label: value for label, value in zip(design_labels, mpnn_data)}

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label == 'Average_InterfaceAAs' or label == '1_InterfaceAAs' or label == '2_InterfaceAAs' or label == '3_InterfaceAAs' or label == '4_InterfaceAAs' or label == '5_InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if mpnn_dict.get(label) is None:
                    continue
                value = mpnn_dict.get(label).get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            # if no threshold, then skip
            value = mpnn_dict.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return True
    # if some filters were unmet, print them out
    else:
        return unmet_conditions
