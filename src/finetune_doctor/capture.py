"""
capture.py — Gathers GPU, environment, and configuration state at the moment of failure.

Every capture function is individually wrapped in try/except so that a missing
dependency (torch, pynvml, etc.) or a hardware difference (no NVIDIA GPU) never
crashes the diagnostic tool itself — the worst case is an empty field in the
captured context dict.
"""

from __future__ import annotations

import platform
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CapturedContext:
    """Structured snapshot of everything we know at the moment of failure."""

    # ── Exception info ──────────────────────────────────────────────────
    exception_type: str = ""
    exception_message: str = ""
    traceback_text: str = ""

    # ── CUDA / torch memory (bytes) ─────────────────────────────────────
    cuda_memory_allocated: Optional[int] = None
    cuda_memory_reserved: Optional[int] = None
    cuda_total_memory: Optional[int] = None

    # ── pynvml GPU stats (point-in-time, from in-process capture) ───────
    gpu_utilization: Optional[int] = None
    gpu_temperature: Optional[int] = None
    gpu_processes: List[Dict[str, Any]] = field(default_factory=list)

    # ── Peak GPU stats (from CLI background polling) ────────────────────
    peak_gpu_memory_used: Optional[int] = None
    peak_gpu_memory_total: Optional[int] = None
    peak_gpu_utilization: Optional[int] = None
    peak_gpu_temperature: Optional[int] = None

    # ── System RAM ──────────────────────────────────────────────────────
    ram_total: Optional[int] = None
    ram_available: Optional[int] = None
    ram_used: Optional[int] = None
    ram_percent: Optional[float] = None

    # ── Package versions ────────────────────────────────────────────────
    package_versions: Dict[str, str] = field(default_factory=dict)

    # ── Platform ────────────────────────────────────────────────────────
    python_version: str = ""
    os_platform: str = ""

    # ── Training hints (optional, passed explicitly by the user) ────────
    dataset_format_hint: Optional[str] = None

    def merge_gpu_snapshot(self, snapshot: Any) -> None:
        """Merge peak GPU stats from a ``GpuSnapshot`` into this context.

        Parameters
        ----------
        snapshot : gpu_poller.GpuSnapshot
            The accumulated peak stats from background polling.
        """
        if snapshot is None:
            return
        self.peak_gpu_memory_used = snapshot.peak_memory_used
        self.peak_gpu_memory_total = snapshot.peak_memory_total
        self.peak_gpu_utilization = snapshot.peak_utilization
        self.peak_gpu_temperature = snapshot.peak_temperature
        if snapshot.peak_processes:
            self.gpu_processes = snapshot.peak_processes


def capture(exc: BaseException) -> CapturedContext:
    """Capture everything we can about the current failure environment.

    Parameters
    ----------
    exc : BaseException
        The exception that was raised.

    Returns
    -------
    CapturedContext
        A structured snapshot — fields will be empty/None where info was
        unavailable rather than raising.
    """
    ctx = CapturedContext()

    # ── 1. Exception info ───────────────────────────────────────────────
    try:
        ctx.exception_type = type(exc).__name__
        ctx.exception_message = str(exc)
        ctx.traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    except Exception:
        pass

    # ── 2. torch CUDA memory ────────────────────────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            ctx.cuda_memory_allocated = torch.cuda.memory_allocated(0)
            ctx.cuda_memory_reserved = torch.cuda.memory_reserved(0)
            props = torch.cuda.get_device_properties(0)
            ctx.cuda_total_memory = props.total_mem
    except Exception:
        pass

    # ── 3. pynvml GPU stats ─────────────────────────────────────────────
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            ctx.gpu_utilization = util.gpu
        except Exception:
            pass

        try:
            ctx.gpu_temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            ctx.gpu_processes = [
                {"pid": p.pid, "used_gpu_memory": p.usedGpuMemory}
                for p in procs
            ]
        except Exception:
            pass

        pynvml.nvmlShutdown()
    except Exception:
        pass

    # ── 4. System RAM via psutil ────────────────────────────────────────
    try:
        import psutil

        vm = psutil.virtual_memory()
        ctx.ram_total = vm.total
        ctx.ram_available = vm.available
        ctx.ram_used = vm.used
        ctx.ram_percent = vm.percent
    except Exception:
        pass

    # ── 5. Package versions ─────────────────────────────────────────────
    _PACKAGES_TO_CHECK = [
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "unsloth",
    ]
    for pkg in _PACKAGES_TO_CHECK:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed (version unknown)")
            ctx.package_versions[pkg] = ver
        except Exception:
            pass  # package not installed — simply skip

    # ── 6. Python / OS ──────────────────────────────────────────────────
    try:
        ctx.python_version = sys.version
        ctx.os_platform = platform.platform()
    except Exception:
        pass

    return ctx
