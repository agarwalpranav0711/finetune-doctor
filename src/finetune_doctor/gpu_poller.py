"""
gpu_poller.py — Background-thread GPU stats poller using pynvml.

Runs alongside a subprocess, sampling GPU memory and utilization at a
configurable interval.  Tracks *peak* values so that even after the
subprocess dies (and its GPU memory is freed), we still know the high-water
mark that likely caused the OOM.

Usage::

    poller = GpuPoller(interval=0.5)
    poller.start()
    # ... subprocess runs ...
    poller.stop()
    snapshot = poller.peak_snapshot()  # dict with peak stats
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GpuSnapshot:
    """Peak GPU stats accumulated during a polling session."""

    peak_memory_used: Optional[int] = None       # bytes — highest observed
    peak_memory_total: Optional[int] = None       # bytes — GPU total (constant)
    peak_utilization: Optional[int] = None        # 0-100 %
    peak_temperature: Optional[int] = None        # °C
    peak_processes: List[Dict[str, Any]] = field(default_factory=list)
    samples_taken: int = 0
    polling_active: bool = False
    error: Optional[str] = None  # why polling couldn't start / failed


class GpuPoller:
    """Polls GPU stats on a background daemon thread.

    Parameters
    ----------
    interval : float
        Seconds between polls.  Default 0.5s.
    device_index : int
        NVML device index to poll.  Default 0 (first GPU).
    """

    def __init__(self, interval: float = 0.5, device_index: int = 0) -> None:
        self._interval = interval
        self._device_index = device_index

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Accumulated peak stats — accessed from both threads, but only
        # the poller thread writes during polling and the main thread
        # reads after stop(), so no lock is needed.
        self._snapshot = GpuSnapshot()

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background polling thread.

        If pynvml isn't available or no GPU is present, records the reason
        in ``snapshot.error`` and returns immediately — never crashes.
        """
        # Pre-check: can we even import pynvml?
        try:
            import pynvml
            pynvml.nvmlInit()
            pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            pynvml.nvmlShutdown()
        except Exception as exc:
            self._snapshot.error = f"GPU polling unavailable: {exc}"
            return

        self._snapshot.polling_active = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="finetune-doctor-gpu-poller",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._snapshot.polling_active = False

    def peak_snapshot(self) -> GpuSnapshot:
        """Return the accumulated peak GPU stats."""
        return self._snapshot

    # ── Internal polling loop ───────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Runs on the background thread — polls pynvml until stopped."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)

            # Get total memory once (it doesn't change)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._snapshot.peak_memory_total = mem_info.total
            except Exception:
                pass

            while not self._stop_event.is_set():
                self._sample_once(pynvml, handle)
                self._stop_event.wait(timeout=self._interval)

            # One final sample right before stopping
            self._sample_once(pynvml, handle)

            pynvml.nvmlShutdown()
        except Exception as exc:
            self._snapshot.error = f"GPU polling failed: {exc}"

    def _sample_once(self, pynvml: Any, handle: Any) -> None:
        """Take a single GPU stats sample and update peaks."""
        self._snapshot.samples_taken += 1

        # Memory used
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used
            if (
                self._snapshot.peak_memory_used is None
                or used > self._snapshot.peak_memory_used
            ):
                self._snapshot.peak_memory_used = used
        except Exception:
            pass

        # GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            if (
                self._snapshot.peak_utilization is None
                or util.gpu > self._snapshot.peak_utilization
            ):
                self._snapshot.peak_utilization = util.gpu
        except Exception:
            pass

        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            if (
                self._snapshot.peak_temperature is None
                or temp > self._snapshot.peak_temperature
            ):
                self._snapshot.peak_temperature = temp
        except Exception:
            pass

        # Per-process memory (snapshot, not accumulated — we keep the latest)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            self._snapshot.peak_processes = [
                {"pid": p.pid, "used_gpu_memory": p.usedGpuMemory}
                for p in procs
            ]
        except Exception:
            pass
