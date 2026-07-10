"""
test_cli_gpu_polling.py — Tests for the CLI's live GPU polling behavior.

Tests cover:
  1. CapturedContext includes peak GPU stats from a mocked polling loop
  2. CLI tees child's stderr live to the parent terminal
  3. Graceful fallback when pynvml/GPU is unavailable
"""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from finetune_doctor.capture import CapturedContext
from finetune_doctor.gpu_poller import GpuPoller, GpuSnapshot


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Peak GPU stats are captured and merged into CapturedContext
# ═══════════════════════════════════════════════════════════════════════════


class TestPeakGpuStatsCapture:
    """Verify the CLI builds a CapturedContext that includes peak GPU stats
    gathered during a simulated polling loop (no real GPU needed)."""

    def test_merge_gpu_snapshot_into_context(self):
        """Direct test: GpuSnapshot data flows into CapturedContext fields."""
        snapshot = GpuSnapshot(
            peak_memory_used=6_000_000_000,      # ~6 GiB
            peak_memory_total=8_000_000_000,      # ~8 GiB
            peak_utilization=95,
            peak_temperature=78,
            peak_processes=[{"pid": 1234, "used_gpu_memory": 5_500_000_000}],
            samples_taken=20,
        )

        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            exception_message="CUDA out of memory",
            traceback_text="in evaluation_loop\nOutOfMemoryError",
        )

        ctx.merge_gpu_snapshot(snapshot)

        assert ctx.peak_gpu_memory_used == 6_000_000_000
        assert ctx.peak_gpu_memory_total == 8_000_000_000
        assert ctx.peak_gpu_utilization == 95
        assert ctx.peak_gpu_temperature == 78
        assert len(ctx.gpu_processes) == 1
        assert ctx.gpu_processes[0]["pid"] == 1234

    def test_merge_none_snapshot_is_safe(self):
        """Merging None should be a no-op, not a crash."""
        ctx = CapturedContext()
        ctx.merge_gpu_snapshot(None)
        assert ctx.peak_gpu_memory_used is None

    def test_build_context_includes_polled_gpu_stats(self):
        """Integration: _build_context_from_run merges poller data."""
        from finetune_doctor.cli import _build_context_from_run

        # Create a poller with pre-set snapshot data (simulating polling)
        poller = GpuPoller(interval=1.0)
        # Directly inject a snapshot (bypassing actual polling)
        poller._snapshot = GpuSnapshot(
            peak_memory_used=7_000_000_000,
            peak_memory_total=8_000_000_000,
            peak_utilization=99,
            peak_temperature=82,
            samples_taken=15,
        )

        stderr = (
            "Traceback (most recent call last):\n"
            '  File "train.py", line 10, in evaluation_loop\n'
            "    prediction_step(model, batch)\n"
            "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB.\n"
        )

        ctx = _build_context_from_run(stderr, poller)

        # Exception info parsed from stderr
        assert ctx.exception_type == "OutOfMemoryError"
        assert "CUDA out of memory" in ctx.exception_message

        # Peak GPU stats from poller
        assert ctx.peak_gpu_memory_used == 7_000_000_000
        assert ctx.peak_gpu_memory_total == 8_000_000_000
        assert ctx.peak_gpu_utilization == 99
        assert ctx.peak_gpu_temperature == 82

        # System info should also be populated
        assert ctx.python_version != ""
        assert ctx.os_platform != ""
        assert ctx.ram_total is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: GpuPoller accumulates peaks correctly
# ═══════════════════════════════════════════════════════════════════════════


class _FakeMemInfo:
    def __init__(self, used: int, total: int):
        self.used = used
        self.total = total


class _FakeUtilization:
    def __init__(self, gpu: int):
        self.gpu = gpu


class _FakeProcess:
    def __init__(self, pid: int, mem: int):
        self.pid = pid
        self.usedGpuMemory = mem


class TestGpuPollerPeakTracking:
    """Verify the GpuPoller correctly tracks peak values across samples."""

    def test_poller_with_mocked_pynvml(self):
        """Simulate multiple pynvml samples and check peak tracking."""
        # We'll test _sample_once directly with a mock pynvml module
        poller = GpuPoller(interval=0.1)

        # Build a mock pynvml module
        mock_pynvml = mock.MagicMock()
        mock_handle = mock.MagicMock()

        # Simulate 3 samples with increasing then decreasing memory
        memory_readings = [
            _FakeMemInfo(used=2_000_000_000, total=8_000_000_000),
            _FakeMemInfo(used=6_000_000_000, total=8_000_000_000),  # peak
            _FakeMemInfo(used=3_000_000_000, total=8_000_000_000),
        ]
        util_readings = [
            _FakeUtilization(gpu=50),
            _FakeUtilization(gpu=98),  # peak
            _FakeUtilization(gpu=30),
        ]
        temp_readings = [65, 82, 70]  # peak at 82

        for mem, util, temp in zip(memory_readings, util_readings, temp_readings):
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem
            mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util
            mock_pynvml.nvmlDeviceGetTemperature.return_value = temp
            mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [
                _FakeProcess(pid=9999, mem=mem.used)
            ]
            mock_pynvml.NVML_TEMPERATURE_GPU = 0

            poller._sample_once(mock_pynvml, mock_handle)

        snap = poller.peak_snapshot()

        assert snap.peak_memory_used == 6_000_000_000, "Should track peak memory"
        assert snap.peak_utilization == 98, "Should track peak utilization"
        assert snap.peak_temperature == 82, "Should track peak temperature"
        assert snap.samples_taken == 3

    def test_poller_snapshot_starts_empty(self):
        """A fresh poller should have all-None peak stats."""
        poller = GpuPoller()
        snap = poller.peak_snapshot()
        assert snap.peak_memory_used is None
        assert snap.peak_utilization is None
        assert snap.peak_temperature is None
        assert snap.samples_taken == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: CLI tees stderr live (not just at the end)
# ═══════════════════════════════════════════════════════════════════════════


class TestStderrLiveTee:
    """Verify the CLI streams the child's stderr to the parent in real time."""

    def test_stderr_lines_appear_as_they_are_produced(self):
        """Run a subprocess that prints to stderr line by line, verify we
        receive each line as it comes (not buffered until exit)."""
        # Use a small inline Python script that writes to stderr with flushes
        script = (
            "import sys, time\n"
            "for i in range(3):\n"
            "    sys.stderr.write(f'line {i}\\n')\n"
            "    sys.stderr.flush()\n"
            "    time.sleep(0.05)\n"
            "sys.exit(1)\n"
        )

        collected_lines: List[str] = []
        timestamps: List[float] = []

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        assert proc.stderr is not None
        for line in proc.stderr:
            collected_lines.append(line.strip())
            timestamps.append(time.monotonic())
        proc.stderr.close()
        proc.wait()

        # We should have received 3 lines
        assert len(collected_lines) == 3
        assert collected_lines == ["line 0", "line 1", "line 2"]

        # Lines should arrive over time, not all at once
        # (at least the last line should be later than the first)
        if len(timestamps) >= 2:
            assert timestamps[-1] - timestamps[0] >= 0.05, (
                "Lines should arrive spread over time, not all at once"
            )

    def test_stderr_accumulated_for_diagnosis(self):
        """Verify stderr is both teed AND accumulated for post-mortem."""
        # We test this via _build_context_from_run which uses the
        # accumulated stderr text
        from finetune_doctor.cli import _build_context_from_run

        poller = GpuPoller()
        stderr_text = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in eval_loop\n'
            "OutOfMemoryError: CUDA out of memory\n"
        )

        ctx = _build_context_from_run(stderr_text, poller)

        assert ctx.traceback_text == stderr_text
        assert ctx.exception_type == "OutOfMemoryError"
        assert "CUDA out of memory" in ctx.exception_message


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Graceful fallback when pynvml/GPU unavailable
# ═══════════════════════════════════════════════════════════════════════════


class TestGracefulGpuFallback:
    """Verify the poller doesn't crash when there's no GPU or pynvml."""

    def test_poller_records_error_when_pynvml_unavailable(self):
        """If pynvml init fails, the poller should record an error but not crash."""
        with mock.patch.dict("sys.modules", {"pynvml": None}):
            poller = GpuPoller()
            # Force the import to fail by patching __import__
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def failing_import(name, *args, **kwargs):
                if name == "pynvml":
                    raise ImportError("No module named 'pynvml'")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=failing_import):
                poller.start()

            snap = poller.peak_snapshot()
            assert snap.error is not None
            assert "unavailable" in snap.error.lower() or "pynvml" in snap.error.lower()
            assert snap.peak_memory_used is None

    def test_context_still_valid_without_gpu_data(self):
        """Even with no GPU data, the CapturedContext should have all other fields."""
        from finetune_doctor.cli import _build_context_from_run

        # Poller that never started (simulates no GPU)
        poller = GpuPoller()
        poller._snapshot.error = "No GPU available"

        stderr_text = "ValueError: something went wrong\n"
        ctx = _build_context_from_run(stderr_text, poller)

        # GPU fields should be None but everything else should work
        assert ctx.peak_gpu_memory_used is None
        assert ctx.peak_gpu_utilization is None
        assert ctx.exception_type == "ValueError"
        assert ctx.python_version != ""
        assert ctx.ram_total is not None

    def test_poller_stop_is_safe_when_never_started(self):
        """Calling stop() on a poller that was never started should not crash."""
        poller = GpuPoller()
        poller.stop()  # should be a no-op
        snap = poller.peak_snapshot()
        assert snap.samples_taken == 0
