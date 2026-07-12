"""
cli.py — Typer-based CLI entry point for finetune-doctor.

Provides a ``run`` command that wraps an arbitrary training script, capturing
failures and running them through the diagnosis pipeline.

The CLI path now:
  1. Launches the user's command via ``subprocess.Popen`` with stderr piped.
  2. Tees the child's stdout/stderr to the parent terminal in real time.
  3. Polls GPU stats on a background thread during the run (peak tracking).
  4. On failure, builds a rich ``CapturedContext`` combining the traceback,
     peak GPU stats, system RAM, and package versions — then runs the same
     ``matcher.py`` used by the in-process ``watch()`` path.

Usage:
    finetune-doctor run -- python examples/example_training_script.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
from typing import List, Optional

import typer

from finetune_doctor import __version__
from finetune_doctor.capture import CapturedContext
from finetune_doctor.gpu_poller import GpuPoller
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import load_signatures
from finetune_doctor.watch import (
    _render_diagnosis,
    _render_no_match,
    console,
    _get_safe_symbol,
)

app = typer.Typer(
    name="finetune-doctor",
    help="Catches LLM fine-tuning failures and explains them in plain English.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"finetune-doctor [bold]{__version__}[/bold]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """🩺 finetune-doctor — catch training failures before they waste your time."""


@app.command()
def run(
    command: List[str] = typer.Argument(
        ...,
        help="The training command to execute, e.g.: python train.py",
    ),
) -> None:
    """Run a training command and diagnose any failure.

    Usage:
        finetune-doctor run -- python examples/example_training_script.py
    """
    if not command:
        console.print("[red]Error: no command provided.[/red]")
        raise typer.Exit(code=1)

    doc_sym = _get_safe_symbol("🩺", "[finetune-doctor]")
    console.print(
        f"[bold cyan]{doc_sym} finetune-doctor[/bold cyan] wrapping: "
        f"[dim]{' '.join(command)}[/dim]"
    )
    console.print()

    # ── Start GPU poller ────────────────────────────────────────────────
    gpu_poller = GpuPoller(interval=0.5)
    gpu_poller.start()

    # ── Launch subprocess with piped stderr (stdout passed through) ─────
    stderr_chunks: List[str] = []

    # Configure env to enforce UTF-8 output from Python subprocesses
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            command,
            stdout=None,           # inherit — streams directly to terminal
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,             # line-buffered
        )

        # Tee stderr: read line by line, write to parent's stderr live,
        # and accumulate for post-mortem analysis.
        if proc.stderr is not None:
            for line in proc.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_chunks.append(line)
            proc.stderr.close()

        returncode = proc.wait()

    except FileNotFoundError:
        console.print(
            f"[bold red]Error: command not found: {command[0]}[/bold red]"
        )
        gpu_poller.stop()
        raise typer.Exit(code=127)

    finally:
        # ── Stop GPU poller ─────────────────────────────────────────────
        gpu_poller.stop()

    # ── Handle success ──────────────────────────────────────────────────
    if returncode == 0:
        success_sym = _get_safe_symbol("✅", "SUCCESS:")
        console.print(
            f"[bold green]{success_sym} Command completed successfully.[/bold green]"
        )
        return

    # ── Handle failure — build CapturedContext and diagnose ─────────────
    stderr_text = "".join(stderr_chunks)
    ctx = _build_context_from_run(stderr_text, gpu_poller)

    # Run matcher
    signatures = load_signatures()
    matched = match(ctx, signatures)

    if matched is not None:
        _render_diagnosis(matched, ctx)
    else:
        console.print()
        console.print(
            f"[bold red]Command exited with code {returncode}.[/bold red]"
        )
        console.print(
            "[dim italic]finetune-doctor: no known failure signature matched "
            "this error.  If you'd like to help improve finetune-doctor, "
            "please open an issue at "
            "https://github.com/finetune-doctor/finetune-doctor/issues "
            "with this traceback.[/dim italic]"
        )

    raise typer.Exit(code=returncode)


def _build_context_from_run(
    stderr_text: str, gpu_poller: GpuPoller
) -> CapturedContext:
    """Build a CapturedContext from subprocess stderr + polled GPU stats.

    Combines:
      - Exception type/message parsed from the traceback text
      - Peak GPU stats gathered by the background poller during the run
      - System RAM via psutil
      - Package versions from the outer (finetune-doctor's) environment
        (TODO: handle the case where the subprocess uses a different venv)
      - Python version and OS platform
    """
    ctx = CapturedContext()
    ctx.traceback_text = stderr_text

    # ── Parse exception type + message from stderr ──────────────────────
    _parse_exception_from_stderr(ctx, stderr_text)

    # ── Merge peak GPU stats from poller ────────────────────────────────
    snapshot = gpu_poller.peak_snapshot()
    ctx.merge_gpu_snapshot(snapshot)

    # ── System RAM via psutil ───────────────────────────────────────────
    try:
        import psutil

        vm = psutil.virtual_memory()
        ctx.ram_total = vm.total
        ctx.ram_available = vm.available
        ctx.ram_used = vm.used
        ctx.ram_percent = vm.percent
    except Exception:
        pass

    # ── Package versions (from *this* environment) ──────────────────────
    # TODO: This reflects the outer environment where finetune-doctor is
    # installed.  If the subprocess uses a different venv (e.g., the user
    # runs `conda run` or a different Python), these versions may not match
    # what the training script actually used.  A future chunk should detect
    # and handle this case.
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
            pass

    # ── Python / OS ─────────────────────────────────────────────────────
    try:
        ctx.python_version = sys.version
        ctx.os_platform = platform.platform()
    except Exception:
        pass

    return ctx


def _parse_exception_from_stderr(ctx: CapturedContext, stderr_text: str) -> None:
    """Extract exception type and message from a Python traceback in stderr.

    Python tracebacks typically end with a line like:
        ExceptionType: message text here
    """
    if not stderr_text.strip():
        return

    for line in reversed(stderr_text.strip().splitlines()):
        line = line.strip()
        if line and not line.startswith("File ") and not line.startswith("^"):
            if ": " in line:
                ctx.exception_type = line.split(":")[0].strip()
                ctx.exception_message = line.split(":", 1)[1].strip()
            elif line:
                ctx.exception_type = line
                ctx.exception_message = ""
            break
