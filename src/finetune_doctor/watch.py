"""
watch.py — The core context manager and decorator that wraps user training code.

Usage (context manager):
    from finetune_doctor import watch

    with watch():
        trainer.train()

    # With optional dataset format hint:
    with watch(dataset_format_hint="chatml"):
        trainer.train()

Usage (decorator):
    from finetune_doctor import watch

    @watch()
    def run_training():
        trainer.train()

On success: exits silently with zero overhead.
On exception: captures context, matches against signatures, prints a rich
diagnosis panel, then re-raises the original exception.
"""

from __future__ import annotations

import functools
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from finetune_doctor.capture import CapturedContext, capture
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import Signature, load_signatures

# Reconfigure streams to UTF-8 on Windows if possible
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

console = Console(stderr=True)


def _get_safe_symbol(symbol: str, fallback: str) -> str:
    """Return the symbol if the console's encoding supports it, else the fallback."""
    encoding = getattr(console, "encoding", "utf-8") or "utf-8"
    try:
        symbol.encode(encoding)
        return symbol
    except Exception:
        return fallback


# ── Pretty output ───────────────────────────────────────────────────────────


def _render_diagnosis(sig: Signature, ctx: CapturedContext) -> None:
    """Print a rich-formatted diagnosis panel to stderr."""

    # Build the panel body
    lines = []

    # Header: signature name
    warning_sym = _get_safe_symbol("⚠", "WARNING:")
    lines.append(f"[bold bright_red]{warning_sym}  {sig.name}[/bold bright_red]")
    lines.append("")

    # What went wrong
    lines.append("[bold cyan]What happened:[/bold cyan]")
    lines.append(f"  {sig.explanation}")
    lines.append("")

    # How to fix it
    lines.append("[bold green]How to fix it:[/bold green]")
    for fix_line in sig.fix.splitlines():
        stripped = fix_line.strip()
        if stripped:
            lines.append(f"  {stripped}")
    lines.append("")

    # Original exception for reference
    lines.append("[dim]Original exception:[/dim]")
    lines.append(f"  [dim]{ctx.exception_type}: {ctx.exception_message}[/dim]")

    body = "\n".join(lines)

    doc_sym = _get_safe_symbol("🩺", "[finetune-doctor]")

    console.print()
    console.print(
        Panel(
            body,
            title=f"[bold yellow]{doc_sym} finetune-doctor diagnosis[/bold yellow]",
            border_style="bright_yellow",
            padding=(1, 2),
            expand=True,
        )
    )
    console.print()


def _render_no_match(ctx: CapturedContext) -> None:
    """When no signature matches, print the raw traceback + a help note."""
    console.print()
    console.print(ctx.traceback_text, highlight=False)
    console.print(
        "[dim italic]finetune-doctor: no known failure signature matched this "
        "error.  If you'd like to help improve finetune-doctor, please open "
        "an issue at https://github.com/finetune-doctor/finetune-doctor/issues "
        "with this traceback.[/dim italic]"
    )
    console.print()


# ── Core diagnosis pipeline ────────────────────────────────────────────────


def _diagnose(
    exc: BaseException,
    dataset_format_hint: Optional[str] = None,
) -> None:
    """Run the full capture → match → render pipeline for an exception."""
    ctx = capture(exc)
    if dataset_format_hint is not None:
        ctx.dataset_format_hint = dataset_format_hint
    signatures = load_signatures()
    matched = match(ctx, signatures)

    if matched is not None:
        _render_diagnosis(matched, ctx)
    else:
        _render_no_match(ctx)


# ── Public API: context manager ─────────────────────────────────────────────


class watch:
    """Context manager AND decorator that wraps training code with diagnostics.

    Can be used as either::

        with watch():
            trainer.train()

    or with optional hints::

        with watch(dataset_format_hint="chatml"):
            trainer.train()

    Parameters
    ----------
    dataset_format_hint : str, optional
        Hint about the dataset format in use (e.g. ``"chatml"``,
        ``"alpaca"``, ``"completion"``).  Enables signatures like
        ``chatml_format_oom`` to provide more targeted diagnosis.
    """

    def __init__(
        self,
        *,
        dataset_format_hint: Optional[str] = None,
    ) -> None:
        self._dataset_format_hint = dataset_format_hint

    def __enter__(self) -> "watch":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        if exc_val is not None:
            _diagnose(exc_val, dataset_format_hint=self._dataset_format_hint)
        # Always return False — never swallow the exception
        return False

    def __call__(self, func: Callable) -> Callable:
        """Allow watch() to be used as a decorator."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper

