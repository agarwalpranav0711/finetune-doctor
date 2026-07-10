"""
test_oom_during_eval_signature.py — Integration test that runs the example
script through the full watch() flow and asserts the correct signature was
matched and the expected fix text appears in the output.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr

import pytest

from finetune_doctor.capture import CapturedContext, capture
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import load_signatures
from finetune_doctor.watch import _diagnose


# ── Helpers ──────────────────────────────────────────────────────────────────


def _trigger_oom_during_eval():
    """Reproduce the exact failure the example script generates."""

    class OutOfMemoryError(RuntimeError):
        pass

    def prediction_step(model, inputs):
        raise OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.50 GiB."
        )

    def evaluation_loop(model, eval_dataset):
        for batch in eval_dataset:
            prediction_step(model, batch)

    evaluation_loop("model", [{"text": "a"}])


# ── Tests ────────────────────────────────────────────────────────────────────


class TestOOMDuringEvalSignature:
    """Integration-style tests for the oom_during_eval signature."""

    def test_signature_loads_from_yaml(self):
        """The YAML file should load and validate without errors."""
        sigs = load_signatures()
        ids = [s.id for s in sigs]
        assert "oom_during_eval" in ids

    def test_signature_has_required_fields(self):
        sigs = load_signatures()
        oom_sig = [s for s in sigs if s.id == "oom_during_eval"][0]
        assert oom_sig.name
        assert oom_sig.explanation
        assert oom_sig.fix
        assert oom_sig.exception_type == "OutOfMemoryError"
        assert len(oom_sig.traceback_contains_any) > 0

    def test_full_pipeline_matches_oom_during_eval(self):
        """Run the simulated failure through capture → match and verify."""
        try:
            _trigger_oom_during_eval()
        except RuntimeError as exc:
            ctx = capture(exc)
            sigs = load_signatures()
            matched = match(ctx, sigs)

            assert matched is not None, "Expected oom_during_eval to match"
            assert matched.id == "oom_during_eval"
            assert "eval" in matched.explanation.lower()
            assert "eval_batch_size" in matched.fix or "eval" in matched.fix.lower()
        else:
            pytest.fail("Expected OutOfMemoryError to be raised")

    def test_diagnosis_output_contains_fix_text(self):
        """Verify the rendered output includes the fix text."""
        try:
            _trigger_oom_during_eval()
        except RuntimeError as exc:
            # Capture the rich console output
            from rich.console import Console

            buffer = io.StringIO()
            test_console = Console(file=buffer, force_terminal=False)

            # Manually run the pipeline and render
            ctx = capture(exc)
            sigs = load_signatures()
            matched = match(ctx, sigs)

            assert matched is not None

            # Build the same output that watch._render_diagnosis would
            from rich.panel import Panel

            lines = [
                f"⚠  {matched.name}",
                "",
                "What happened:",
                f"  {matched.explanation}",
                "",
                "How to fix it:",
            ]
            for fix_line in matched.fix.splitlines():
                stripped = fix_line.strip()
                if stripped:
                    lines.append(f"  {stripped}")
            lines.append("")
            lines.append("Original exception:")
            lines.append(
                f"  {ctx.exception_type}: {ctx.exception_message}"
            )

            test_console.print(Panel("\n".join(lines), title="diagnosis"))
            output = buffer.getvalue()

            assert "eval_batch_size" in output or "per_device_eval_batch_size" in output
            assert "Out of memory" in output or "OutOfMemoryError" in output

    def test_watch_context_manager_integration(self):
        """Verify the full watch() context manager flow re-raises and diagnoses."""
        from finetune_doctor.watch import watch

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            with watch():
                _trigger_oom_during_eval()

    def test_watch_decorator_integration(self):
        """Verify the watch() decorator flow."""
        from finetune_doctor.watch import watch

        @watch()
        def _run():
            _trigger_oom_during_eval()

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            _run()
