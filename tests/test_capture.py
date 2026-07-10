"""
test_capture.py — Verifies that capture.py gathers what it can and never
raises, even when torch/pynvml aren't installed or no GPU is present.
"""

from __future__ import annotations

import sys

from finetune_doctor.capture import CapturedContext, capture


class _FakeError(RuntimeError):
    """A test exception."""
    pass


def _make_exc(msg: str = "test error") -> _FakeError:
    """Create an exception with a real traceback attached."""
    try:
        raise _FakeError(msg)
    except _FakeError as e:
        return e


class TestCapture:
    """Tests for the capture() function."""

    def test_returns_captured_context(self):
        exc = _make_exc()
        ctx = capture(exc)
        assert isinstance(ctx, CapturedContext)

    def test_exception_info_populated(self):
        exc = _make_exc("something broke")
        ctx = capture(exc)
        assert ctx.exception_type == "_FakeError"
        assert "something broke" in ctx.exception_message
        assert "_FakeError" in ctx.traceback_text
        assert "something broke" in ctx.traceback_text

    def test_traceback_contains_test_function_name(self):
        exc = _make_exc()
        ctx = capture(exc)
        # The traceback should reference _make_exc
        assert "_make_exc" in ctx.traceback_text

    def test_python_version_populated(self):
        exc = _make_exc()
        ctx = capture(exc)
        assert ctx.python_version != ""
        # Should contain major.minor at least
        assert str(sys.version_info.major) in ctx.python_version

    def test_os_platform_populated(self):
        exc = _make_exc()
        ctx = capture(exc)
        assert ctx.os_platform != ""

    def test_ram_info_populated(self):
        """psutil should always be available since it's a hard dep."""
        exc = _make_exc()
        ctx = capture(exc)
        assert ctx.ram_total is not None
        assert ctx.ram_total > 0
        assert ctx.ram_available is not None
        assert ctx.ram_percent is not None

    def test_never_raises_on_any_exception_type(self):
        """capture() must never raise, no matter what exception is passed."""
        for exc_class in (RuntimeError, ValueError, TypeError, KeyError, OSError):
            exc = None
            try:
                raise exc_class("test")
            except exc_class as e:
                exc = e
            ctx = capture(exc)
            assert isinstance(ctx, CapturedContext)

    def test_cuda_fields_none_when_no_gpu(self):
        """On machines without CUDA, these should be None, not errors."""
        exc = _make_exc()
        ctx = capture(exc)
        # We can't guarantee CUDA is absent, but we can verify the fields
        # exist and are either None or valid ints
        assert ctx.cuda_memory_allocated is None or isinstance(
            ctx.cuda_memory_allocated, int
        )

    def test_package_versions_is_dict(self):
        exc = _make_exc()
        ctx = capture(exc)
        assert isinstance(ctx.package_versions, dict)
        # The captured versions should only contain packages that are
        # actually importable — we don't know which ones are installed in
        # the test environment, but the dict must never contain bad values
        for pkg, ver in ctx.package_versions.items():
            assert isinstance(pkg, str)
            assert isinstance(ver, str)
