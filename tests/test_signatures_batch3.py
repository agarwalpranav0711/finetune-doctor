"""
test_signatures_batch3.py — Tests for the 5 new signatures added in chunk 3.

Each signature has exactly one positive test and one negative test.
"""

from __future__ import annotations

from finetune_doctor.capture import CapturedContext
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import load_signatures


def _load_all():
    """Load all bundled signatures."""
    return load_signatures()


# ═══════════════════════════════════════════════════════════════════════════
# 1. optimizer_none_config_crash
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimizerNoneConfigCrash:
    """Tests for the optimizer_none_config_crash signature."""

    def test_matches_none_type_in_optimizer_step(self):
        """A TypeError mentioning NoneType in optimizer code should match."""
        ctx = CapturedContext(
            exception_type="TypeError",
            exception_message=(
                "unsupported operand type(s) for *: 'float' and 'NoneType'"
            ),
            traceback_text=(
                'File "torch/optim/adam.py", line 112, in step\n'
                "  exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)\n"
                "TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "optimizer_none_config_crash"

    def test_does_not_match_type_error_without_nonetype(self):
        """A TypeError about something else (no NoneType) should NOT match."""
        ctx = CapturedContext(
            exception_type="TypeError",
            traceback_text=(
                "TypeError: expected str, got int\n"
                "in some_function"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "optimizer_none_config_crash"


# ═══════════════════════════════════════════════════════════════════════════
# 2. tokenizer_embedding_mismatch
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenizerEmbeddingMismatch:
    """Tests for the tokenizer_embedding_mismatch signature."""

    def test_matches_index_out_of_range_in_embedding(self):
        """RuntimeError with 'index out of range in self' from Embedding should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message="index out of range in self",
            traceback_text=(
                'File "torch/nn/modules/sparse.py", line 162, in forward\n'
                "  return F.embedding(input, self.weight)\n"
                "RuntimeError: index out of range in self"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "tokenizer_embedding_mismatch"

    def test_does_not_match_runtime_error_about_indexing(self):
        """A RuntimeError about tensor indexing (no 'index out of range in self'
        or 'Embedding') should NOT match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: index 42 is out of bounds for dimension 0 "
                "with size 10\n"
                "in some_tensor_op"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "tokenizer_embedding_mismatch"


# ═══════════════════════════════════════════════════════════════════════════
# 3. version_incompatibility_import_error
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionIncompatibilityImportError:
    """Tests for the version_incompatibility_import_error signature."""

    def test_matches_bitsandbytes_cuda_import_error(self):
        """ImportError mentioning bitsandbytes should match."""
        ctx = CapturedContext(
            exception_type="ImportError",
            exception_message=(
                "libbitsandbytes_cuda118.so: cannot open shared object file"
            ),
            traceback_text=(
                'File "bitsandbytes/__init__.py", line 5\n'
                "ImportError: libbitsandbytes_cuda118.so: cannot open shared object file"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "version_incompatibility_import_error"

    def test_does_not_match_unrelated_import_error(self):
        """An ImportError for a completely unrelated module should NOT match."""
        ctx = CapturedContext(
            exception_type="ImportError",
            traceback_text=(
                "ImportError: No module named 'my_custom_module'"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "version_incompatibility_import_error"


# ═══════════════════════════════════════════════════════════════════════════
# 4. multi_gpu_backward_oom
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiGpuBackwardOom:
    """Tests for the multi_gpu_backward_oom signature."""

    def test_matches_oom_in_backward_pass(self):
        """OOM with 'backward' in traceback should match."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            exception_message="CUDA out of memory. Tried to allocate 8.00 GiB.",
            traceback_text=(
                'File "torch/autograd/__init__.py", line 200, in backward\n'
                "  Variable._execution_engine.run_backward(\n"
                "OutOfMemoryError: CUDA out of memory."
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "multi_gpu_backward_oom"

    def test_does_not_match_oom_in_eval(self):
        """OOM during eval should match oom_during_eval, NOT backward OOM."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text=(
                "in evaluation_loop\n"
                "OutOfMemoryError: CUDA out of memory."
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "oom_during_eval", (
            "OOM in eval should match eval-specific, not backward OOM"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. flash_attention_build_failure
# ═══════════════════════════════════════════════════════════════════════════


class TestFlashAttentionBuildFailure:
    """Tests for the flash_attention_build_failure signature."""

    def test_matches_flash_attn_import_error(self):
        """ImportError mentioning flash_attn should match."""
        ctx = CapturedContext(
            exception_type="ImportError",
            exception_message=(
                "cannot import name 'flash_attn_func' from 'flash_attn'"
            ),
            traceback_text=(
                'File "transformers/models/llama/modeling_llama.py", line 20\n'
                "  from flash_attn import flash_attn_func\n"
                "ImportError: cannot import name 'flash_attn_func' from 'flash_attn'"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "flash_attention_build_failure"

    def test_does_not_match_unrelated_import_error(self):
        """An ImportError not mentioning flash_attn/CUDA/bitsandbytes should NOT match."""
        ctx = CapturedContext(
            exception_type="ImportError",
            traceback_text=(
                "ImportError: No module named 'pandas'"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "flash_attention_build_failure"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-signature: all 11 signatures load correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestAllSignaturesLoadBatch3:
    """Verify all 11 signatures load and validate correctly."""

    def test_all_signatures_load(self):
        sigs = _load_all()
        ids = {s.id for s in sigs}
        expected = {
            "oom_during_eval",
            "generic_cuda_oom",
            "chatml_format_oom",
            "lora_target_module_mismatch",
            "dtype_mismatch",
            "ddp_unused_parameters",
            "optimizer_none_config_crash",
            "tokenizer_embedding_mismatch",
            "version_incompatibility_import_error",
            "multi_gpu_backward_oom",
            "flash_attention_build_failure",
        }
        assert ids == expected, f"Missing: {expected - ids}, Extra: {ids - expected}"

    def test_signatures_sorted_by_specificity_descending(self):
        sigs = _load_all()
        specificities = [s.specificity for s in sigs]
        # Sorted descending (ties broken by id alphabetically)
        for i in range(len(specificities) - 1):
            assert specificities[i] >= specificities[i + 1], (
                f"Signature {sigs[i].id} (spec={specificities[i]}) should be "
                f">= {sigs[i+1].id} (spec={specificities[i+1]})"
            )
