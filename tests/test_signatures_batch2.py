"""
test_signatures_batch2.py — Tests for the 5 new signatures added in chunk 2.

Each signature has:
  - At least one positive test (should match)
  - At least one negative test (plausible non-matching scenario that could
    otherwise cause a false positive)
"""

from __future__ import annotations

import pytest

from finetune_doctor.capture import CapturedContext
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import load_signatures


def _load_all():
    """Load all bundled signatures."""
    return load_signatures()


# ═══════════════════════════════════════════════════════════════════════════
# 1. generic_cuda_oom
# ═══════════════════════════════════════════════════════════════════════════


class TestGenericCudaOOM:
    """Tests for the generic_cuda_oom catch-all signature."""

    def test_matches_plain_oom_without_eval_context(self):
        """An OOM in the training forward pass (no eval strings) should
        match the generic OOM, not the eval-specific one."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            exception_message="CUDA out of memory. Tried to allocate 4.00 GiB.",
            traceback_text=(
                'File "trainer.py", line 100, in training_step\n'
                "  outputs = model(**inputs)\n"
                "OutOfMemoryError: CUDA out of memory."
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "generic_cuda_oom", (
            "Plain OOM without eval strings should match generic, not eval-specific"
        )

    def test_does_not_match_non_oom_error(self):
        """A RuntimeError that isn't OOM should NOT match this signature."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message="Expected all tensors to be on the same device",
            traceback_text="RuntimeError: Expected all tensors to be on the same device",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        # Should either not match or match something else — but NOT generic_cuda_oom
        if result is not None:
            assert result.id != "generic_cuda_oom"

    def test_oom_during_eval_beats_generic_oom(self):
        """When an OOM happens in evaluation_loop, the eval-specific
        signature (specificity 80) should win over generic (specificity 20)."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="in evaluation_loop\n  OutOfMemoryError",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "oom_during_eval", (
            "eval-specific OOM should win over generic OOM due to higher specificity"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. chatml_format_oom
# ═══════════════════════════════════════════════════════════════════════════


class TestChatmlFormatOOM:
    """Tests for the chatml_format_oom signature (requires hint)."""

    def test_matches_when_chatml_hint_provided(self):
        """OOM + dataset_format_hint='chatml' should match."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            exception_message="CUDA out of memory.",
            traceback_text="OutOfMemoryError: CUDA out of memory.",
            dataset_format_hint="chatml",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "chatml_format_oom"

    def test_does_not_match_without_hint(self):
        """OOM without any hint should NOT match chatml_format_oom —
        it should fall through to generic_cuda_oom instead."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="OutOfMemoryError: CUDA out of memory.",
            # No dataset_format_hint set
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id != "chatml_format_oom", (
            "Should not match chatml_format_oom without the hint"
        )

    def test_does_not_match_with_wrong_hint(self):
        """OOM with a different format hint should NOT match."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="OutOfMemoryError",
            dataset_format_hint="alpaca",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "chatml_format_oom"

    def test_does_not_match_non_oom_with_chatml_hint(self):
        """A non-OOM error with chatml hint should NOT match this."""
        ctx = CapturedContext(
            exception_type="ValueError",
            traceback_text="ValueError: invalid format",
            dataset_format_hint="chatml",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "chatml_format_oom"


# ═══════════════════════════════════════════════════════════════════════════
# 3. lora_target_module_mismatch
# ═══════════════════════════════════════════════════════════════════════════


class TestLoraTargetModuleMismatch:
    """Tests for the lora_target_module_mismatch signature."""

    def test_matches_size_mismatch_in_lora(self):
        """A size mismatch mentioning lora_A should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message=(
                "size mismatch for base_model.model.layers.0.self_attn."
                "q_proj.lora_A.default.weight: copying a param with shape "
                "torch.Size([16, 4096]) from checkpoint, the shape in "
                "current model is torch.Size([16, 5120])."
            ),
            traceback_text=(
                'File "peft/peft_model.py", line 200, in load_adapter\n'
                "  size mismatch for base_model.model.layers.0.self_attn."
                "q_proj.lora_A.default.weight\n"
                "RuntimeError: size mismatch"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "lora_target_module_mismatch"

    def test_matches_target_modules_in_traceback(self):
        """Error mentioning size mismatch on lora_B should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: Error(s) in loading state_dict for PeftModel:\n"
                "  size mismatch for base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "lora_target_module_mismatch"

    def test_does_not_match_generic_runtime_error(self):
        """A RuntimeError about something else should NOT match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: CUDA error: device-side assert triggered\n"
                "in some_cuda_kernel"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "lora_target_module_mismatch"

    def test_does_not_match_oom_with_size_in_message(self):
        """An OOM error that happens to mention 'size' should NOT match
        as a LoRA mismatch — the exception type is different."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="CUDA out of memory. Tried to allocate 2.50 GiB.",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "lora_target_module_mismatch"

    def test_does_not_match_unrelated_shape_mismatch(self):
        """A RuntimeError about shape/size mismatch in a dataloader or generic tensor op
        should not match because it lacks lora_A/lora_B/PeftModel identifiers.
        """
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message="RuntimeError: stack expects each tensor to be equal size, but got [3, 224, 224] and [3, 256, 256]",
            traceback_text=(
                'File "torch/utils/data/_utils/collate.py", line 84, in default_collate\n'
                "  return col_impl(batch)\n"
                "RuntimeError: stack expects each tensor to be equal size, but got [3, 224, 224] and [3, 256, 256]"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "lora_target_module_mismatch"

    def test_lora_collision_with_dtype_mismatch(self):
        """A dtype mismatch occurring inside a LoRA-wrapped call stack should match
        dtype_mismatch (specificity 65) instead of lora_target_module_mismatch (specificity 75).
        """
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                'File "peft/tuners/lora/model.py", line 123, in forward\n'
                "  return lora_A(x)\n"
                'File "torch/nn/modules/linear.py", in forward\n'
                "  return F.linear(input, self.weight, self.bias)\n"
                "RuntimeError: expected mat1 and mat2 to have the same dtype, but got Float and Half"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "dtype_mismatch", (
            "Dtype mismatch inside LoRA should match dtype_mismatch signature, not LoRA mismatch"
        )




# ═══════════════════════════════════════════════════════════════════════════
# 4. dtype_mismatch
# ═══════════════════════════════════════════════════════════════════════════


class TestDtypeMismatch:
    """Tests for the dtype_mismatch signature."""

    def test_matches_mat1_mat2_dtype_error(self):
        """Classic PyTorch dtype mismatch error should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message=(
                "expected mat1 and mat2 to have the same dtype, "
                "but got Float and Half"
            ),
            traceback_text=(
                'File "torch/nn/modules/linear.py", in forward\n'
                "  return F.linear(input, self.weight, self.bias)\n"
                "RuntimeError: expected mat1 and mat2 to have the same dtype, "
                "but got Float and Half"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "dtype_mismatch"

    def test_matches_expected_scalar_type_error(self):
        """The 'expected scalar type X but found Y' variant should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: expected scalar type BFloat16 but found Float"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "dtype_mismatch"

    def test_does_not_match_value_error(self):
        """A ValueError mentioning dtype should NOT match (wrong exception type)."""
        ctx = CapturedContext(
            exception_type="ValueError",
            traceback_text="ValueError: Unsupported dtype float8",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "dtype_mismatch"

    def test_does_not_match_runtime_error_without_dtype_text(self):
        """A RuntimeError about shapes (not dtypes) should NOT match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: mat1 and mat2 shapes cannot be multiplied "
                "(4096x4096 and 5120x4096)"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "dtype_mismatch"


# ═══════════════════════════════════════════════════════════════════════════
# 5. ddp_unused_parameters
# ═══════════════════════════════════════════════════════════════════════════


class TestDDPUnusedParameters:
    """Tests for the ddp_unused_parameters signature."""

    def test_matches_find_unused_parameters_error(self):
        """The classic DDP unused parameters error should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            exception_message=(
                "It looks like your module has parameters that were not "
                "used in producing its output. You can enable unused parameter "
                "detection by passing the keyword argument "
                "`find_unused_parameters=True`"
            ),
            traceback_text=(
                'File "torch/nn/parallel/distributed.py", line 1400\n'
                "RuntimeError: It looks like your module has parameters that "
                "were not used in producing its output. You can enable unused "
                "parameter detection by passing the keyword argument "
                "`find_unused_parameters=True`"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "ddp_unused_parameters"

    def test_matches_expected_finished_reduction(self):
        """The 'Expected to have finished reduction' variant should match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: Expected to have finished reduction in the "
                "prior iteration before starting a new one."
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "ddp_unused_parameters"

    def test_does_not_match_non_ddp_runtime_error(self):
        """A RuntimeError not related to DDP should NOT match."""
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text=(
                "RuntimeError: NCCL communicator was aborted. "
                "Original error: Connection timed out"
            ),
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "ddp_unused_parameters"

    def test_does_not_match_value_error_with_parameter_text(self):
        """A ValueError that happens to mention 'parameters' should not match."""
        ctx = CapturedContext(
            exception_type="ValueError",
            traceback_text="ValueError: Invalid parameters for optimizer",
        )
        sigs = _load_all()
        result = match(ctx, sigs)
        if result is not None:
            assert result.id != "ddp_unused_parameters"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-signature: all 6 signatures load correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestAllSignaturesLoad:
    """Verify batch 2 signatures load and validate correctly."""

    def test_all_signatures_load(self):
        sigs = _load_all()
        ids = {s.id for s in sigs}
        # Batch 2 signatures must be present (batch 3+ may add more)
        batch2_expected = {
            "oom_during_eval",
            "generic_cuda_oom",
            "chatml_format_oom",
            "lora_target_module_mismatch",
            "dtype_mismatch",
            "ddp_unused_parameters",
        }
        assert batch2_expected.issubset(ids), (
            f"Missing batch 2 signatures: {batch2_expected - ids}"
        )

    def test_signatures_sorted_by_specificity_descending(self):
        sigs = _load_all()
        specificities = [s.specificity for s in sigs]
        assert specificities == sorted(specificities, reverse=True), (
            "Signatures should be sorted by specificity descending"
        )

    def test_all_signatures_have_required_fields(self):
        sigs = _load_all()
        for sig in sigs:
            assert sig.id, f"Signature missing id"
            assert sig.name, f"Signature {sig.id} missing name"
            assert sig.specificity >= 0, f"Signature {sig.id} has invalid specificity"
            assert sig.explanation, f"Signature {sig.id} missing explanation"
            assert sig.fix, f"Signature {sig.id} missing fix"
