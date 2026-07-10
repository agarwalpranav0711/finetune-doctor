"""
test_matcher.py — Verifies matching logic against mocked CapturedContext
objects for both matching and non-matching cases, plus specificity-based
precedence.
"""

from __future__ import annotations

from finetune_doctor.capture import CapturedContext
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import Signature


def _make_oom_eval_signature() -> Signature:
    """Create a signature mimicking oom_during_eval.yaml."""
    return Signature(
        id="oom_during_eval",
        name="Out of memory during evaluation, not training",
        specificity=80,
        match={
            "exception_type": "OutOfMemoryError",
            "traceback_contains_any": [
                "evaluation_loop",
                "prediction_step",
                "eval_loop",
            ],
        },
        explanation="Training fits but eval OOMs.",
        fix="Lower eval batch size.",
    )


def _make_unrelated_signature() -> Signature:
    """A signature that should NOT match an OOM scenario."""
    return Signature(
        id="dtype_mismatch",
        name="Dtype mismatch in model forward pass",
        specificity=65,
        match={
            "exception_type": "RuntimeError",
            "traceback_contains_any": ["expected scalar type Float but found Half"],
        },
        explanation="Dtype mismatch.",
        fix="Cast to the correct dtype.",
    )


class TestMatcherPositive:
    """Tests where a signature SHOULD match."""

    def test_exact_match_oom_during_eval(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            exception_message="CUDA out of memory.",
            traceback_text=(
                'File "trainer.py", line 42, in evaluation_loop\n'
                "  prediction_step(model, batch)\n"
                "OutOfMemoryError: CUDA out of memory."
            ),
        )
        sigs = [_make_oom_eval_signature()]
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "oom_during_eval"

    def test_matches_on_prediction_step_alone(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text='File "x.py", in prediction_step\n  OutOfMemoryError',
        )
        sigs = [_make_oom_eval_signature()]
        result = match(ctx, sigs)
        assert result is not None
        assert result.id == "oom_during_eval"

    def test_matches_on_eval_loop(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text='File "y.py", in eval_loop\n  OutOfMemoryError',
        )
        sigs = [_make_oom_eval_signature()]
        assert match(ctx, sigs) is not None

    def test_first_matching_signature_wins(self):
        """If multiple signatures could match, the first one wins."""
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="evaluation_loop -> OutOfMemoryError",
        )
        sig1 = _make_oom_eval_signature()
        sig2 = Signature(
            id="oom_generic",
            name="Generic OOM",
            specificity=20,
            match={"exception_type": "OutOfMemoryError"},
            explanation="OOM.",
            fix="Free memory.",
        )
        result = match(ctx, [sig1, sig2])
        assert result.id == "oom_during_eval"


class TestMatcherNegative:
    """Tests where NO signature should match."""

    def test_no_match_wrong_exception_type(self):
        ctx = CapturedContext(
            exception_type="ValueError",
            traceback_text="evaluation_loop -> ValueError",
        )
        sigs = [_make_oom_eval_signature()]
        assert match(ctx, sigs) is None

    def test_no_match_wrong_traceback(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="in forward\n  in backward\n  OutOfMemoryError",
        )
        sigs = [_make_oom_eval_signature()]
        # Traceback doesn't contain evaluation_loop, prediction_step, or eval_loop
        assert match(ctx, sigs) is None

    def test_no_match_empty_signatures(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="evaluation_loop",
        )
        assert match(ctx, []) is None

    def test_no_match_unrelated_signature(self):
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="in evaluation_loop",
        )
        sigs = [_make_unrelated_signature()]
        assert match(ctx, sigs) is None

    def test_no_match_empty_context(self):
        ctx = CapturedContext()
        sigs = [_make_oom_eval_signature()]
        # Empty exception_type won't contain "OutOfMemoryError"
        assert match(ctx, sigs) is None


class TestSpecificityPrecedence:
    """Verify that specificity ordering drives which signature wins,
    regardless of the order signatures are passed in."""

    def test_high_specificity_wins_over_low_even_when_loaded_second(self):
        """The critical precedence test: a generic OOM (specificity 20) is
        loaded FIRST, a specific eval-OOM (specificity 80) is loaded SECOND.
        The specific one should still win because it has higher specificity.

        This test would FAIL if the matcher used list order instead of
        specificity-based sorting."""
        generic = Signature(
            id="oom_generic",
            name="Generic OOM",
            specificity=20,
            match={"exception_type": "OutOfMemoryError"},
            explanation="Generic OOM.",
            fix="Reduce batch size.",
        )
        specific = Signature(
            id="oom_during_eval",
            name="OOM during eval",
            specificity=80,
            match={
                "exception_type": "OutOfMemoryError",
                "traceback_contains_any": ["evaluation_loop"],
            },
            explanation="Eval OOM.",
            fix="Lower eval batch size.",
        )

        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="in evaluation_loop\n  OutOfMemoryError",
        )

        # Load order: generic FIRST, specific SECOND
        result = match(ctx, [generic, specific])
        assert result is not None
        assert result.id == "oom_during_eval", (
            "Higher-specificity signature should win regardless of load order"
        )

    def test_low_specificity_wins_when_high_specificity_doesnt_match(self):
        """If the high-specificity signature doesn't match the context,
        the low-specificity one should still be returned."""
        generic = Signature(
            id="oom_generic",
            name="Generic OOM",
            specificity=20,
            match={"exception_type": "OutOfMemoryError"},
            explanation="Generic OOM.",
            fix="Reduce batch size.",
        )
        specific = Signature(
            id="oom_during_eval",
            name="OOM during eval",
            specificity=80,
            match={
                "exception_type": "OutOfMemoryError",
                "traceback_contains_any": ["evaluation_loop"],
            },
            explanation="Eval OOM.",
            fix="Lower eval batch size.",
        )

        # Context with OOM but NOT in evaluation_loop — only generic matches
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="in forward\n  OutOfMemoryError",
        )

        result = match(ctx, [specific, generic])
        assert result is not None
        assert result.id == "oom_generic"

    def test_three_way_precedence(self):
        """Three signatures at different specificity levels, all match —
        the highest should win."""
        low = Signature(
            id="low", name="Low", specificity=10,
            match={"exception_type": "OutOfMemoryError"},
            explanation="L.", fix="L.",
        )
        mid = Signature(
            id="mid", name="Mid", specificity=50,
            match={"exception_type": "OutOfMemoryError"},
            explanation="M.", fix="M.",
        )
        high = Signature(
            id="high", name="High", specificity=90,
            match={"exception_type": "OutOfMemoryError"},
            explanation="H.", fix="H.",
        )

        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="something",
        )

        # Load order is deliberately scrambled
        result = match(ctx, [mid, low, high])
        assert result.id == "high"

    def test_deterministic_tiebreaker(self):
        """Create two signatures with the same specificity but different IDs.
        Verify that the alphabetically first ID always wins regardless of load order.
        """
        sig_a = Signature(
            id="a_first_id",
            name="A First",
            specificity=50,
            match={"exception_type": "OutOfMemoryError"},
            explanation="A.",
            fix="A.",
        )
        sig_b = Signature(
            id="b_second_id",
            name="B Second",
            specificity=50,
            match={"exception_type": "OutOfMemoryError"},
            explanation="B.",
            fix="B.",
        )

        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="something",
        )

        # Order 1: sig_a, then sig_b
        result1 = match(ctx, [sig_a, sig_b])
        assert result1.id == "a_first_id"

        # Order 2: sig_b, then sig_a
        result2 = match(ctx, [sig_b, sig_a])
        assert result2.id == "a_first_id"


class TestAndMatching:
    """Tests for the traceback_contains_all (AND matching) logic."""

    def test_all_present_passes(self):
        sig = Signature(
            id="and_sig",
            name="And Signature",
            specificity=50,
            match={
                "exception_type": "RuntimeError",
                "traceback_contains_all": ["mismatch", "lora_A"],
            },
            explanation="Explanation",
            fix="Fix",
        )
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text="size mismatch for base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        )
        assert match(ctx, [sig]) is not None

    def test_one_missing_fails(self):
        sig = Signature(
            id="and_sig",
            name="And Signature",
            specificity=50,
            match={
                "exception_type": "RuntimeError",
                "traceback_contains_all": ["mismatch", "lora_A"],
            },
            explanation="Explanation",
            fix="Fix",
        )
        # Missing "lora_A"
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text="size mismatch for base_model.model.layers.0.self_attn.q_proj.weight",
        )
        assert match(ctx, [sig]) is None

    def test_both_any_and_all_present_passes(self):
        sig = Signature(
            id="both_sig",
            name="Both Sig",
            specificity=50,
            match={
                "exception_type": "RuntimeError",
                "traceback_contains_any": ["lora_A", "lora_B"],
                "traceback_contains_all": ["size mismatch", "PeftModel"],
            },
            explanation="Explanation",
            fix="Fix",
        )
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text="RuntimeError: Error(s) in loading state_dict for PeftModel: size mismatch for lora_A",
        )
        assert match(ctx, [sig]) is not None

    def test_both_any_and_all_one_failing(self):
        sig = Signature(
            id="both_sig",
            name="Both Sig",
            specificity=50,
            match={
                "exception_type": "RuntimeError",
                "traceback_contains_any": ["lora_A", "lora_B"],
                "traceback_contains_all": ["size mismatch", "PeftModel"],
            },
            explanation="Explanation",
            fix="Fix",
        )
        ctx = CapturedContext(
            exception_type="RuntimeError",
            traceback_text="RuntimeError: size mismatch for lora_A",
        )
        assert match(ctx, [sig]) is None


class TestExceptionTypeSeparation:
    """Verify that subclassed exceptions (like torch.cuda.OutOfMemoryError 
    subclassing RuntimeError) do not falsely trigger generic signatures."""

    def test_oom_does_not_trigger_runtime_error_signature(self):
        """Even though torch.cuda.OutOfMemoryError is a subclass of RuntimeError,
        the matcher uses exact class name substring matching (type(exc).__name__).
        Therefore, an OOM should not trigger a signature looking for 'RuntimeError'.
        """
        sig = Signature(
            id="runtime_error_sig",
            name="Runtime Error Sig",
            specificity=50,
            match={
                "exception_type": "RuntimeError",
                "traceback_contains_any": ["some common string"],
            },
            explanation="Explanation",
            fix="Fix",
        )
        ctx = CapturedContext(
            exception_type="OutOfMemoryError",
            traceback_text="some common string in an OOM",
        )
        
        # substring "RuntimeError" is not in "OutOfMemoryError"
        assert match(ctx, [sig]) is None


