# Contributing to finetune-doctor

Welcome! `finetune-doctor` is designed to be extensible. The most valuable contributions you can make are **new failure signatures** based on real-world errors you've encountered while fine-tuning LLMs.

This guide will walk you through adding a new signature and testing it. You don't need to be an expert in the Python codebase to contribute — signatures are just YAML files!

## 1. How Signatures Work

A signature is a `.yaml` file located in `src/finetune_doctor/signatures/rules/`. It defines how to detect a specific failure and what to tell the user when it happens.

When a training script crashes, `finetune-doctor` captures the exception type and the traceback text. It then evaluates all signatures to see which ones match. The matching signature with the highest `specificity` wins.

## 2. Creating a New Signature

1. Create a new file in `src/finetune_doctor/signatures/rules/`, e.g., `my_new_error.yaml`.
2. Fill in the required fields:

```yaml
# A unique, snake_case identifier for this signature.
id: my_new_error

# A short, human-readable title that will appear at the top of the diagnosis panel.
name: "Short description of the error"

# An integer determining precedence (0-100). Higher = checked first.
# See "Choosing a Specificity" below.
specificity: 65

# The rules used to match the captured error.
match:
  # (Optional) The name of the Python exception class, e.g., "RuntimeError", "OutOfMemoryError".
  exception_type: "RuntimeError"

  # (Optional) A list of strings. The signature matches if ANY of these strings appear in the traceback.
  traceback_contains_any:
    - "specific_function_name"
    - "a unique phrase from the error message"

  # (Optional) A list of strings. The signature matches only if ALL of these strings appear.
  traceback_contains_all:
    - "must have this"
    - "and must also have this"

  # (Optional) A required hint the user must provide (e.g., "chatml").
  # dataset_format_hint: "chatml"

# A plain English explanation of what went wrong.
explanation: >
  Write a clear, beginner-friendly explanation of why this error happens.
  Use multiple lines if needed.

# Concrete steps to fix the issue.
fix: >
  1. Tell the user exactly what to change in their code or config.
  2. Provide examples if possible.
```

### Choosing a Specificity

Specificity determines which signature wins if multiple could match the same error. 
- **20-40**: Broad catch-alls (e.g., generic CUDA OOM).
- **50-70**: Standard, targeted signatures (e.g., dtype mismatch, missing configs).
- **75-90+**: Highly specific, narrow signatures that rely on exact combinations of signals (e.g., OOM specifically during the eval loop).

When picking a number, look at the existing YAML files and place yours relative to them. If your signature is very specific, give it a higher number than broader signatures that might also match.

## 3. Writing Tests

Every signature must have at least two tests:
- **One positive test**: Proving it matches a mock traceback of the error.
- **One negative test**: Proving it does *not* match a similar but unrelated error.

1. Open a test file in the `tests/` directory (e.g., `tests/test_signatures_batch3.py`, or create a new one).
2. Write tests using the `CapturedContext` helper:

```python
from finetune_doctor.capture import CapturedContext
from finetune_doctor.matcher import match
from finetune_doctor.signatures.loader import load_signatures

def test_matches_my_new_error():
    ctx = CapturedContext(
        exception_type="RuntimeError",
        traceback_text="... your mock traceback containing the trigger words ..."
    )
    sigs = load_signatures()
    result = match(ctx, sigs)
    assert result is not None
    assert result.id == "my_new_error"

def test_does_not_match_unrelated_error():
    ctx = CapturedContext(
        exception_type="RuntimeError",
        traceback_text="... a traceback that shouldn't match ..."
    )
    sigs = load_signatures()
    result = match(ctx, sigs)
    if result is not None:
        assert result.id != "my_new_error"
```

## 4. Running the Test Suite

Before opening a Pull Request, make sure all tests pass:

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest -v
```

If everything is green, you're ready to open a PR!
