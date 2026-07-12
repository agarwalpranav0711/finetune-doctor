# 🩺 finetune-doctor

**Catches LLM fine-tuning failures the moment they happen, explains *why* in plain English, and tells you exactly how to fix them.**

---

## The Problem: Before vs. After

### ❌ Before (raw stack trace — good luck figuring this out)

```
Traceback (most recent call last):
  File "train.py", line 87, in <module>
    trainer.train()
  File "/site-packages/transformers/trainer.py", line 1859, in train
    return inner_training_loop(...)
  File "/site-packages/transformers/trainer.py", line 2203, in _inner_training_loop
    self._maybe_log_save_evaluate(...)
  File "/site-packages/transformers/trainer.py", line 2571, in _maybe_log_save_evaluate
    metrics = self.evaluate()
  File "/site-packages/transformers/trainer.py", line 3524, in evaluate
    output = self.evaluation_loop(...)
  File "/site-packages/transformers/trainer.py", line 3710, in evaluation_loop
    logits = self.prediction_step(model, inputs)
  File "/site-packages/transformers/trainer.py", line 3891, in prediction_step
    outputs = model(**inputs)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB.
GPU 0 has a total capacity of 7.79 GiB of which 1.22 GiB is free.
```

You stare at this. Training was *working*. Why did it suddenly OOM? You start searching GitHub issues...

### ✅ After (with finetune-doctor)

```
╭────────────────── 🩺 finetune-doctor diagnosis ──────────────────╮
│                                                                   │
│  ⚠  Out of memory during evaluation, not training                │
│                                                                   │
│  What happened:                                                   │
│    Training itself fits in memory, but evaluation is using a      │
│    larger effective batch size or longer sequences than training,  │
│    causing a separate out-of-memory crash during the eval step.   │
│                                                                   │
│  How to fix it:                                                   │
│    1. Set a smaller per_device_eval_batch_size — often half of    │
│       your training batch size is enough.                         │
│    2. If your eval dataset contains sequences longer than your    │
│       training max_seq_length, truncate them or set max_length    │
│       explicitly in the evaluation config.                        │
│    3. Disable evaluation during training entirely                 │
│       (evaluation_strategy="no") and run evaluation as a          │
│       separate pass afterward.                                    │
│    4. Add torch.cuda.empty_cache() in a custom callback before    │
│       each evaluation step to reclaim fragmented memory.          │
│                                                                   │
│  Original exception:                                              │
│    OutOfMemoryError: CUDA out of memory. Tried to allocate...    │
│                                                                   │
╰───────────────────────────────────────────────────────────────────╯
```

30 seconds instead of 30 minutes.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/finetune-doctor/finetune-doctor.git
cd finetune-doctor

# Install in editable mode (not on PyPI yet)
pip install -e ".[dev]"
```

---

## Usage

### Option 1: Python context manager (recommended)

Add two lines to your existing training script:

```python
from finetune_doctor import watch

with watch():
    # your existing training code — unchanged
    trainer.train()
```

That's it. If training succeeds, `watch()` does nothing. If it fails, you get a diagnosis panel instead of a cryptic traceback.

### Option 2: Decorator

```python
from finetune_doctor import watch

@watch()
def run_training():
    trainer.train()

run_training()
```

### Option 3: CLI wrapper (no code changes needed)

Wrap your existing launch command:

```bash
finetune-doctor run -- python train.py
```

---

## How Signatures Work

finetune-doctor matches failures against a catalogue of **signatures** — YAML files that encode known failure patterns. Each signature looks like this:

```yaml
id: oom_during_eval
name: "Out of memory during evaluation, not training"
specificity: 80   # higher = more specific = checked first

match:
  exception_type: "OutOfMemoryError"
  traceback_contains_any:
    - "evaluation_loop"
    - "prediction_step"
    - "eval_loop"

explanation: >
  Training itself fits in memory, but evaluation is using a larger
  effective batch size or longer sequences, causing a separate OOM
  crash during the eval step.

fix: >
  Set a smaller per_device_eval_batch_size, truncate eval sequences,
  or disable eval during training.
```

### Signature precedence (specificity)

When multiple signatures could match the same failure, **specificity** determines which one wins. Higher specificity = more specific = checked first. This ensures that a targeted diagnosis (like "OOM during eval") always beats a generic catch-all (like "CUDA out of memory") regardless of file load order.

Currently implemented signatures:

| Signature | Specificity | Matches |
|-----------|-------------|---------|
| `oom_during_eval` | 80 | OOM with eval-related traceback |
| `lora_target_module_mismatch` | 75 | Shape/size mismatch in LoRA layers |
| `chatml_format_oom` | 70 | OOM with ChatML dataset hint |
| `tokenizer_embedding_mismatch` | 70 | Vocab size exceeds model embeddings |
| `ddp_unused_parameters` | 70 | DDP unused parameter errors |
| `dtype_mismatch` | 65 | Precision conflicts (bf16/fp16/fp32) |
| `optimizer_none_config_crash` | 60 | Optimizer received None for config |
| `version_incompatibility_import_error` | 55 | CUDA/library version incompatibility |
| `flash_attention_build_failure` | 50 | Flash Attention import/compile error |
| `multi_gpu_backward_oom` | 45 | OOM during backward pass |
| `generic_cuda_oom` | 20 | Any CUDA OOM (catch-all) |

### The `dataset_format_hint` parameter

Some signatures (like `chatml_format_oom`) need context about your dataset format that can't be inferred from the traceback alone. You can provide this via the `dataset_format_hint` parameter:

```python
from finetune_doctor import watch

# If using ChatML/messages-format data:
with watch(dataset_format_hint="chatml"):
    trainer.train()
```

> **Known limitation:** dataset format is not auto-detected yet — you must pass the hint explicitly. Auto-detection is planned for a future chunk.

### Contributing new signatures

Found a training failure you keep hitting? Add a new `.yaml` file to `src/finetune_doctor/signatures/rules/` with:

1. **`id`** — unique snake_case identifier
2. **`name`** — human-readable title
3. **`specificity`** — integer (higher = more specific, checked first). Use 20-40 for catch-alls, 60-80 for specific patterns, 90+ for very narrow matches
4. **`match`** — rules to detect the failure (exception type, traceback patterns, optional hints)
5. **`explanation`** — what went wrong, in plain English
6. **`fix`** — concrete steps to resolve it

Then open a PR. No Python code changes needed — just the YAML file.

---

## Current Limitations

> **This is v0.1 — early and honest about it.**

- **11 signatures so far:** covers OOMs (generic, eval, backward-pass, ChatML), LoRA misconfigurations, dtype mismatches, tokenizer embedding size mismatch, missing optimizer configs, and flash-attention / CUDA import errors.
- **Simple matching:** the current matcher does string/substring matching on exception types and traceback text. Future versions will add regex and config-aware rules.
- **ChatML signature requires manual hint:** the `chatml_format_oom` signature only activates when you pass `dataset_format_hint="chatml"` — auto-detection is not yet implemented.
- **No LLM fallback:** there's no AI-powered diagnosis for unknown failures yet. When no signature matches, you get the raw traceback plus a pointer to open an issue.
- **CLI package versions reflect the outer environment:** when using `finetune-doctor run --`, package versions are detected from the environment where finetune-doctor is installed. If the subprocess uses a different virtualenv or conda environment, the reported versions may not match what the training script actually used. This will be addressed in a future chunk.
- **Signatures needing real-world refinement:** The `version_incompatibility_import_error` signature relies on the string "CUDA" in an `ImportError`. This is broad and may be prone to false positives; it is flagged as "may need tightening based on real-world reports."
- **Not on PyPI:** install from source with `pip install -e ".[dev]"` for now.

### How the CLI captures GPU stats

Both the `watch()` context manager and the `finetune-doctor run` CLI now capture rich GPU context. The CLI does this by running a **background polling thread** (via `pynvml`) that samples GPU memory, utilization, and temperature every 0.5 seconds while your training command runs. It tracks the **peak** (high-water mark) values observed during the entire run — so even after the subprocess crashes and its GPU memory is freed, finetune-doctor knows exactly how much memory was in use at the worst moment. If no NVIDIA GPU is present or `pynvml` is unavailable, the CLI gracefully skips GPU polling and still diagnoses based on the traceback and system info.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — see [LICENSE](./LICENSE).
