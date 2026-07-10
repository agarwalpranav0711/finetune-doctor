"""
example_training_script.py — Simulates an OOM-during-eval failure.

This script does NOT require a real GPU or any model download. It deliberately
raises an OutOfMemoryError from inside a function named ``evaluation_loop()``
so that the traceback naturally contains that string — exactly what the
oom_during_eval signature looks for.

Run it directly to see the raw crash:
    python examples/example_training_script.py

Run it through finetune-doctor to see the diagnosis:
    finetune-doctor run -- python examples/example_training_script.py
"""

from __future__ import annotations

import sys


class OutOfMemoryError(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError when torch isn't installed.

    In a real training run, this would be raised by PyTorch's CUDA allocator.
    We define it here so the example works without torch installed, but the
    exception class name matches what the signature expects.
    """
    pass


# ── Simulated Hugging Face Trainer internals ────────────────────────────────
# These function names mirror the real HF Trainer call stack so the traceback
# contains the strings our signature looks for.


def prediction_step(model, inputs):
    """Simulates transformers.Trainer.prediction_step — the inner eval call."""
    # Simulate processing a batch that's too large for GPU memory
    raise OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 2.50 GiB. GPU 0 has a total "
        "capacity of 7.79 GiB of which 1.22 GiB is free. Including "
        "non-PyTorch memory, this process has 6.44 GiB memory in use. Of "
        "the allocated memory 5.80 GiB is allocated by PyTorch, and 412.00 "
        "MiB is reserved by PyTorch but unallocated."
    )


def evaluation_loop(model, eval_dataset):
    """Simulates transformers.Trainer.evaluation_loop."""
    print("Running evaluation on", len(eval_dataset), "samples...")
    for i, batch in enumerate(eval_dataset):
        prediction_step(model, batch)


def train(model, train_dataset, eval_dataset):
    """Simulates a basic training loop with periodic eval."""
    print("Starting training...")
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3 — training step (OK)")

    # After training completes, eval triggers the OOM
    print("Training complete. Starting evaluation...")
    evaluation_loop(model, eval_dataset)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Run the simulated training + eval pipeline."""
    # Fake model and datasets — these are just placeholders
    fake_model = "llama-3-8b-lora"
    fake_train_data = [{"text": f"sample_{i}"} for i in range(100)]
    fake_eval_data = [{"text": f"eval_{i}"} for i in range(50)]

    print(f"Model: {fake_model}")
    print(f"Train samples: {len(fake_train_data)}")
    print(f"Eval samples: {len(fake_eval_data)}")
    print()

    train(fake_model, fake_train_data, fake_eval_data)


if __name__ == "__main__":
    main()
