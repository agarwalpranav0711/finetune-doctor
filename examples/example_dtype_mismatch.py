"""
example_dtype_mismatch.py — Simulates a dtype mismatch failure.

Mimics the real PyTorch error when model and training precision disagree.
"""

class RuntimeError(Exception):
    pass


def linear_forward(input_tensor, weight):
    """Simulates torch.nn.functional.linear — where dtype mismatch surfaces."""
    raise RuntimeError(
        "expected mat1 and mat2 to have the same dtype, but got Float and Half"
    )


def model_forward(model, inputs):
    """Simulates model(**inputs)."""
    return linear_forward(inputs, model)


def training_step(model, inputs):
    """Simulates transformers.Trainer.training_step."""
    outputs = model_forward(model, inputs)
    return outputs


def main():
    print("Loading model in bf16...")
    print("Starting training with fp16=True (mismatch!)...")
    print()

    fake_model = "llama-3-8b-qlora"
    fake_inputs = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    training_step(fake_model, fake_inputs)


if __name__ == "__main__":
    main()
