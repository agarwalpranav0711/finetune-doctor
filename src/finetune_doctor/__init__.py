"""finetune-doctor — catches LLM fine-tuning failures and explains them in plain English."""

__version__ = "0.1.0"

from finetune_doctor.watch import watch  # noqa: F401

__all__ = ["watch", "__version__"]
