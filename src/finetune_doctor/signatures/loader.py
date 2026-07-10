"""
loader.py — Discovers, loads, and validates YAML signature files.

Each signature YAML is expected to have at minimum:
  id, name, specificity, match, explanation, fix
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Signature:
    """A single failure signature loaded from YAML."""

    id: str
    name: str
    specificity: int
    match: Dict[str, Any]
    explanation: str
    fix: str

    # ── Derived match helpers ────────────────────────────────────────────
    @property
    def exception_type(self) -> Optional[str]:
        return self.match.get("exception_type")

    @property
    def traceback_contains_any(self) -> List[str]:
        return self.match.get("traceback_contains_any", [])

    @property
    def traceback_contains_all(self) -> List[str]:
        return self.match.get("traceback_contains_all", [])

    @property
    def dataset_format_hint(self) -> Optional[str]:
        return self.match.get("dataset_format_hint")



_REQUIRED_FIELDS = {"id", "name", "specificity", "match", "explanation", "fix"}


def _validate_raw(data: Dict[str, Any], filepath: Path) -> None:
    """Raise with a clear message if required fields are missing."""
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Signature file {filepath} is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )
    if not isinstance(data["match"], dict):
        raise ValueError(
            f"Signature file {filepath}: 'match' must be a mapping, "
            f"got {type(data['match']).__name__}"
        )
    if not isinstance(data.get("specificity"), int):
        raise ValueError(
            f"Signature file {filepath}: 'specificity' must be an integer, "
            f"got {type(data.get('specificity')).__name__}"
        )


def _rules_directory() -> Path:
    """Return the default path to the bundled rules/ directory."""
    return Path(__file__).parent / "rules"


def load_signatures(rules_dir: Optional[Path] = None) -> List[Signature]:
    """Load and validate every .yaml file from the rules directory.

    Parameters
    ----------
    rules_dir : Path, optional
        Override the default bundled ``rules/`` directory (useful for testing).

    Returns
    -------
    list[Signature]
        Validated signature objects sorted by specificity descending
        (most specific first) for precedence-based matching.

    Raises
    ------
    ValueError
        If any YAML file is malformed or missing required fields.
    FileNotFoundError
        If the rules directory does not exist.
    """
    rules_dir = rules_dir or _rules_directory()

    if not rules_dir.is_dir():
        raise FileNotFoundError(
            f"Signature rules directory not found: {rules_dir}"
        )

    signatures: List[Signature] = []

    for yaml_file in sorted(rules_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raise ValueError(
                f"Signature file {yaml_file} is empty or contains no YAML data."
            )

        _validate_raw(raw, yaml_file)

        signatures.append(
            Signature(
                id=raw["id"],
                name=raw["name"],
                specificity=raw["specificity"],
                match=raw["match"],
                explanation=raw["explanation"].strip(),
                fix=raw["fix"].strip(),
            )
        )

    # Sort by specificity descending, then by id ascending (alphabetically)
    # to break ties deterministically.
    signatures.sort(key=lambda s: (-s.specificity, s.id))

    return signatures

