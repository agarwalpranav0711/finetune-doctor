"""
matcher.py — Matches a CapturedContext against loaded failure signatures.

Signatures are checked in specificity order (highest first), so more specific
signatures always win over generic catch-alls regardless of load order.

Match rules checked per signature:
  1. exception_type — substring match against the raised exception's class name.
  2. traceback_contains_any — if ANY listed string appears in the traceback.
  3. dataset_format_hint — if the signature requires a specific hint, the
     captured context must have that exact hint set.
  4. Return the first matching signature (highest specificity), or None.
"""

from __future__ import annotations

from typing import List, Optional

from finetune_doctor.capture import CapturedContext
from finetune_doctor.signatures.loader import Signature


def match(
    ctx: CapturedContext,
    signatures: List[Signature],
) -> Optional[Signature]:
    """Find the first signature whose match rules fit the captured context.

    Signatures are expected to be pre-sorted by specificity descending
    (the loader does this), so the most specific matching signature is
    always returned first.

    Parameters
    ----------
    ctx : CapturedContext
        The failure snapshot gathered by ``capture.capture()``.
    signatures : list[Signature]
        The loaded signature catalogue (pre-sorted by specificity desc).

    Returns
    -------
    Signature or None
        The first matching signature, or ``None`` if nothing matches.
    """
    # Defensive re-sort in case signatures weren't pre-sorted — ensures
    # precedence always works even if someone passes an unsorted list.
    # Secondary sort by id (ascending) breaks ties deterministically:
    # if two signatures have the same specificity, the one whose id comes
    # first alphabetically wins.
    sorted_sigs = sorted(
        signatures, key=lambda s: (-s.specificity, s.id)
    )

    for sig in sorted_sigs:
        if not _exception_type_matches(ctx, sig):
            continue
        if not _traceback_matches(ctx, sig):
            continue
        if not _traceback_contains_all_matches(ctx, sig):
            continue
        if not _dataset_format_hint_matches(ctx, sig):
            continue
        return sig

    return None



# ── Private helpers ──────────────────────────────────────────────────────────


def _exception_type_matches(ctx: CapturedContext, sig: Signature) -> bool:
    """Return True if the signature's exception_type is a substring of the
    captured exception's class name.

    If the signature doesn't specify an exception_type, treat it as a
    wildcard (always matches).
    """
    if sig.exception_type is None:
        return True
    return sig.exception_type in ctx.exception_type


def _traceback_matches(ctx: CapturedContext, sig: Signature) -> bool:
    """Return True if ANY of the signature's traceback_contains_any strings
    appear anywhere in the captured traceback text.

    If the signature doesn't specify traceback_contains_any, treat it as a
    wildcard (always matches).
    """
    patterns = sig.traceback_contains_any
    if not patterns:
        return True
    tb = ctx.traceback_text
    return any(pattern in tb for pattern in patterns)


def _traceback_contains_all_matches(ctx: CapturedContext, sig: Signature) -> bool:
    """Return True if ALL of the signature's traceback_contains_all strings
    appear anywhere in the captured traceback text.

    If the signature doesn't specify traceback_contains_all, treat it as a
    wildcard (always matches).
    """
    patterns = sig.traceback_contains_all
    if not patterns:
        return True
    tb = ctx.traceback_text
    return all(pattern in tb for pattern in patterns)


def _dataset_format_hint_matches(ctx: CapturedContext, sig: Signature) -> bool:
    """Return True if the signature's dataset_format_hint matches the
    captured context's hint.

    If the signature doesn't require a dataset_format_hint, treat it as a
    wildcard (always matches).  If the signature DOES require one but the
    context doesn't have one set, the signature does NOT match — this
    prevents false positives when the user hasn't provided the hint.
    """
    required_hint = sig.dataset_format_hint
    if required_hint is None:
        return True
    return ctx.dataset_format_hint == required_hint
