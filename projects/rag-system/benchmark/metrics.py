"""
Retrieval metrics (Phase 2).

Separate retrieval evaluation from generation evaluation.  For a single
query we compute Hit@k and MRR from the ranked list of retrieved chunks.

Two matching levels are supported:

* ``document`` — a chunk is correct if ``chunk.source == gold_doc_id``
  (does the gold document appear anywhere in the retrieved set).
* ``section``  — a chunk is correct if ``chunk.source == gold_doc_id`` AND
  ``chunk.page == gold_section_id`` (does the gold *section* of the gold
  document appear).

The section level is meaningful here because the corpus builder tags every
chunk with ``page = section_id`` (see ``open_ragbench_loader``).  Both are
reported so the two architectures can be compared fairly at both granularities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Hit@k and reciprocal rank for one query."""

    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_10: int = 0
    reciprocal_rank: float = 0.0
    # 1-based rank at which the gold evidence first appeared (0 if never).
    first_hit_rank: int = 0


def compute_metrics(
    retrieved: list,
    gold_doc_id: str,
    gold_section_id: int | None = None,
    level: str = "document",
) -> RetrievalMetrics:
    """
    Compute retrieval metrics for one query.

    Parameters
    ----------
    retrieved : list
        Ranked chunks; each must expose ``.source`` and ``.page``
        (e.g. ``RetrievalResult`` objects).
    gold_doc_id : str
        The benchmark's gold document id.
    gold_section_id : int | None
        The gold section id (required for ``level="section"``).
    level : str
        ``"document"`` or ``"section"``.
    """
    if level not in ("document", "section"):
        raise ValueError(f"Unknown level: {level!r}")

    def is_relevant(chunk) -> bool:
        if level == "section":
            return (
                chunk.source == gold_doc_id
                and chunk.page == gold_section_id
            )
        return chunk.source == gold_doc_id

    hits = [bool(is_relevant(r)) for r in retrieved]

    first_hit_rank = 0
    reciprocal_rank = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            first_hit_rank = idx
            reciprocal_rank = 1.0 / idx
            break

    return RetrievalMetrics(
        hit_at_1=int(any(hits[:1])),
        hit_at_3=int(any(hits[:3])),
        hit_at_5=int(any(hits[:5])),
        hit_at_10=int(any(hits[:10])),
        reciprocal_rank=reciprocal_rank,
        first_hit_rank=first_hit_rank,
    )


def aggregate_metrics(metrics_list: list) -> dict:
    """
    Average a list of :class:`RetrievalMetrics` into a single summary dict.

    Returns keys ``hit_at_1`` ... ``hit_at_10`` and ``mrr`` as means.
    """
    if not metrics_list:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0,
                "hit_at_10": 0.0, "mrr": 0.0}

    n = len(metrics_list)
    return {
        "hit_at_1": sum(m.hit_at_1 for m in metrics_list) / n,
        "hit_at_3": sum(m.hit_at_3 for m in metrics_list) / n,
        "hit_at_5": sum(m.hit_at_5 for m in metrics_list) / n,
        "hit_at_10": sum(m.hit_at_10 for m in metrics_list) / n,
        "mrr": sum(m.reciprocal_rank for m in metrics_list) / n,
    }