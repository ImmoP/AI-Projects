"""
Qualitative comparison (Phase 9).

A single aggregate score is not enough.  This module aligns the per-query
results of both systems (by ``query_id``) and buckets each query into one of:

* ``agentic_wins_retrieval``  — agentic found the gold doc, classical did not
* ``classical_wins_retrieval`` — classical found the gold doc, agentic did not
* ``both_succeeded``          — both found the gold doc
* ``both_failed``             — neither found the gold doc

For agentic wins it also surfaces the agent's generated queries and which
retrieval call actually found the gold document, so the reader can judge
whether extra retrieval calls genuinely helped.
"""

from __future__ import annotations


def document_hit(record: dict) -> bool:
    """Document-level hit (gold doc appears in the retrieved set)."""
    return record.get("hit_at_1_doc") == 1 or record.get("hit_at_5_doc") == 1


def align_records(classical: list[dict], agentic: list[dict]) -> dict:
    """Return ``{query_id: {"classical": record, "agentic": record}}``."""
    agentic_by_id = {r["query_id"]: r for r in agentic}
    aligned = {}
    for c in classical:
        a = agentic_by_id.get(c["query_id"])
        if a is not None:
            aligned[c["query_id"]] = {"classical": c, "agentic": a}
    return aligned


def categorize(aligned: dict) -> dict:
    """Bucket aligned query pairs by retrieval outcome (document level)."""
    buckets = {
        "agentic_wins_retrieval": [],
        "classical_wins_retrieval": [],
        "both_succeeded": [],
        "both_failed": [],
    }

    for qid, pair in aligned.items():
        c_hit = document_hit(pair["classical"])
        a_hit = document_hit(pair["agentic"])
        if a_hit and not c_hit:
            buckets["agentic_wins_retrieval"].append(qid)
        elif c_hit and not a_hit:
            buckets["classical_wins_retrieval"].append(qid)
        elif c_hit and a_hit:
            buckets["both_succeeded"].append(qid)
        else:
            buckets["both_failed"].append(qid)

    return buckets


def _classical_top_sources(classical_rec: dict, k: int = 3) -> list[str]:
    return list(classical_rec.get("retrieved_sources", []))[:k]


def agentic_win_summary(aligned: dict, bucket: str) -> list[dict]:
    """Build a human-readable detail for every query in a bucket."""
    details = []
    for qid, pair in aligned.items():
        c = pair["classical"]
        a = pair["agentic"]
        if not (bucket == "agentic_wins_retrieval" and document_hit(a) and not document_hit(c)):
            continue
        details.append(
            {
                "query_id": qid,
                "question": c.get("question"),
                "classical_top_sources": _classical_top_sources(c),
                "agentic_queries": list(a.get("queries_generated", [])),
                "gold_found_at_call": a.get("gold_found_at_call"),
                "retrieval_calls": a.get("retrieval_calls"),
            }
        )
    return details