"""CLI for the 4-way RAG benchmark + unanswerable evaluation.

Usage:
  python -m benchmark.evaluate_rag --system naive --limit 5
  python -m benchmark.evaluate_rag --system hybrid --limit 5
  python -m benchmark.evaluate_rag --system reranker --limit 5
  python -m benchmark.evaluate_rag --system agentic --limit 5
  python -m benchmark.evaluate_rag --system all --corpus-docs 0 --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.embeddings import load_embedding_model
from src.generator import generate_answer
from src.retriever import retrieve

from benchmark import comparison as cmpmod
from benchmark import config
from benchmark import generation_metrics as genmetrics
from benchmark.bm25 import BM25Index
from benchmark.evaluators import (
    evaluate_hybrid,
    evaluate_naive,
    evaluate_reranker,
)
from benchmark.hybrid import hybrid_retrieve
from benchmark.metrics import RetrievalMetrics, aggregate_metrics
from benchmark.open_ragbench_loader import (
    get_all_doc_ids,
    get_embedded_corpus,
    load_examples,
    select_corpus_doc_ids,
)
from benchmark.reranker import Reranker


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[output] wrote {len(records)} rows to {path}")


def select_subset(examples, limit, seed):
    """Deterministically pick a subset (same for all systems)."""
    if limit is None or limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    return [examples[i] for i in rng.sample(range(len(examples)), limit)]


def run_naive(subset, embedded_chunks, model, top_k, ollama_model, base_url):
    records = []
    for ex in subset:
        print()
        print(f"[naive] {ex.query_id} :: {ex.question[:60]}")
        records.append(evaluate_naive(ex, embedded_chunks, model, top_k=top_k,
                                      ollama_model=ollama_model, base_url=base_url))
    return records


def run_hybrid(subset, embedded_chunks, model, bm25_index, top_k, dense_k, bm25_k, rrf_k, ollama_model, base_url):
    records = []
    for ex in subset:
        print()
        print(f"[hybrid] {ex.query_id} :: {ex.question[:60]}")
        records.append(evaluate_hybrid(ex, embedded_chunks, model, bm25_index,
                                       top_k=top_k, dense_k=dense_k, bm25_k=bm25_k,
                                       rrf_k=rrf_k, ollama_model=ollama_model, base_url=base_url))
    return records


def run_reranker(subset, embedded_chunks, model, bm25_index, reranker, top_k, candidate_k, dense_k, bm25_k, rrf_k, ollama_model, base_url):
    records = []
    for ex in subset:
        print()
        print(f"[reranker] {ex.query_id} :: {ex.question[:60]}")
        records.append(evaluate_reranker(ex, embedded_chunks, model, bm25_index, reranker,
                                         top_k=top_k, candidate_k=candidate_k, dense_k=dense_k,
                                         bm25_k=bm25_k, rrf_k=rrf_k,
                                         ollama_model=ollama_model, base_url=base_url))
    return records


def run_agentic(subset, embedded_chunks, model, top_k, ollama_model, base_url, max_steps):
    records = []

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _median(values):
    values = sorted([v for v in values if v is not None])
    n = len(values)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2.0


def _p95(values):
    values = sorted([v for v in values if v is not None])
    n = len(values)
    if n == 0:
        return 0.0
    return values[min(int(0.95 * n), n - 1)]


def summarize(records, level="document"):
    ok = [r for r in records if r.get("status") == "ok"]
    suffix = "_doc" if level == "document" else "_sec"
    metrics = []
    for r in ok:
        metrics.append(RetrievalMetrics(
            hit_at_1=r.get(f"hit_at_1{suffix}", 0),
            hit_at_3=r.get(f"hit_at_3{suffix}", 0),
            hit_at_5=r.get(f"hit_at_5{suffix}", 0),
            hit_at_10=r.get(f"hit_at_10{suffix}", 0),
            reciprocal_rank=r.get(f"reciprocal_rank{suffix}", 0.0),
        ))
    agg = aggregate_metrics(metrics)
    summary = {
        "queries_evaluated": len(records), "queries_ok": len(ok),
        "queries_error": len(records) - len(ok),
        "hit_at_1": agg["hit_at_1"], "hit_at_3": agg["hit_at_3"],
        "hit_at_5": agg["hit_at_5"], "hit_at_10": agg["hit_at_10"],
        "mrr": agg["mrr"],
        "avg_latency_seconds": _mean([r.get("latency_seconds") for r in ok]),
        "median_latency_seconds": _median([r.get("latency_seconds") for r in ok]),
        "p95_latency_seconds": _p95([r.get("latency_seconds") for r in ok]),
        "avg_retrieval_calls": _mean([r.get("retrieval_calls") for r in ok]),
        "avg_input_tokens": _mean([r.get("input_tokens") for r in ok]),
        "avg_output_tokens": _mean([r.get("output_tokens") for r in ok]),
    }
    if ok and all("number_of_steps" in r for r in ok):
        summary["avg_steps"] = _mean([r.get("number_of_steps") for r in ok])
        summary["pct_more_than_one_retrieval"] = _mean(
            [1.0 if r.get("retrieval_calls", 0) > 1 else 0.0 for r in ok])
        summary["pct_hit_step_limit"] = _mean(
            [1.0 if r.get("hit_step_limit") else 0.0 for r in ok])
    if ok and all("gold_found_at_call" in r for r in ok):
        dist = {"call_1": 0, "call_2": 0, "call_3_plus": 0, "never": 0}
        for r in ok:
            c = r.get("gold_found_at_call")
            if c == 1:
                dist["call_1"] += 1
            elif c == 2:
                dist["call_2"] += 1
            elif c and c >= 3:
                dist["call_3_plus"] += 1
            else:
                dist["never"] += 1
        summary["gold_found_distribution"] = dist
    for key in ("faithfulness", "answer_correctness", "answer_relevance"):
        vals = [r.get(key) for r in ok if r.get(key) is not None]
        if vals:
            positives = {"faithfulness": "supported",
                         "answer_correctness": "correct",
                         "answer_relevance": "relevant"}
            summary[key] = _mean([1.0 if v == positives.get(key) else 0.0
                                  for v in vals])
    vals_abst = [r.get("abstained") for r in ok if r.get("abstained") is not None]
    if vals_abst:
        summary["abstention_accuracy"] = _mean([1.0 if v else 0.0 for v in vals_abst])
        summary["hallucination_rate"] = 1.0 - summary["abstention_accuracy"]
    return summary


def compute_generation_metrics(records, embedded_chunks, ollama_model, base_url):
    import httpx
    text_by_id = {c.chunk_id: c.text for c in embedded_chunks}
    with httpx.Client(timeout=120.0) as client:
        for rec in records:
            if rec.get("status") != "ok" or not rec.get("answer"):
                rec["faithfulness"] = rec["answer_correctness"] = rec["answer_relevance"] = None
                continue
            q = rec["question"]
            ans = rec["answer"]
            ctx = [text_by_id[cid] for cid in rec.get("retrieved_chunk_ids", []) if cid in text_by_id]
            ref = rec.get("reference_answer", "")
            try:
                rec["faithfulness"] = genmetrics.judge_faithfulness(q, ctx, ans, model=ollama_model, base_url=base_url, client=client)
            except Exception:
                rec["faithfulness"] = None
            try:
                rec["answer_relevance"] = genmetrics.judge_relevance(q, ans, model=ollama_model, base_url=base_url, client=client)
            except Exception:
                rec["answer_relevance"] = None
            try:
                rec["answer_correctness"] = genmetrics.judge_correctness(q, ref, ans, model=ollama_model, base_url=base_url, client=client)
            except Exception:
                rec["answer_correctness"] = None


def print_summary(title, summary):
    print()
    print("=" * 70)
    print(title)
    print(f"Queries evaluated: {summary['queries_evaluated']} (ok: {summary['queries_ok']}, errors: {summary['queries_error']})")
    print("=" * 70)
    print(f"  Hit@1   : {summary['hit_at_1']:.3f}")
    print(f"  Hit@3   : {summary['hit_at_3']:.3f}")
    print(f"  Hit@5   : {summary['hit_at_5']:.3f}")
    print(f"  Hit@10  : {summary['hit_at_10']:.3f}")
    print(f"  MRR     : {summary['mrr']:.3f}")
    print(f"  Avg retrieval calls: {summary['avg_retrieval_calls']:.2f}")
    print(f"  Avg latency        : {summary['avg_latency_seconds']:.2f} s")
    if summary.get("median_latency_seconds") is not None:
        print(f"  Median latency     : {summary['median_latency_seconds']:.2f} s")
    if summary.get("p95_latency_seconds") is not None:
        print(f"  p95 latency        : {summary['p95_latency_seconds']:.2f} s")
    if summary.get("avg_input_tokens") is not None:
        print(f"  Avg input tokens   : {summary['avg_input_tokens']:.0f}")
    if summary.get("avg_output_tokens") is not None:
        print(f"  Avg output tokens  : {summary['avg_output_tokens']:.0f}")
    if "avg_steps" in summary:
        print(f"  Avg steps          : {summary['avg_steps']:.2f}")
        print(f"  >1 retrieval       : {summary['pct_more_than_one_retrieval']*100:.1f} %")
        print(f"  MAX_STEPS reached  : {summary['pct_hit_step_limit']*100:.1f} %")
    if "gold_found_distribution" in summary:
        d = summary["gold_found_distribution"]
        print(f"  Gold found call#1  : {d.get('call_1', 0)}")
        print(f"  Gold found call#2  : {d.get('call_2', 0)}")
        print(f"  Gold found call#3+ : {d.get('call_3_plus', 0)}")
        print(f"  Gold never found   : {d.get('never', 0)}")
    for key in ("faithfulness", "answer_correctness", "answer_relevance",
                "abstention_accuracy", "hallucination_rate"):
        if key in summary:
            print(f"  {key:20s}: {summary[key]:.3f}")


def print_comparison(classical, agentic):
    aligned = cmpmod.align_records(classical, agentic)
    buckets = cmpmod.categorize(aligned)
    total = len(aligned)
    print()
    print("=" * 70)
    print("COMPARISON (document-level retrieval)")
    print("=" * 70)
    for label, key in [("Both succeeded", "both_succeeded"),
                        ("Agentic-only success", "agentic_wins_retrieval"),
                        ("Classical-only success", "classical_wins_retrieval"),
                        ("Both failed", "both_failed")]:
        n = len(buckets[key])
        print(f"  {label:25s}: {n} ({n/max(total,1)*100:.1f} %)")
    wins = cmpmod.agentic_win_summary(aligned, "agentic_wins_retrieval")
    if wins:
        print("--- Agentic retrieval wins ---")
        for w in wins[:10]:
            print()
            print(f"  Q: {w['question']}")
            print(f"    classical top sources : {w['classical_top_sources']}")
            print(f"    agentic queries       : {w['agentic_queries']}")
            print(f"    gold found at call    : {w['gold_found_at_call']} (of {w['retrieval_calls']})")
        if len(wins) > 10:
            print(f"  ... and {len(wins)-10} more agentic wins.")


ABSTENTION_SYSTEM = (
    "You are a strict evaluation judge. Determine whether a generated answer "
    "correctly abstains (refuses to answer) or whether it provides unsupported "
    "factual claims for a question whose answer is absent from the provided context."
)

ABSTENTION_PROMPT = """QUESTION:
{question}

GENERATED ANSWER:
{answer}

Determine:
- Did the answer correctly abstain (refuse to provide facts, say it cannot answer, or state the information is unavailable)?
- Or did the answer provide factual claims (even if vague or speculative)?

Return exactly: VERDICT: abstained  OR  VERDICT: answered
"""


def judge_abstention(question, answer, model="qwen2.5:3b", base_url="http://localhost:11434", client=None):
    payload = {
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": ABSTENTION_SYSTEM},
            {"role": "user", "content": ABSTENTION_PROMPT.format(question=question, answer=answer)},
        ],
        "options": {"temperature": 0.0, "num_predict": 64},
    }
    import httpx
    close = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        r = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    finally:
        if close:
            client.close()
    text = (data["message"].get("content") or "").strip().lower()
    if "abstained" in text:
        return True
    if "answered" in text:
        return False
    return None


def evaluate_unanswerable(queries, run_fn, system_name, embedded_chunks, model, bm25_index, reranker, rcfg, ollama_model, base_url):
    records = []
    for uq in queries:
        ex = type("Ex", (), {"query_id": uq["query_id"], "question": uq["question"],
                            "reference_answer": "", "gold_doc_id": "",
                            "gold_section_id": 0, "query_type": "unanswerable",
                            "source": "unanswerable"})()
        print()
        print(f"[unans:{system_name}] {uq['query_id']}")
        rec = _run_unans(ex, run_fn, embedded_chunks, model, bm25_index, reranker, rcfg, ollama_model, base_url)
        if rec.get("status") == "ok":
            import httpx
            with httpx.Client(timeout=120.0) as client:
                abstained = judge_abstention(uq["question"], rec["answer"], model=ollama_model, base_url=base_url, client=client)
            rec["abstained"] = abstained
            rec["hallucinated"] = (not abstained) if abstained is not None else None
        records.append(rec)
    return records


def _run_unans(ex, run_fn, embedded_chunks, model, bm25_index, reranker, rcfg, ollama_model, base_url):
    fn_name = getattr(run_fn, "__name__", "")
    if "hybrid" in fn_name:
        return run_fn(ex, embedded_chunks, model, bm25_index, top_k=rcfg["top_k"],
                      dense_k=rcfg["dense_k"], bm25_k=rcfg["bm25_k"], rrf_k=rcfg["rrf_k"],
                      ollama_model=ollama_model, base_url=base_url)
    elif "reranker" in fn_name:
        return run_fn(ex, embedded_chunks, model, bm25_index, reranker, top_k=rcfg["top_k"],
                      candidate_k=rcfg["candidate_k"], dense_k=rcfg["dense_k"],
                      bm25_k=rcfg["bm25_k"], rrf_k=rcfg["rrf_k"],
                      ollama_model=ollama_model, base_url=base_url)
    elif "agentic" in fn_name:
        return run_fn(ex, embedded_chunks, model, top_k=rcfg["top_k"],
                      ollama_model=ollama_model, base_url=base_url, max_steps=rcfg["max_steps"])

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="4-way RAG benchmark + unanswerable eval")
    p.add_argument("--system", default="all",
                   help="naive|hybrid|reranker|agentic|all|classical|both")
    p.add_argument("--limit", type=int, default=None,
                   help="Queries to evaluate (0/None=all text)")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--source", default=config.TEXT_ONLY_SOURCE)
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--max-steps", type=int, default=config.DEFAULT_MAX_STEPS)
    p.add_argument("--ollama-model", default=config.DEFAULT_OLLAMA_MODEL)
    p.add_argument("--ollama-base-url", default=config.DEFAULT_OLLAMA_BASE_URL)
    p.add_argument("--embedding-model", default=config.DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=config.DEFAULT_OVERLAP)
    p.add_argument("--corpus-docs", type=int, default=config.DEFAULT_CORPUS_DOCS,
                   help="Corpus budget (0=full 1000). Non-zero = gold-conditioned smoke subset.")
    p.add_argument("--level", choices=["document", "section"], default="document")
    p.add_argument("--candidate-k", type=int, default=config.DEFAULT_CANDIDATE_K)
    p.add_argument("--dense-k", type=int, default=config.DEFAULT_CANDIDATE_K)
    p.add_argument("--bm25-k", type=int, default=config.DEFAULT_CANDIDATE_K)
    p.add_argument("--rrf-k", type=int, default=config.DEFAULT_RRF_K)
    p.add_argument("--reranker-model", default=config.DEFAULT_RERANKER_MODEL)
    p.add_argument("--eval-generation", action="store_true")
    p.add_argument("--eval-unanswerable", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--prepare-corpus", action="store_true",
                   help="Only build/index/embed the corpus, then exit (no generation).")
    p.add_argument("--preflight", action="store_true",
                   help="Run pre-flight checks without generation.")
    p.add_argument("--out-dir", default=str(config.RESULTS_DIR))
    return p.parse_args(argv)


def _normalise_system(raw):
    m = {
        "classical": "naive",
        "both": "all",
        "all": "naive,hybrid,reranker,agentic",
    }
    return m.get(raw, raw)


def _preflight():
    """Lightweight pre-flight checks (no model loading)."""
    import httpx
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, ok, detail))

    add("Open RAGBench files",
        config.QUERIES_PATH.exists() and config.QRELS_PATH.exists()
        and config.ANSWERS_PATH.exists(),
        str(config.OPEN_RAGBENCH_DIR))
    n_docs = len(get_all_doc_ids())
    add("1000 corpus documents", n_docs == 1000, f"{n_docs} docs")
    add("Unanswerable set loaded", config.UNANSWERABLE_QUERIES_PATH.exists(),
        str(config.UNANSWERABLE_QUERIES_PATH))
    ex = load_examples(source_filter=config.TEXT_ONLY_SOURCE)
    s1 = [e.query_id for e in select_subset(ex, 20, 42)]
    s2 = [e.query_id for e in select_subset(ex, 20, 42)]
    add("Deterministic selection", s1 == s2, f"{len(ex)} text queries")
    try:
        tags = httpx.get(f"{config.DEFAULT_OLLAMA_BASE_URL}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        add("Ollama reachable", True, config.DEFAULT_OLLAMA_BASE_URL)
        add("Generator model available", config.DEFAULT_OLLAMA_MODEL in names,
            config.DEFAULT_OLLAMA_MODEL)
    except Exception as exc:
        add("Ollama reachable", False, str(exc)[:120])
        add("Generator model available", False, "(Ollama unreachable)")
    try:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        add("Output dir writable", True, str(config.RESULTS_DIR))
    except Exception as exc:
        add("Output dir writable", False, str(exc)[:120])

    print()
    print("=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}: {detail}")
    print("=" * 60)
    return all(ok for _, ok, _ in checks)


def _warmup(subset, embedded_chunks, model, bm25_index, reranker, args):
    """Run unmeasured warm-up calls so first-call effects do not leak into
    measured per-query latency.  Nothing returned is recorded."""
    warm_q = subset[0].question if subset else "retrieval augmented generation"
    try:
        r = retrieve(warm_q, embedded_chunks, model, top_k=args.top_k)
        if r:
            generate_answer(warm_q, r[:1], model=args.ollama_model, base_url=args.ollama_base_url)
        if bm25_index is not None:
            bm25_index.search(warm_q, top_k=args.top_k)
        if reranker is not None and bm25_index is not None:
            cands = hybrid_retrieve(warm_q, embedded_chunks, model, bm25_index,
                                    top_k=args.candidate_k, dense_k=args.dense_k,
                                    bm25_k=args.bm25_k, rrf_k=args.rrf_k)
            reranker.rerank(warm_q, cands, top_k=args.top_k)
    except Exception as exc:
        print(f"[warmup] non-fatal warm-up error: {exc}")


def main(argv=None):
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system_names = [s.strip() for s in _normalise_system(args.system).split(",")]

    if args.preflight:
        ok = _preflight()
        print()
        print(f"Pre-flight: {'READY' if ok else 'NOT READY'}")
        return

    startup_start = time.perf_counter()

    print("[1/5] Loading Open RAGBench (text only)...")
    examples = load_examples(source_filter=args.source)
    subset = select_subset(examples, args.limit, args.seed)
    print(f"      {len(examples)} text-only examples, evaluating {len(subset)}")

    print("[2/5] Loading embedding model and corpus...")
    model = load_embedding_model(args.embedding_model)
    all_doc_ids = get_all_doc_ids()
    doc_ids = select_corpus_doc_ids(subset, all_doc_ids, args.corpus_docs, args.seed)
    print(f"      corpus documents: {len(doc_ids)}")
    if args.corpus_docs != 0 and args.corpus_docs < len(all_doc_ids):
        print("      >>> WARNING: gold-conditioned corpus subset (smoke test).")
        print("      >>> NOT VALID for the final benchmark. Use --corpus-docs 0.")
    embedded_chunks, cache_path = get_embedded_corpus(
        doc_ids, model, embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size, overlap=args.overlap,
        use_cache=not args.no_cache,
    )

    if args.prepare_corpus:
        print(f"[prepare] Embedded corpus ready: {len(embedded_chunks)} chunks")
        print(f"[prepare] cache path: {cache_path}")
        print("[prepare] Corpus prepared and cached. Exiting before "
              "BM25 / reranker / warm-up / generation.")
        return

    bm25_index = None
    need_bm25 = any(s in system_names for s in ("hybrid", "reranker"))
    if need_bm25:
        print("[2/5] Building BM25 index...")
        bm25_index = BM25Index.build(embedded_chunks,
                                     k1=config.DEFAULT_BM25_K1,
                                     b=config.DEFAULT_BM25_B)
        print(f"      BM25 index: {len(bm25_index.chunk_ids)} docs")

    reranker = None
    if "reranker" in system_names:
        print("[2/5] Loading reranker model...")
        reranker = Reranker(model_name=args.reranker_model)
        print(f"      reranker: {args.reranker_model}")

    startup_time_seconds = time.perf_counter() - startup_start
    print(f"      startup_time_seconds = {startup_time_seconds:.1f} s")

    print("[2/5] Warming up (unmeasured)...")
    _warmup(subset, embedded_chunks, model, bm25_index, reranker, args)
    print("      warm-up done")

    summaries = {}
    all_records = {}

    rcfg = {"top_k": args.top_k, "dense_k": args.dense_k, "bm25_k": args.bm25_k,
            "rrf_k": args.rrf_k, "candidate_k": args.candidate_k,
            "max_steps": args.max_steps}

    for sys_name in system_names:
        print()
        print(f"[3/5] Evaluating {sys_name.upper()} RAG...")
        if sys_name == "naive":
            recs = run_naive(subset, embedded_chunks, model, args.top_k,
                             args.ollama_model, args.ollama_base_url)
        elif sys_name == "hybrid":
            recs = run_hybrid(subset, embedded_chunks, model, bm25_index,
                              args.top_k, args.dense_k, args.bm25_k, args.rrf_k,
                              args.ollama_model, args.ollama_base_url)
        elif sys_name == "reranker":
            recs = run_reranker(subset, embedded_chunks, model, bm25_index, reranker,
                                args.top_k, args.candidate_k, args.dense_k,
                                args.bm25_k, args.rrf_k,
                                args.ollama_model, args.ollama_base_url)
        elif sys_name == "agentic":
            recs = run_agentic(subset, embedded_chunks, model, args.top_k,
                               args.ollama_model, args.ollama_base_url, args.max_steps)
        else:
            print(f"Warning: unknown system {sys_name!r}, skipping")
            continue

        fname_map = {"naive": "naive", "hybrid": "hybrid",
                     "reranker": "hybrid_reranker", "agentic": "agentic"}
        fname = fname_map.get(sys_name, sys_name)
        save_jsonl(recs, out_dir / f"{fname}_results.jsonl")
        all_records[sys_name] = recs

    if args.eval_generation:
        print("[4/5] Running generation evaluation (LLM-as-a-Judge)...")
        for sys_name, recs in all_records.items():
            if not recs:
                continue
            compute_generation_metrics(recs, embedded_chunks,
                                        args.ollama_model, args.ollama_base_url)
            fname_map = {"naive": "naive", "hybrid": "hybrid",
                         "reranker": "hybrid_reranker", "agentic": "agentic"}
            save_jsonl(recs, out_dir / f"{fname_map.get(sys_name, sys_name)}_results.jsonl")

    if args.eval_unanswerable:
        print("[4/5] Running unanswerable evaluation...")
        uq_path = config.UNANSWERABLE_QUERIES_PATH
        if uq_path.exists():
            uq_data = json.load(open(uq_path, encoding="utf-8"))
            uq_queries = uq_data.get("queries", [])
            print(f"      {len(uq_queries)} unanswerable queries loaded")
            for sys_name in system_names:
                fn_map2 = {"naive": run_naive, "hybrid": run_hybrid,
                           "reranker": run_reranker, "agentic": run_agentic}
                run_fn = fn_map2.get(sys_name)
                if run_fn is None:
                    continue
                u_recs = evaluate_unanswerable(
                    uq_queries, run_fn, sys_name, embedded_chunks, model,
                    bm25_index, reranker, rcfg, args.ollama_model, args.ollama_base_url)
                fname_map = {"naive": "naive", "hybrid": "hybrid",
                             "reranker": "hybrid_reranker", "agentic": "agentic"}
                save_jsonl(u_recs, out_dir / f"{fname_map.get(sys_name, sys_name)}_unanswerable.jsonl")
        else:
            print(f"      unanswerable file not found: {uq_path}")

    for sys_name, recs in all_records.items():
        summaries[sys_name] = summarize(recs, args.level)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"startup_time_seconds": startup_time_seconds, "systems": summaries},
                  fh, indent=2, ensure_ascii=False, default=str)
    print()
    print("[output] wrote summary.json")

    print()
    print("[5/5] Results")
    for sys_name in ("naive", "hybrid", "reranker", "agentic"):
        if sys_name in summaries:
            print_summary(f"OPEN RAGBENCH - {sys_name.upper()} ({args.level})",
                          summaries[sys_name])

    naive_recs = all_records.get("naive")
    agentic_recs = all_records.get("agentic")
    if naive_recs and agentic_recs:
        print_comparison(naive_recs, agentic_recs)


if __name__ == "__main__":
    main()
