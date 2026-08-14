"""
Benchmark package for comparing Classical RAG vs Agentic RAG on Open RAGBench.

This is a fully self-contained evaluation harness. It reuses the existing
RAG modules under ``src/`` (chunking, embeddings, retrieval, generation,
and the ReAct agent) and does NOT re-implement any of that logic.

Modules
-------
- open_ragbench_loader : load/filter the benchmark data and build the embedded corpus
- metrics              : retrieval metrics (Hit@k, MRR)
- evaluators           : run Classical and Agentic RAG and record results
- generation_metrics   : optional LLM-as-a-Judge generation metrics
- comparison           : qualitative head-to-head comparison
- evaluate_rag         : command-line entry point
"""