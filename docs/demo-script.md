# Demo script

This script is designed for a Sales Engineering or internal interview demo.
It explains what to ask, what to inspect in the RAG trace, and how to talk
about the answer without making the demo feel magical.

## Before the demo

1. Start from a clean terminal.
2. Load the OpenAI key from macOS Keychain or `.env`.
3. Make sure the corpus is indexed.
4. Open Streamlit.

```bash
export OPENAI_API_KEY="$(security find-generic-password -a "$USER" -s "OPENAI_API_KEY" -w)"
.venv311/bin/python rag_pdf.py indexer
.venv311/bin/streamlit run app_streamlit.py
```

For a public demo without private PDFs, use the sample corpus:

```bash
export RAG_PDF_DIR="$PWD/sample_corpus/pdfs"
export RAG_INDEX_DIR="$PWD/sample_corpus/index"
.venv311/bin/python rag_pdf.py indexer
.venv311/bin/streamlit run app_streamlit.py
```

## Five questions to ask

1. `What does the Acme platform do?`
2. `Which Acme offer is best for a sales engineering demo?`
3. `List the support levels and response times.`
4. `How can the solution generate value for a system integrator?`
5. `What is the warranty period for the quantum module?`

The fifth question is intentionally out of scope. The expected behavior is that
the assistant says it does not know instead of inventing an answer.

## What to inspect in the RAG trace

- Retrieved documents: did the right file appear?
- Query expansion: did synonyms or rewritten queries help?
- Scores: do FAISS, BM25 and final score tell a coherent story?
- Chunks: is the answer grounded in the displayed excerpts?
- Citations: does each important claim point to a source number?
- Model choice: does a larger model improve synthesis or only style?

## How to explain it in an interview

Short version:

> This is not a chatbot over documents. It is a controlled retrieval pipeline.
> The LLM only writes after the system has selected evidence from a known corpus.

Longer version:

> I built the project to demonstrate the full RAG chain: ingestion, chunking,
> embeddings, FAISS retrieval, BM25 lexical scoring, optional reranking, prompt
> construction and sourced generation. The important part is not only the final
> answer. The trace shows how the answer was built, which is what makes the
> system explainable for Sales Engineering.

## What good looks like

- The answer is concise and directly addresses the question.
- The sources are clickable.
- The trace shows relevant chunks, not random nearby text.
- The system admits uncertainty when the corpus does not contain the answer.
- You can explain which lever to adjust: `top_k`, `candidate_k`, reranker,
  query rewrite, or corpus quality.
