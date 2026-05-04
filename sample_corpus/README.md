# Sample corpus

This folder contains a tiny non-sensitive PDF corpus for reproducing the demo
without private customer or vendor documents.

## Files

- `pdfs/acme-platform-overview.pdf`
- `pdfs/acme-product-portfolio.pdf`
- `pdfs/acme-support-policy.pdf`

## Run with the sample corpus

```bash
export RAG_PDF_DIR="$PWD/sample_corpus/pdfs"
export RAG_INDEX_DIR="$PWD/sample_corpus/index"
export RAG_DOMAIN_PROFILE=generic

.venv311/bin/python rag_pdf.py indexer
.venv311/bin/python rag_pdf.py retrieve "Which Acme offer is best for a sales engineering demo?"
.venv311/bin/python rag_pdf.py demander "Which Acme offer is best for a sales engineering demo?" --sources
```

The sample questions and expected retrieval signals are in:

```text
sample_corpus/evaluation/questions.json
sample_corpus/evaluation/answers.json
```
