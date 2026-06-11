# Retrieval-Augmented QA over a Custom Corpus

A three-stage retrieval-augmented generation (RAG) pipeline for open-domain question answering, built end to end: dense retrieval, cross-encoder re-ranking, and a LoRA fine-tuned reader. Trained and evaluated on SQuAD 2.0, including its unanswerable questions, so the system has to know when *not* to answer.

This is the retrieval-and-generation core that sits underneath most practical assistant systems — answering a question over a body of documents the model was never trained on. I wanted to build the whole thing from the ground up rather than calling a managed RAG API, to understand where each stage helps, where it doesn't, and how retrieval quality caps everything downstream.

## What it does

1. **Bi-encoder retrieval** — `all-MiniLM-L6-v2` embeds a corpus of 20,233 Wikipedia passages into a FAISS IVFFlat index; a question retrieves the top 20 candidates by cosine similarity in a few milliseconds.
2. **Cross-encoder re-ranking** — `ms-marco-MiniLM-L-6-v2` reads each (question, passage) pair jointly and re-orders the shortlist, picking the best 5. No training, ~120 ms/query, measurably better passage selection.
3. **LoRA fine-tuned reader** — `deberta-v3-base-squad2` extracts the answer span (or abstains). Fine-tuned with LoRA (r=16, α=32) — 886K trainable parameters, 0.48% of the model — in 31 minutes on a single A100.

## Results

Evaluated on 500 validation questions with the official SQuAD 2.0 metric.

| System | Exact Match | F1 |
|---|---|---|
| Baseline RAG (frozen reader) | 61.4% | 64.5% |
| + LoRA + cross-encoder re-ranker | **62.8%** | **66.4%** |
| Human performance | 86.9% | 89.5% |

The gap to human performance is mostly retrieval, not the reader: hit rate plateaus at ~84% by k=5, so roughly 1 in 6 answerable questions is already lost before the reader sees a token. No amount of reader tuning recovers those — which is itself the most useful thing the project taught me about where to spend effort in a RAG system.

## The most instructive part: an 11-point regression

My first LoRA run made the system *worse* — F1 dropped 11.6 points. The cause wasn't bad luck, it was a conceptual error: I was fine-tuning from the raw `microsoft/deberta-v3-base` on 20K examples while trying to beat `deepset/deberta-v3-base-squad2`, which had already been trained on all 130K. I was replacing a fully trained model with a half-trained one and calling it an improvement. Fine-tuning *from* the trained baseline instead recovered the loss and then exceeded it.

There's a second one worth keeping: fp16 training threw "Attempting to unscale FP16 gradients" because LoRA keeps adapter weights in fp32 while the base sat in fp16, which the AMP grad scaler can't reconcile. Switching to bf16 fixed it and was faster anyway. Both are the kind of failure you only understand by building the pipeline yourself.

## Relevance to assistant / privacy-sensitive systems

This is the same retrieval-over-personal-context problem that an assistant has to solve to answer questions grounded in a user's own documents. One caveat the project made concrete: the embeddings in a RAG index can leak information about the source text, so building this over *private* data isn't just a modelling problem — it needs secure indexing, access control, and audit logging as first-class concerns rather than afterthoughts. The retrieval architecture here is the part that would have to be rethought to run client-side or inside a trusted boundary if the underlying data can't be exposed to the server.

## Quick start (Google Colab Pro)

1. Open `rag_qa_pipeline.ipynb` in Colab.
2. Runtime → Change runtime type → GPU → **A100**.
3. Run **Cell 1 only** — it pins dependencies in the correct order and auto-restarts the runtime.
4. After restart, skip Cell 1 and run all remaining cells top to bottom.

Plots and metrics are written to `results/`. The dataset loads automatically from the Hugging Face Hub — no manual download.

## Stack

`PyTorch` · `transformers` · `peft` (LoRA) · `sentence-transformers` · `FAISS` · `datasets` · `evaluate`

Dataset: [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) (CC BY-SA 4.0).

## One dependency note that will save you an hour

numpy must be pinned to `1.26.4`. numpy 2.x breaks pyarrow 14, which breaks the `datasets` library with a cryptic dataclass error. Cell 1 handles this by uninstalling and reinstalling in the right order. `bitsandbytes` is deliberately not used — its GPU build fails on Colab's Python 3.12 + CUDA 12.x, and bf16 LoRA fits in A100 memory without 4-bit quantization anyway.
