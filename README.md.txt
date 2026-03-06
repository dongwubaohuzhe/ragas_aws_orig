# RAGAS Evaluation Tool

## Overview

Automated testing framework for RAG (Retrieval Augmented Generation) applications using RAGAS (Retrieval Augmented Generation Assessment). The Streamlit app runs RAGAS evaluations against your knowledge base API and Bedrock LLMs, then lets you download results as CSV.

## Features

- **Knowledge base retrieval testing** via configurable API (tenant, KB name, bearer token)
- **RAGAS metrics**: Faithfulness, Context Recall, Context Precision, Answer Relevancy
- **Multiple Bedrock models**: Claude 3.5/3.7 Sonnet, Claude Sonnet 4.5, Amazon Nova Pro, Titan
- **GovCloud**: Uses cross-region inference profile for Claude Sonnet 4.5 (`us-gov-west-1`)
- **Robust CSV handling**: Ignores pandas’ default NA conversion; tolerates API responses with `NaN` in JSON
- **Evaluation tuning**: Higher per-metric timeout (360s) and lower concurrency (8 workers) to reduce TimeoutErrors

## Setup

**Requirements:** Python 3.12, pip

```bash
# Create and use a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_ragas_eval.py
```

Open the URL shown (e.g. http://localhost:8501).

## Configuration (Sidebar)

- **API URL** – Your chat/retrieval API base URL
- **Bearer Token** – Auth for the API
- **Tenant** & **Knowledge Base Name** – Used in API payloads
- **LLM Model** – Model used for answer generation and for RAGAS scoring (see below)
- **Embedding Model** – Used for RAGAS metrics that need embeddings (e.g. Answer Relevancy)

## Supported LLM Models

| Model | Notes |
|-------|--------|
| Claude Sonnet 4.5 | Use inference profile ID: `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0` (required in GovCloud) |
| Claude 3.7 Sonnet | `anthropic.claude-3-7-sonnet-20250219-v1:0` |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20240620-v1:0` |
| Amazon Nova Pro | `amazon.nova-pro-v1:0` (Converse API) |
| Amazon Titan Text Express | `amazon.titan-text-express-v1` |

All models use the Bedrock Converse API via `langchain-aws` with configurable `temperature` and `max_tokens` (see `model_config.py`).

## Test Plan CSV Format

- **Required columns:** `question`, `ground_truth`
- Rows with empty question or ground_truth are dropped; a warning shows how many.
- CSV is read with `keep_default_na=False` so values like "NA", "N/A", "null" stay as text instead of becoming NaN.

## RAGAS Metrics

- **Faithfulness** – Answer consistent with retrieved context
- **Context Recall** – How much of the ground truth is supported by context
- **Context Precision** – Relevance of retrieved contexts to the question (often the slowest; can timeout on long contexts)
- **Answer Relevancy** – Relevance of the answer to the question

Failed metric jobs (e.g. timeouts) are logged in the console and produce `nan` for that cell; the rest of the run continues.

## Evaluation Behavior

- **Timeout:** 360 seconds per (row, metric) job (configurable via `RunConfig` in code).
- **Concurrency:** 8 workers by default to avoid overloading Bedrock.
- **Progress:** A progress bar appears in the console during evaluation.

To change timeout or concurrency, edit `streamlit_ragas_eval.py` and adjust `RunConfig(timeout=..., max_workers=...)`.

## Output

After evaluation completes you can download a timestamped CSV of scores plus metadata (knowledge base name, model IDs, etc.).

## Troubleshooting

- **ValidationException / inference profile:** For Claude Sonnet 4.5 in GovCloud, the app uses the inference profile ID `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0`. Do not use the base model ID for that model.
- **TimeoutError in Job[N]:** Usually **context_precision** on rows with many/long contexts. Increase `RunConfig(timeout=600, ...)` or reduce `max_workers` if needed.
- **NaN / float errors:** Ensure CSV has `question` and `ground_truth` populated; the app normalizes and drops empty rows. If your API returns JSON with `NaN`, the client parses it safely.
