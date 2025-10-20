# End-to-End E-Commerce RAG Pipeline (Daraz & StarTech)
**Version:** 2.0.0 \
**Status:** Operational (Production-Ready)

---

## Abstract

This repository contains a complete, end-to-end pipeline for an intelligent product search engine. The system is a powerful demonstration of a modern data-to-application workflow:

1.  **Ingestion:** Automatically scrapes product data from Daraz and StarTech using a resilient, AI-powered scraping framework.
2.  **Automation:** A CI/CD pipeline (`.github/workflows/pipeline.yml`) automates the scraping and corpus-building process, ensuring the product data remains fresh.
3.  **Indexing:** The raw data is processed into a unified markdown corpus (`out/combined_corpus.md`). This corpus is then indexed using an advanced `Chonkie` chunker and the highly-effective `BM25` lexical retriever.
4.  **Retrieval & Generation (RAG):** The system serves *two* parallel Streamlit applications, allowing the user to choose their preferred LLM backend:
    * **`app_openai.py`**: Utilizes OpenAI models (e.g., GPT-4o-mini) for answer synthesis.
    * **`app_google.py`**: Utilizes Google Generative AI models (e.g., Gemini) for answer synthesis.

The core architectural decision to use `BM25` is not an assumption but a **data-driven conclusion** from prior experimentation, which proved lexical search is superior to vector/hybrid search for this specific, keyword-rich product dataset.

---

## Table of Contents

1.  [System Architecture](#i-system-architecture)
    1.  [Data Ingestion (Scraping)](#11-data-ingestion-scraping)
    2.  [Automated Corpus Pipeline (CI/CD)](#12-automated-corpus-pipeline-cicd)
    3.  [RAG Core: The `BM25` + `Chonkie` Index](#13-rag-core-the-bm25--chonkie-index)
    4.  [Dual LLM Frontends](#14-dual-llm-frontends)
2.  [Key Features](#ii-key-features)
3.  [Technology Stack](#iii-technology-stack)
4.  [Installation & Setup](#iv-installation--setup)
    1.  [Prerequisites](#41-prerequisites)
    2.  [Dependencies](#42-dependencies)
    3.  [Environment Variables](#43-environment-variables)
5.  [Execution](#v-execution)
    1.  [Option A: Run the Automated Pipeline (Recommended)](#51-option-a-run-the-automated-pipeline-recommended)
    2.  [Option B: Run Manually (Local Development)](#52-option-b-run-manually-local-development)
6.  [Architectural Rationale: Why `BM25`?](#vi-architectural-rationale-why-bm25)

---

## I. System Architecture

### 1.1. Data Ingestion (Scraping)

The data pipeline begins with resilient scrapers designed for the target sites.
* **Daraz:** Utilizes `crawl4ai` for robust, AI-powered scraping (see `main.py`, `detail_main.py`).
* **StarTech:** Employs `firecrawl-py` and `beautifulsoup4` for targeted data extraction (see `startech_scraper.py`).

These scripts handle pagination, retries (`tenacity`), and asynchronous operations to produce raw data, which is stored in the `result/` directory.

### 1.2. Automated Corpus Pipeline (CI/CD)

This project is designed for autonomous operation. The `.github/workflows/pipeline.yml` file defines a GitHub Actions workflow that:
1.  Runs on a schedule (e.g., daily).
2.  Executes the Daraz and StarTech scrapers to fetch the latest product data.
3.  Runs the `build_combined_corpus.py` script to process all raw data into the final `out/combined_corpus.md`.
4.  Commits the updated corpus back to the repository.

This makes the RAG application a "living" system that automatically stays current.

### 1.3. RAG Core: The `BM25` + `Chonkie` Index

Both Streamlit applications are powered by a single, shared, and highly-optimized retrieval core.
1.  **Parsing:** At runtime, the app parses the `combined_corpus.md` into structured `ProductDoc` objects. The parser is robust to missing fields and variations.
2.  **Chunking:** Each `ProductDoc` is chunked using `chonkie.RecursiveChunker` with a markdown-aware recipe.
3.  **Indexing:** All chunks are indexed using `rank_bm25.BM25Okapi`, a powerful and efficient lexical search algorithm.
4.  **Persistence:** The built index is cached locally in the `./index` directory. A SHA1 hash of the corpus content ensures the index is only rebuilt if the data changes, enabling sub-second startups.

### 1.4. Dual LLM Frontends

The user can choose which generative backend to use by running the desired application:
* **`app_openai.py`:**
    * **Backend:** Uses the `openai` Python client.
    * **Models:** Configured for `gpt-4o-mini`, `gpt-4.1-mini`, etc.
    * **API Key:** Requires `OPENAI_API_KEY`.
* **`app_google.py`:**
    * **Backend:** Uses the `google.generativeai` Python client.
    * **Models:** Configured for `gemini-1.5-flash`, `gemini-pro`, etc.
    * **API Key:** Requires `GOOGLE_API_KEY`.

Both apps provide an identical user experience, including intelligent filter parsing, streaming answers, and results-exporting capabilities.

## II. Key Features

* **Dual LLM Backends:** Choose between OpenAI or Google Gemini for answer synthesis without changing the retrieval core.
* **Fully Automated Data Pipeline:** CI/CD workflow ensures product data is scraped and updated automatically.
* **Empirically-Validated Retrieval:** Uses `BM25` + `Chonkie`, a combination proven by experiment to be superior to vector search for this dataset.
* **High-Performance Caching:** On-disk index cache provides near-instantaneous app startups.
* **Intelligent Query Parsing:** Automatically extracts filters (price, source, rating) from natural language queries (e.g., "laptops under 50k startech only").
* **Streaming & Export:** Real-time answer streaming and a "Download results (JSON)" button for data portability.
* **Flexible Corpus URL:** The data source URL can be overridden in the Streamlit sidebar for testing.

## III. Technology Stack

This project is segmented into distinct technological components:

| Component | Technology | File(s) |
| :--- | :--- | :--- |
| **Data Scraping** | `crawl4ai`, `firecrawl-py`, `beautifulsoup4`, `tenacity` | `main.py`, `startech_scraper.py` |
| **RAG Core** | `rank_bm25`, `chonkie` | `app_openai.py`, `app_google.py` |
| **LLM Backends** | `openai`, `google-generativeai` | `app_openai.py`, `app_google.py` |
| **Frontend App** | `streamlit` | `app_openai.py`, `app_google.py` |
| **Automation (CI/CD)** | `GitHub Actions` | `.github/workflows/pipeline.yml` |
| **Data Handling** | `requests`, `python-dotenv` | `requirements.txt` |

## IV. Installation & Setup

### 4.1. Prerequisites
* Python 3.9+
* Git

### 4.2. Dependencies

1.  Clone the repository:
    ```bash
    git clone [https://github.com/minhaz-engg/scrape-scheduler.git](https://github.com/minhaz-engg/scrape-scheduler.git)
    cd scrape-scheduler
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install all required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 4.3. Environment Variables

The applications require API keys. Create a file named `.env` in the root of the repository and add your keys:

```bash
# .env file
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
```
The apps will load these keys automatically.

## V. Execution
### 5.1. Option A: Run the Automated Pipeline (Recommended)
This is a "set it and forget it" project.
1. Ensure your API keys (if needed by scrapers) are set as GitHub Secrets.
2. Enable GitHub Actions on your repository fork.
3. The workflow in .github/workflows/pipeline.yml will run on its schedule, automatically updating the out/combined_corpus.md file.
4. You can then deploy the Streamlit apps (e.g., on Streamlit Community Cloud) to point to this auto-updating corpus URL.

### 5.2. Option B: Run Manually (Local Development)
#### Step 1: Run Scrapers (Optional, if corpus is old)

```
# Example commands (adjust as per scripts)
python main.py        # Runs Daraz scraper
python startech_scraper.py  # Runs StarTech scraper
```
#### Step 2: Build Corpus (Optional, if corpus is old)
```
python build_combined_corpus.py  # Merges all data into out/combined_corpus.md
```
#### Step 3: Run the RAG Application
You can run either application. The first time you run one, it will build the BM25 index and cache it.

To run the OpenAI version:

```
streamlit run app_openai.py

```
To run the Google Gemini version:

```
streamlit run app_google.py
```


## VI. Architectural Rationale: Why BM25?
A common trend in RAG systems is the immediate adoption of dense vector search (e.g., FAISS) and embedding models. This project deliberately eschews that approach as a principled decision based on formal experimentation.

1. Data-Driven Decision: Prior experiments on this exact dataset (summarized in the "RAG Experiment Report") demonstrated that the lexical-based BM25 retriever consistently outperformed dense vector and hybrid (BM25 + Dense) search methods on quality metrics (e.g., MRR@5).

2. Lexically-Anchored Data: Product search is dominated by specific keywords: brand names, model numbers, product titles, and categories. BM25 is exceptionally powerful at this "lexical anchoring." A query for "Ryzen 5 5600G" needs to find that exact string, a task BM25 excels at.

3. Efficiency & Cost: Vector-based RAG introduces significant overhead:

4. Cost: An API call to an embedding model (e.g., text-embedding-3-small) for every chunk during indexing.

5. Latency: The complexity of running a vector database and a separate BM25 index, plus a fusion algorithm.

6. Complexity: Managing vector state and embedding models.

Our experiments proved that this additional overhead was superfluous and provided negative value (lower accuracy) for this use case. The current architecture is, therefore, the empirically-validated optimal solution: it is faster, cheaper, less complex, and more accurate.
