# -*- coding: utf-8 -*-
"""
Daraz + StarTech Products RAG — RAPTOR Hierarchical Indexing (Streamlit App)
===========================================================================

What this file does (high level):
- Implements the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) methodology.
- Loads a product corpus from a raw GitHub URL.
- Parses products into `ProductDoc` objects. These initial documents form the "leaf nodes" (Level 0) of our tree.
- Recursively builds a hierarchical tree:
  1. Embeds the text nodes at the current level.
  2. Reduces embedding dimensionality using UMAP for efficient clustering.
  3. Clusters semantically similar nodes using Gaussian Mixture Models (GMM).
  4. Summarizes the content of each cluster using an LLM to create parent nodes for the next level.
  5. Repeats this process until no new, meaningful clusters can be formed.
- Creates a "Collapsed Tree" Index: All nodes (original chunks and generated summaries from all levels) are indexed together in a single FAISS vector store.
- Allows searching and filtering (source, category, price).
- Streams a grounded LLM answer using the retrieved multi-resolution context.

Dependencies (pip):
    streamlit, python-dotenv, openai, requests, langchain, langchain-openai,
    sentence-transformers, faiss-cpu, umap-learn, scikit-learn, numpy, pandas, tiktoken

Environment:
    export OPENAI_API_KEY="sk-..."
"""

import os
import re
import io
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import streamlit as st
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# App Config
# ----------------------------
INDEX_DIR = "raptor_index"  # local cache folder for RAPTOR index
os.makedirs(INDEX_DIR, exist_ok=True)

# ---!!! IMPORTANT: SET YOUR DEFAULT URL HERE!!! ---
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
# ---

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 10
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ProductDoc:
    """A single product parsed from the combined corpus."""
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    url: Optional[str]
    raw_md: str

@dataclass
class NodeRec:
    """A single node in our RAPTOR tree (can be a leaf chunk or a summary)."""
    node_id: str
    text: str
    metadata: Dict[str, Any]

# ----------------------------
# Utilities
# ----------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _index_paths(sig: str) -> Tuple[str, str]:
    return (
        os.path.join(INDEX_DIR, f"faiss_{sig}.pkl"),
        os.path.join(INDEX_DIR, f"nodes_{sig}.pkl"),
    )

def _parse_price_value(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "")
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return None
    try:
        vals = [float(x) for x in nums]
        return min(vals) if vals else None
    except Exception:
        return None

def _ensure_client_is_ready():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

# ----------------------------
# Parsing combined corpus
# ----------------------------

def parse_combined_products_from_md(md_text: str) -> List:
    """Robust parser for the combined corpus."""
    text = (md_text or "").strip()
    text = re.sub(r"^#\s*Combined.*?\n", "", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"\s+---\s+", text) if p.strip()]

    products: List =
    for part in parts:
        m = re.search(r"##\s*(.+?)\s*(?=\*\*DocID:\*\*|\*\*DOCID:\*\*|DocID:|DOCID:)", part, re.IGNORECASE | re.DOTALL)
        title = (m.group(1).strip() if m else "").strip()
        if not title:
            continue

        m = re.search(r"\*\*DocID:\*\*\s*`?([A-Za-z0-9_\-]+)`?|DocID:\s*`?([A-Za-z0-9_\-]+)`?", part, re.IGNORECASE)
        doc_id = (m.group(1) or m.group(2) or "").strip() if m else None
        if not doc_id:
            continue

        m = re.search(r"\*\*URL:\*\*\s*(\S+)", part, re.IGNORECASE)
        url = m.group(1).strip() if m else None

        m = re.search(r"\*\*Source:\*\*\s*([^*]+)", part, re.IGNORECASE)
        source = m.group(1).strip() if m else None
        if not source:
            if doc_id.lower().startswith("daraz_"): source = "Daraz"
            elif doc_id.lower().startswith("startech_"): source = "StarTech"

        m = re.search(r"\*\*Category:\*\*\s*([^*]+)", part, re.IGNORECASE)
        category = m.group(1).strip() if m else None

        m = re.search(r"\*\*Price:\*\*\s*([^*]+)", part, re.IGNORECASE)
        price_value = _parse_price_value(m.group(1)) if m else None

        bits =
        meta_fields =
        if source: meta_fields.append(f"**Source:** {source}")
        if category: meta_fields.append(f"**Category:** {category}")
        if price_value is not None: meta_fields.append(f"**Price:** ৳{int(price_value)}")
        if url: meta_fields.append(f"**URL:** {url}")
        bits.append(" ".join(meta_fields))
        
        # Add the rest of the description
        desc_part = part.split("---")
        desc_lines = desc_part.split("\n")
        # Find the start of the description (after the header line)
        header_line_index = -1
        for i, line in enumerate(desc_lines):
            if doc_id in line:
                header_line_index = i
                break
        if header_line_index!= -1 and header_line_index + 1 < len(desc_lines):
            description = "\n".join(desc_lines[header_line_index+1:]).strip()
            if description:
                bits.append("\n" + description)

        raw_md = "\n".join(bits)

        products.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category,
            price_value=price_value, url=url, raw_md=raw_md
        ))
    return products

# ----------------------------
# RAPTOR: Hierarchical Indexing Engine
# ----------------------------

def embed_texts(texts: List[str], embedding_model) -> np.ndarray:
    """Embed a list of texts."""
    embeddings = embedding_model.embed_documents(texts)
    return np.array(embeddings)

def reduce_and_cluster(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Reduce dimensionality with UMAP and cluster with GMM."""
    # Reduce dimensionality, ensuring n_neighbors is less than sample size
    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors <= 1: return np.array([-1] * len(embeddings)) # Not enough samples to cluster

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=min(50, len(embeddings)-1), metric="cosine", random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Cluster with GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gmm.fit_predict(reduced_embeddings)

def summarize_cluster(texts_in_cluster: List[str], llm) -> str:
    """Summarize the content of a cluster using an LLM."""
    combined_text = "\n\n---\n\n".join(texts_in_cluster)
    
    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert in synthesizing information. Summarize the following collection of product descriptions, "
        "capturing the main product types, key features, and themes. The summary should be abstractive and concise.\n\n"
        "Content:\n{text}"
    )
    
    summarization_chain = prompt_template | llm | StrOutputParser()
    return summarization_chain.invoke({"text": combined_text})

@st.cache_resource(show_spinner="Building RAPTOR Index...")
def build_raptor_index(_products: List, _corpus_hash: str) -> Tuple]:
    """
    Builds the RAPTOR tree and collapsed index.
    Streamlit's cache decorator handles caching based on the corpus hash.
    """
    _ensure_client_is_ready()
    
    embedding_model = SentenceTransformerEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    llm = ChatOpenAI(temperature=0, model=DEFAULT_MODEL)

    # Level 0: Leaf nodes are the initial product docs
    leaf_nodes =

    all_nodes = list(leaf_nodes)
    current_level_nodes = leaf_nodes
    level = 0

    while True:
        print(f"\n--- Processing Level {level} with {len(current_level_nodes)} nodes ---")

        current_texts = [node.text for node in current_level_nodes]
        current_embeddings = embed_texts(current_texts, embedding_model)

        # Stopping condition
        if len(current_level_nodes) <= 3:
            print("Stopping: Number of nodes is too small for further clustering.")
            break
        
        # Heuristic for determining number of clusters
        n_clusters = max(2, int(len(current_level_nodes) / 5))
        if n_clusters >= len(current_level_nodes):
            print("Stopping: Cannot have more clusters than nodes.")
            break

        cluster_labels = reduce_and_cluster(current_embeddings, n_clusters)
        
        clustered_texts = {}
        for i, label in enumerate(cluster_labels):
            if label not in clustered_texts:
                clustered_texts[label] =
            clustered_texts[label].append(current_texts[i])
            
        next_level_nodes =
        for cluster_id, texts in clustered_texts.items():
            if len(texts) > 1:
                print(f"Summarizing cluster {cluster_id} with {len(texts)} nodes...")
                summary = summarize_cluster(texts, llm)
                summary_id = f"L{level+1}_C{cluster_id}"
                new_node = NodeRec(
                    node_id=summary_id,
                    text=summary,
                    metadata={"level": level + 1, "doc_id": summary_id, "title": f"Summary of Cluster {cluster_id}"}
                )
                next_level_nodes.append(new_node)
        
        if not next_level_nodes:
            print("Stopping: No new summaries were generated.")
            break
            
        all_nodes.extend(next_level_nodes)
        current_level_nodes = next_level_nodes
        level += 1

    print("\n--- Tree construction complete ---")
    
    # Index the collapsed tree
    node_texts = [node.text for node in all_nodes]
    node_metadatas = [node.metadata for node in all_nodes]
    
    vectorstore = FAISS.from_texts(texts=node_texts, embedding=embedding_model, metadatas=node_metadatas)
    print(f"Collapsed tree with {len(all_nodes)} nodes indexed in FAISS.")
    
    return vectorstore, all_nodes

# ----------------------------
# Retrieval + filtering
# ----------------------------

def _passes_filters(
    metadata: Dict[str, Any],
    allowed_sources: Optional[set],
    allowed_categories: Optional[set],
    category_contains: Optional[str],
    price_max: Optional[float],
) -> bool:
    # Summaries (level > 0) are not filtered by product attributes
    if metadata.get("level", 0) > 0:
        return True
        
    if allowed_sources and (metadata.get("source") not in allowed_sources):
        return False
    if allowed_categories and (metadata.get("category") not in allowed_categories):
        return False
    if category_contains:
        cc = (metadata.get("category") or "").lower()
        if category_contains.lower() not in cc:
            return False
    if price_max is not None and (metadata.get("price_value") is not None) and (metadata["price_value"] > price_max):
        return False
    return True

def vector_search(
    vectorstore: FAISS,
    query: str,
    top_k: int,
    allowed_sources: Optional[set] = None,
    allowed_categories: Optional[set] = None,
    category_contains: Optional[str] = None,
    price_max: Optional[float] = None,
) -> List, float]]:
    
    # Perform an initial, larger search to account for filtering
    initial_k = top_k * 5
    results_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)
    
    filtered_results =
    seen_doc_ids = set()
    
    for doc, score in results_with_scores:
        if _passes_filters(doc.metadata, allowed_sources, allowed_categories, category_contains, price_max):
            # Diversify: prefer one result per original doc_id first
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in seen_doc_ids:
                filtered_results.append((doc.metadata, doc.page_content, score))
                seen_doc_ids.add(doc_id)
        
        if len(filtered_results) >= top_k:
            break
            
    return filtered_results

# ----------------------------
# OpenAI helpers
# ----------------------------

def _build_messages(query: str, results: List]) -> List]:
    ctx_blocks =
    for i, (meta, text, score) in enumerate(results, 1):
        title = meta.get('title', 'Summary')
        doc_id = meta.get('doc_id', 'N/A')
        level = meta.get('level', 'N/A')
        
        head = f"[{i}] Title: {title} — DocID: {doc_id} — Level: {level}"
        
        fields =
        if meta.get('source'): fields.append(f"Source: {meta['source']}")
        if meta.get('category'): fields.append(f"Category: {meta['category']}")
        if meta.get('price_value') is not None: fields.append(f"Price: ~৳{int(meta['price_value'])}")
        
        meta_line = " | ".join(fields)
        ctx_blocks.append(f"{head}\n{meta_line}\n---\n{text}\n")

    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Keep answers concise with bullets. "
        "Cite as [#] with DocID."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def stream_answer(model: str, messages: List], temperature: float = 0.2):
    _ensure_client_is_ready()
    client = ChatOpenAI(model=model, temperature=temperature)
    for chunk in client.stream(messages):
        yield chunk.content

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="RAG: Daraz + StarTech (RAPTOR)", layout="wide")
st.title("Daraz + StarTech Products RAG — RAPTOR Hierarchical Indexing")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"], index=0)
    top_k = st.slider("Top-K results", 1, 25, DEFAULT_TOPK)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.markdown("### 📚 Corpus Source")
    st.caption("Leave blank for default, or provide a new raw URL.")
    remote_url_override = st.text_input("Override Corpus URL", "", placeholder=DEFAULT_CORPUS_URL)

# Load corpus
url_to_fetch = remote_url_override.strip() or DEFAULT_CORPUS_URL
md_text = None
try:
    import requests
    with st.spinner(f"Fetching corpus from {url_to_fetch[:70]}..."):
        r = requests.get(url_to_fetch, timeout=30)
        if r.ok:
            md_text = r.text
        else:
            st.error(f"Failed to fetch corpus. HTTP Status: {r.status_code}")
except Exception as e:
    st.error(f"Error fetching data: {e}")

if not md_text:
    st.error("Corpus text could not be loaded. App cannot continue.")
    st.stop()

# Parse products
with st.spinner("Parsing product corpus..."):
    products = parse_combined_products_from_md(md_text)

if not products:
    st.error("No products detected in the corpus.")
    st.stop()

# Build RAPTOR index (this will be cached by Streamlit)
corpus_hash = _sha1(md_text)
vectorstore, all_nodes = build_raptor_index(products, corpus_hash)

# Facets for filtering
all_sources = sorted({p.source for p in products if p.source})
all_categories = sorted({p.category for p in products if p.category})

st.success(f"Parsed **{len(products):,}** products → **{len(all_nodes):,}** total nodes in RAPTOR index. Ready to search.")

st.markdown("#### Filters")
c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1.5])
with c1:
    sel_sources = st.multiselect("Source", options=all_sources, default=)
with c2:
    sel_categories = st.multiselect("Category (exact)", options=all_categories, default=)
with c3:
    cat_contains = st.text_input("Category contains", "")
with c4:
    price_max_ui = st.text_input("Max price (BDT)", "")

def _to_float(x: str) -> Optional[float]:
    x = x.strip().replace(",", "")
    return float(x) if x and re.match(r"^\d+(?:\.\d+)?$", x) else None

price_max_filter = _to_float(price_max_ui)

# Query UI
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'best wireless gamepad under 1500')", "")
go = st.button("Search")

if go and query.strip():
    with st.spinner("Searching with RAPTOR index..."):
        results = vector_search(
            vectorstore, query, top_k,
            allowed_sources=set(sel_sources) if sel_sources else None,
            allowed_categories=set(sel_categories) if sel_categories else None,
            category_contains=cat_contains.strip() or None,
            price_max=price_max_filter,
        )

    if not results:
        st.warning("No results matched your query/filters.")
        st.stop()

    st.subheader("Answer")
    messages = _build_messages(query, results)
    try:
        st.write_stream(stream_answer(model, messages, temperature=temperature))
    except Exception as e:
        st.error(f"OpenAI error: {e}")

    with st.expander("View Top Matches (Context Used)", expanded=False):
        st.subheader("Top matches")
        for i, (meta, text, score) in enumerate(results, 1):
            title = meta.get('title', 'Summary')
            doc_id = meta.get('doc_id', 'N/A')
            level = meta.get('level', 'N/A')
            
            meta_bits = [f"**Level:** {level}"]
            if meta.get('source'): meta_bits.append(f"**Source:** {meta['source']}")
            if meta.get('category'): meta_bits.append(f"**Category:** {meta['category']}")
            if meta.get('price_value') is not None: meta_bits.append(f"**Price:** ~৳{int(meta['price_value'])}")
            
            st.markdown(
                f"**[{i}] {title}**\n"
                f"DocID: `{doc_id}` • Score: `{score:.3f}`\n"
                + (" • ".join(meta_bits))
            )
            with st.expander("View node text"):
                st.write(text)

    # Export matched items as JSON
    export_rows = [
        {
            "rank": i, "score": score, "text": text, **meta
        } for i, (meta, text, score) in enumerate(results, 1)
    ]
    export_bytes = io.BytesIO(json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8"))
    st.download_button("Download results (JSON)", data=export_bytes, file_name="results.json", mime="application/json")

