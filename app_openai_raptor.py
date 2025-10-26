# -*- coding: utf-8 -*-
"""
Daraz + StarTech Products RAG — BM25 + Chonkie + RAPTOR (Streamlit App)
======================================================================

What this file does (high level):
- Loads your combined corpus (Daraz + StarTech) from a default raw GitHub URL (can override in UI).
- Parses each inline product entry like:
    ## <Title> **DocID:** `<id>` **Source:** <Daraz|StarTech> **Category:** <txt> **Price:** <৳...> ---
- Converts products into compact ProductDoc objects with metadata (source, category, price_value, etc.).
- Uses **Chonkie** to chunk each product (usually 1 short chunk per item).
- Builds **BM25** over chunks, and (optionally) a **RAPTOR** tree:
    - Leaf embeddings (OpenAI) + hierarchical clustering (scikit-learn)
    - LLM summaries per cluster (OpenAI), recursively
    - Cached fully in ./index/ for reuse.
- Retrieval modes:
    1. BM25 only
    2. RAPTOR only (semantic)
    3. Hybrid (RRF rank fusion — BM25 + RAPTOR)
- Streams a strictly grounded LLM answer. Citations reference DocIDs.

Dependencies (pip):
    streamlit, python-dotenv, openai, rank_bm25, chonkie, requests, scikit-learn, numpy

Environment:
    export OPENAI_API_KEY="sk-..."
"""

import os
import re
import io
import json
import math
import time
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from sklearn.cluster import KMeans

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# App Config
# ----------------------------
INDEX_DIR = "index"                             # cache folder for BM25 + RAPTOR
os.makedirs(INDEX_DIR, exist_ok=True)

# --- !!! IMPORTANT: SET YOUR DEFAULT URL HERE !!! ---
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
# ---

DEFAULT_MODEL = "gpt-4o-mini"                   # chat generation
DEFAULT_TOPK = 10                               # how many chunks for LLM grounding
DEFAULT_LANG = "en"                             # Chonkie recipe language

# RAPTOR defaults (tunable in UI)
EMBED_MODEL_DEFAULT = "text-embedding-3-small"
SUMMARIZER_MODEL_DEFAULT = "gpt-4o-mini"
RAPTOR_CLUSTER_SIZE_DEFAULT = 8     # target children per parent cluster
RAPTOR_MAX_DEPTH_DEFAULT = 3        # how many parent layers to build (stops early if few nodes)
RAPTOR_TOPN_PER_LEVEL_DEFAULT = 6   # nodes to keep per level for traversal
RRF_K_DEFAULT = 60                  # RRF constant

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ProductDoc:
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    url: Optional[str]
    raw_md: str

@dataclass
class ChunkRec:
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    url: Optional[str]
    text: str

# RAPTOR internal node (serializable)
@dataclass
class RaptorNode:
    node_id: str
    level: int
    children: List[str]          # node_ids
    text_ids: List[int]          # indices of leaf chunks
    size: int                    # number of leaf chunks under this node
    summary: Optional[str]       # None for leaves
    embedding: Optional[List[float]]  # embedding for this node (list for pickle)
    kind: str                    # "leaf" or "summary"


# ----------------------------
# Regex helpers for the combined corpus
# ----------------------------

ITEM_RE = re.compile(
    r"##\s*(?P<title>.*?)\s*"
    r"\*\*DocID:\*\*\s*`(?P<docid>[^`]+)`"
    r"(?:\s*\*\*Source:\*\*\s*(?P<source>[^*]+?))?"
    r"(?:\s*\*\*Category:\*\*\s*(?P<category>[^*]+?))?"
    r"(?:\s*\*\*Price:\*\*\s*(?P<price>.*?))?"
    r"(?:\s*\*\*URL:\*\*\s*(?P<url>\S+))?"
    r"\s*(?:---|$)",
    re.IGNORECASE | re.DOTALL,
)

RATING_RE = re.compile(
    r"\*\*Rating:\*\*\s*([0-5](?:\.\d+)?)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?",
    re.IGNORECASE,
)

# ----------------------------
# Utilities
# ----------------------------

STOPWORDS = set([
    "the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were",
    "this","that","these","those","it","its","as","be","can","will","has","have"
])

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _index_paths(sig: str) -> Tuple[str, str]:
    return (
        os.path.join(INDEX_DIR, f"bm25_{sig}.pkl"),
        os.path.join(INDEX_DIR, f"meta_{sig}.pkl"),
    )

def _raptor_path(sig: str) -> str:
    return os.path.join(INDEX_DIR, f"raptor_{sig}.pkl")

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

def _clean_for_bm25(text: str) -> str:
    clean_lines = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll:
            continue
        if ll.lower().startswith("**images"):
            continue
        if "http://" in ll or "https://" in ll:
            parts = re.split(r"\s+https?://\S+", ll)
            ll = " ".join([p for p in parts if p.strip()])
            if not ll:
                continue
        clean_lines.append(ll)
    return "\n".join(clean_lines)

def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in toks if t not in STOPWORDS]

def _ensure_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
    return OpenAI()

def _cosine_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q: (d,), M: (n,d)
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return (Mn @ qn)

def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn + 1e-12:
        return np.ones_like(x)
    return (x - mn) / (mx - mn)

# ----------------------------
# Parsing combined corpus
# ----------------------------

def parse_combined_products_from_md(md_text: str) -> List[ProductDoc]:
    text = (md_text or "").strip()
    text = re.sub(r"^#\s*Combined.*?\n", "", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"\s+---\s+", text) if p.strip()]

    products: List[ProductDoc] = []
    for part in parts:
        m = re.search(r"##\s*(.+?)\s*(?=\*\*DocID:\*\*|\*\*DOCID:\*\*|DocID:|DOCID:)", part, re.IGNORECASE | re.DOTALL)
        title = (m.group(1).strip() if m else "").strip()
        if not title:
            continue

        m = re.search(r"\*\*DocID:\*\*\s*`?([A-Za-z0-9_\-]+)`?|DocID:\s*`?([A-Za-z0-9_\-]+)`?", part, re.IGNORECASE)
        doc_id = None
        if m:
            doc_id = (m.group(1) or m.group(2) or "").strip()
        if not doc_id:
            continue

        m = re.search(r"\*\*URL:\*\*\s*(\S+)", part, re.IGNORECASE)
        url = m.group(1).strip() if m else None

        m = re.search(r"\*\*Source:\*\*\s*([^*]+)", part, re.IGNORECASE)
        source = m.group(1).strip() if m else None
        if not source:
            m2 = re.search(r"\bSource:\s*([A-Za-z][A-Za-z \-]+)", part, re.IGNORECASE)
            source = m2.group(1).strip() if m2 else None
        if not source:
            if doc_id.lower().startswith("daraz_"):
                source = "Daraz"
            elif doc_id.lower().startswith("startech_"):
                source = "StarTech"

        m = re.search(r"\*\*Category:\*\*\s*([^*]+)", part, re.IGNORECASE)
        category = m.group(1).strip() if m else None

        m = re.search(r"\*\*Price:\*\*\s*([^*]+)", part, re.IGNORECASE)
        price_value = _parse_price_value(m.group(1)) if m else None

        rating_avg, rating_cnt = None, None
        r = re.search(r"\*\*Rating:\*\*\s*([0-5](?:\.\d+)?)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", part, re.IGNORECASE)
        if r:
            try:
                rating_avg = float(r.group(1))
            except Exception:
                rating_avg = None
            try:
                rating_cnt = int(r.group(2)) if r.group(2) else None
            except Exception:
                rating_cnt = None

        bits = [title]
        meta = []
        if source: meta.append(f"Source: {source}")
        if category: meta.append(f"Category: {category}")
        if price_value is not None: meta.append(f"Price: ~৳{int(price_value)}")
        raw_md = "\n".join([title, " • ".join(meta)]) if meta else title

        products.append(ProductDoc(
            doc_id=doc_id, title=title, source=source, category=category,
            price_value=price_value, rating_avg=rating_avg, rating_cnt=rating_cnt,
            url=url, raw_md=raw_md
        ))
    return products

# ----------------------------
# Chunking (Chonkie)
# ----------------------------

def build_chunker(lang: str = DEFAULT_LANG) -> RecursiveChunker:
    return RecursiveChunker.from_recipe("markdown", lang=lang)

def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    chunks = []
    try:
        chonks = chunker(product.raw_md)
    except Exception:
        splits = [s.strip() for s in re.split(r"\n{2,}", product.raw_md) if s.strip()]
        chonks = [{"text": s} for s in splits]

    for c in chonks:
        text = (getattr(c, "text", None) or (c["text"] if isinstance(c, dict) else "")).strip()
        if not text:
            continue
        indexed_text = _clean_for_bm25(text)
        if not indexed_text:
            continue
        chunks.append(ChunkRec(
            doc_id=product.doc_id, title=product.title, source=product.source,
            category=product.category, price_value=product.price_value,
            rating_avg=product.rating_avg, rating_cnt=product.rating_cnt,
            url=product.url, text=indexed_text
        ))
    return chunks

# ----------------------------
# BM25 indexing
# ----------------------------

def build_or_load_bm25(products: List[ProductDoc], lang: str) -> Tuple[BM25Okapi, List[ChunkRec], List[List[str]]]:
    chunker = build_chunker(lang=lang)
    all_chunks: List[ChunkRec] = []
    for p in products:
        all_chunks.extend(product_to_chunks(p, chunker))

    content_sig = _sha1("\n".join([c.doc_id + "\t" + c.text for c in all_chunks]))
    sig = _sha1(f"v2combined|lang={lang}|{content_sig}")
    bm25_pkl, meta_pkl = _index_paths(sig)

    if os.path.exists(bm25_pkl) and os.path.exists(meta_pkl):
        with open(bm25_pkl, "rb") as f:
            bm25 = pickle.load(f)
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        return bm25, meta["chunks"], meta["tokenized_corpus"]

    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_pkl, "wb") as f:
        pickle.dump(bm25, f)
    with open(meta_pkl, "wb") as f:
        pickle.dump({"tokenized_corpus": tokenized_corpus, "chunks": all_chunks}, f)
    return bm25, all_chunks, tokenized_corpus

# ----------------------------
# Retrieval + filtering (shared)
# ----------------------------

def _passes_filters(
    chunk: ChunkRec,
    allowed_sources: Optional[set],
    allowed_categories: Optional[set],
    category_contains: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    rating_min: Optional[float],
) -> bool:
    if allowed_sources and (chunk.source not in allowed_sources):
        return False
    if allowed_categories and (chunk.category not in allowed_categories):
        return False
    if category_contains:
        cc = (chunk.category or "").lower()
        if category_contains.lower() not in cc:
            return False
    if price_min is not None and (chunk.price_value is not None) and (chunk.price_value < price_min):
        return False
    if price_max is not None and (chunk.price_value is not None) and (chunk.price_value > price_max):
        return False
    if rating_min is not None and (chunk.rating_avg is not None) and (chunk.rating_avg < rating_min):
        return False
    return True

def _parse_query_constraints(q: str) -> Dict[str, Optional[float]]:
    qn = q.lower().replace(",", "")
    price_min = None
    price_max = None
    rating_min = None
    source_hint = None

    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", qn)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        price_min, price_max = (min(a, b), max(a, b))
    m = re.search(r"(?:under|below|<=|less than)\s*(\d+(?:\.\d+)?)", qn)
    if m:
        price_max = float(m.group(1))
    m = re.search(r"(?:>=|at least)\s*(\d+(?:\.\d+)?)\s*(?:bdt|৳|tk|taka)?", qn)
    if m:
        price_min = max(price_min or 0.0, float(m.group(1)))

    m = re.search(r"rating\s*(?:>=|at least|of at least)?\s*([0-5](?:\.\d+)?)", qn)
    if m:
        rating_min = float(m.group(1))
    else:
        m = re.search(r"([0-5](?:\.\d+)?)\s*\+\s*rating", qn)
        if m:
            rating_min = float(m.group(1))
        else:
            m = re.search(r"(?:at least|minimum|min)\s*([0-5](?:\.\d+)?)\s*(?:stars|rating)", qn)
            if m:
                rating_min = float(m.group(1))

    if "daraz only" in qn or "only daraz" in qn:
        source_hint = "Daraz"
    elif "startech only" in qn or "only startech" in qn or "star tech" in qn:
        source_hint = "StarTech"

    return {"price_min": price_min, "price_max": price_max, "rating_min": rating_min, "source_hint": source_hint}

# ----------------------------
# BM25 search
# ----------------------------

def bm25_search(
    bm25: BM25Okapi,
    chunks: List[ChunkRec],
    tokenized_corpus: List[List[str]],
    query: str,
    top_k: int,
    allowed_sources: Optional[set] = None,
    allowed_categories: Optional[set] = None,
    category_contains: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    rating_min: Optional[float] = None,
    diversify: bool = True,
) -> List[Tuple[ChunkRec, float]]:

    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)

    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        c = chunks[i]
        if _passes_filters(c, allowed_sources, allowed_categories, category_contains, price_min, price_max, rating_min):
            pairs.append((i, float(sc)))

    q_words = set(q_tokens)
    def _boost(idx: int, s: float) -> float:
        c = chunks[idx]
        boost = 0.0
        title_words = set(_tokenize(c.title))
        if q_words & title_words:
            boost += 0.10 * s
        src_w = set(_tokenize(c.source or ""))
        if q_words & src_w:
            boost += 0.05 * s
        return s + boost

    pairs = [(i, _boost(i, s)) for (i, s) in pairs]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if not diversify:
        return [(chunks[i], s) for i, s in pairs[:top_k]]

    seen_docs = set()
    diversified: List[Tuple[ChunkRec, float]] = []
    for i, s in pairs:
        c = chunks[i]
        if c.doc_id in seen_docs:
            continue
        diversified.append((c, s))
        seen_docs.add(c.doc_id)
        if len(diversified) >= top_k:
            return diversified

    if len(diversified) < top_k:
        for i, s in pairs:
            c = chunks[i]
            diversified.append((c, s))
            if len(diversified) >= top_k:
                break
    return diversified

# ----------------------------
# OpenAI helpers
# ----------------------------

def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    ctx_blocks = []
    for i, (c, s) in enumerate(results, 1):
        head = f"[{i}] {c.title} — DocID: {c.doc_id}"
        if c.url:
            head += f" — {c.url}"
        fields = []
        if c.source: fields.append(f"Source: {c.source}")
        if c.category: fields.append(f"Category: {c.category}")
        if c.price_value is not None: fields.append(f"PriceValue: {int(c.price_value)}")
        if c.rating_avg is not None: fields.append(f"Rating: {c.rating_avg}/5")
        meta_line = " | ".join(fields)
        ctx_blocks.append(f"{head}\n{meta_line}\n---\n{c.text}\n")

    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Keep answers concise with bullets. "
        "Cite as [#] with DocID, and include URLs when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float = 0.2):
    client = _ensure_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

# ----------------------------
# RAPTOR: build + search
# ----------------------------

def _embed_texts(texts: List[str], model: str, batch: int = 256) -> np.ndarray:
    """Embed many texts; returns (n, d) numpy array."""
    client = _ensure_client()
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch):
        sub = texts[i:i+batch]
        ok = False
        for attempt in range(4):
            try:
                res = client.embeddings.create(model=model, input=sub)
                all_vecs.extend([d.embedding for d in res.data])
                ok = True
                break
            except Exception as e:
                time.sleep(1.5 * (attempt + 1))
        if not ok:
            # Fallback: zero vectors (keeps shape)
            dim = len(all_vecs[0]) if all_vecs else 1536
            all_vecs.extend([[0.0]*dim for _ in sub])
    return np.array(all_vecs, dtype=np.float32)

def _summarize_cluster_texts(texts: List[str], model: str, max_chars: int = 2500) -> str:
    """Abstractive summary for a cluster (concise, bullet-like)."""
    content = "\n\n".join(texts)
    content = content[:max_chars]
    client = _ensure_client()
    prompt = (
        "You will receive a set of short product snippets from a catalog cluster. "
        "Write a concise 3–6 bullet summary capturing shared themes, key specs, brands, "
        "typical price ranges if present, and differentiators. No fluff."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Fallback: extractive-like top lines
        lines = []
        for t in texts:
            for ln in t.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append("• " + ln)
                    break
        return "\n".join(lines[:6])

def _raptor_signature(chunks: List[ChunkRec], embed_model: str, summarizer_model: str,
                      cluster_size: int, max_depth: int) -> str:
    base = "\n".join([c.doc_id + "\t" + c.text for c in chunks])
    return _sha1(f"raptor|{embed_model}|{summarizer_model}|cs={cluster_size}|d={max_depth}|{_sha1(base)}")

def build_or_load_raptor(
    chunks: List[ChunkRec],
    embed_model: str,
    summarizer_model: str,
    cluster_size: int,
    max_depth: int,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dict index:
    {
        "leaf_embeddings": np.ndarray (n,d),
        "nodes_by_level": List[List[RaptorNode]],  # level 0 = leaves, last = top
        "id2node": Dict[str, RaptorNode],
        "params": {...}
    }
    """
    sig = _raptor_signature(chunks, embed_model, summarizer_model, cluster_size, max_depth)
    pkl = _raptor_path(sig)

    if (not rebuild) and os.path.exists(pkl):
        with open(pkl, "rb") as f:
            saved = pickle.load(f)
        # Convert embeddings back to numpy
        saved["leaf_embeddings"] = np.array(saved["leaf_embeddings"], dtype=np.float32)
        # nodes already contain embeddings as list; keep as list to save memory
        return saved

    # Build leaves
    leaf_texts = [c.text for c in chunks]
    with st.spinner("RAPTOR: embedding leaves…"):
        leaf_embeddings = _embed_texts(leaf_texts, embed_model)

    nodes_by_level: List[List[RaptorNode]] = []
    id2node: Dict[str, RaptorNode] = {}

    leaves: List[RaptorNode] = []
    for i in range(len(chunks)):
        nid = f"L{i}"
        node = RaptorNode(
            node_id=nid, level=0, children=[], text_ids=[i], size=1,
            summary=None, embedding=leaf_embeddings[i].tolist(), kind="leaf"
        )
        leaves.append(node)
        id2node[nid] = node
    nodes_by_level.append(leaves)

    # Build up the tree
    cur_nodes = leaves
    level = 1
    rng = np.random.RandomState(42)
    while level <= max_depth and len(cur_nodes) > max(cluster_size, 2):
        k = max(2, int(math.ceil(len(cur_nodes) / float(cluster_size))))
        X = np.array([n.embedding for n in cur_nodes], dtype=np.float32)

        with st.spinner(f"RAPTOR: clustering level {level} into ~{k} groups…"):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)

        groups: Dict[int, List[RaptorNode]] = {}
        for n, lab in zip(cur_nodes, labels):
            groups.setdefault(lab, []).append(n)

        parent_nodes: List[RaptorNode] = []
        with st.spinner(f"RAPTOR: summarizing {len(groups)} clusters…"):
            for gi, members in groups.items():
                # collect texts (prefer summaries of children if available; else leaf texts)
                children_ids = [m.node_id for m in members]
                # Gather a concise set of representative texts for summarization
                sample_texts: List[str] = []
                for m in members:
                    if m.kind == "leaf":
                        # include first line of leaf text
                        sample_texts.append(chunks[m.text_ids[0]].text.split("\n", 1)[0])
                    else:
                        sample_texts.append(m.summary or "")
                # summarize cluster
                summary = _summarize_cluster_texts(sample_texts, summarizer_model, max_chars=2500)
                # embed summary
                v = _embed_texts([summary], embed_model)[0].tolist()
                text_ids = []
                for m in members:
                    text_ids.extend(m.text_ids)
                nid = f"N{level}_{gi}_{rng.randint(10**6)}"
                node = RaptorNode(
                    node_id=nid, level=level, children=children_ids, text_ids=text_ids,
                    size=len(text_ids), summary=summary, embedding=v, kind="summary"
                )
                id2node[nid] = node
                parent_nodes.append(node)

        nodes_by_level.append(parent_nodes)
        cur_nodes = parent_nodes
        level += 1

    index = {
        "leaf_embeddings": leaf_embeddings,
        "nodes_by_level": nodes_by_level,
        "id2node": id2node,
        "params": {
            "embed_model": embed_model,
            "summarizer_model": summarizer_model,
            "cluster_size": cluster_size,
            "max_depth": max_depth,
            "sig": sig,
        }
    }

    # Save (store leaf embeddings and node embeddings as lists)
    save_obj = {
        "leaf_embeddings": leaf_embeddings.tolist(),
        "nodes_by_level": nodes_by_level,
        "params": index["params"],
    }
    with open(pkl, "wb") as f:
        pickle.dump(save_obj, f)

    return index

def raptor_search(
    raptor_index: Dict[str, Any],
    chunks: List[ChunkRec],
    query: str,
    top_k: int,
    allowed_sources: Optional[set] = None,
    allowed_categories: Optional[set] = None,
    category_contains: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    rating_min: Optional[float] = None,
    topn_per_level: int = RAPTOR_TOPN_PER_LEVEL_DEFAULT,
    diversify: bool = True,
) -> List[Tuple[ChunkRec, float]]:

    client = _ensure_client()
    q_vec = np.array(client.embeddings.create(model=raptor_index["params"]["embed_model"], input=[query]).data[0].embedding, dtype=np.float32)

    # Collect top nodes from each level
    nodes_by_level: List[List[RaptorNode]] = raptor_index["nodes_by_level"]
    leaf_embeddings: np.ndarray = raptor_index["leaf_embeddings"]

    leaf_candidates: set = set()

    # Traverse all levels (multi-level selection)
    for lvl in range(len(nodes_by_level)-1, -1, -1):
        nodes = nodes_by_level[lvl]
        if not nodes:
            continue
        M = np.array([n.embedding for n in nodes], dtype=np.float32)
        sims = _cosine_sim_matrix(q_vec, M)
        order = np.argsort(-sims)
        keep = order[:min(topn_per_level, len(order))]
        for idx in keep:
            n = nodes[idx]
            if n.level == 0 and n.kind == "leaf":
                leaf_candidates.update(n.text_ids)
            else:
                # Add the leaves under this node (could be many; cap softly)
                if len(n.text_ids) <= 50:
                    leaf_candidates.update(n.text_ids)
                else:
                    # take the first 50
                    for tid in n.text_ids[:50]:
                        leaf_candidates.add(tid)

    if not leaf_candidates:
        # Fallback: all leaves
        leaf_candidates = set(range(len(chunks)))

    cand_list = sorted(list(leaf_candidates))
    E = leaf_embeddings[cand_list, :]
    sims = _cosine_sim_matrix(q_vec, E)

    # Pair with filters
    pairs: List[Tuple[int, float]] = []
    for idx_in_list, score in enumerate(sims):
        i = cand_list[idx_in_list]
        c = chunks[i]
        if _passes_filters(c, allowed_sources, allowed_categories, category_contains, price_min, price_max, rating_min):
            pairs.append((i, float(score)))

    # Sort by similarity
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Diversify by doc (at most 1 chunk per product first)
    if diversify:
        seen_docs = set()
        diversified: List[Tuple[ChunkRec, float]] = []
        for i, s in pairs:
            c = chunks[i]
            if c.doc_id in seen_docs:
                continue
            diversified.append((c, s))
            seen_docs.add(c.doc_id)
            if len(diversified) >= top_k:
                return diversified

        # Allow repeats if still needed
        if len(diversified) < top_k:
            for i, s in pairs:
                c = chunks[i]
                diversified.append((c, s))
                if len(diversified) >= top_k:
                    break
        return diversified

    return [(chunks[i], s) for i, s in pairs[:top_k]]

# ----------------------------
# Hybrid fusion (BM25 + RAPTOR)
# ----------------------------

def _rrf_fusion(
    A: List[Tuple[ChunkRec, float]],
    B: List[Tuple[ChunkRec, float]],
    top_k: int,
    rrf_k: int = RRF_K_DEFAULT,
    diversify: bool = True
) -> List[Tuple[ChunkRec, float]]:
    """
    Reciprocal Rank Fusion on chunk-level results (by doc_id to limit duplicates).
    Score(doc) = sum(1 / (rrf_k + rank))
    """
    def to_rank_map(res: List[Tuple[ChunkRec, float]]) -> Dict[str, int]:
        m = {}
        for r, (c, _) in enumerate(res, 1):
            if c.doc_id not in m:
                m[c.doc_id] = r
        return m

    rankA = to_rank_map(A)
    rankB = to_rank_map(B)
    all_doc_ids = set(rankA) | set(rankB)

    # For each doc, pick the "best" chunk from either list to display
    doc2best: Dict[str, Tuple[ChunkRec, float]] = {}
    for res in (A, B):
        for (c, s) in res:
            prev = doc2best.get(c.doc_id)
            if (not prev) or (s > prev[1]):
                doc2best[c.doc_id] = (c, s)

    fused = []
    for d in all_doc_ids:
        r1 = rankA.get(d, 10**9)
        r2 = rankB.get(d, 10**9)
        score = 1.0 / (rrf_k + r1) + 1.0 / (rrf_k + r2)
        fused.append((d, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    out: List[Tuple[ChunkRec, float]] = []
    for d, s in fused:
        c, best_s = doc2best[d]
        out.append((c, s))
        if len(out) >= top_k:
            break
    return out

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="RAG: Daraz + StarTech (BM25 + RAPTOR)", layout="wide")
st.title("Daraz + StarTech Products RAG — BM25 + Chonkie + RAPTOR")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("OpenAI model (answering)", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    lang = st.selectbox("Chunk recipe language", ["en"], index=0)
    top_k = st.slider("Top-K chunks for answer", 1, 25, DEFAULT_TOPK)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    diversify = st.checkbox("Diversify (limit 1 chunk per product first)", value=True)

    st.markdown("---")
    st.markdown("### 📚 Corpus Source")
    st.caption("Leave blank to use the default URL, or provide a new raw URL below to override it.")
    remote_url_override = st.text_input(
        "Override Corpus URL",
        value="",
        placeholder=DEFAULT_CORPUS_URL
    )

    st.markdown("---")
    st.markdown("### 🦅 RAPTOR")
    retrieval_mode = st.radio(
        "Retrieval mode",
        ["BM25 only", "RAPTOR only (semantic)", "Hybrid (BM25 + RAPTOR)"],
        index=0
    )

    embed_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    summarizer_model = st.selectbox("Summarizer model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    raptor_cluster_size = st.slider("RAPTOR cluster size (≈children per parent)", 4, 16, RAPTOR_CLUSTER_SIZE_DEFAULT)
    raptor_max_depth = st.slider("RAPTOR max depth", 1, 5, RAPTOR_MAX_DEPTH_DEFAULT)
    raptor_topn_per_level = st.slider("RAPTOR top nodes per level", 2, 12, RAPTOR_TOPN_PER_LEVEL_DEFAULT)
    rrf_k = st.slider("Hybrid: RRF k constant", 10, 200, RRF_K_DEFAULT, help="Higher = softer fusion")
    rebuild_raptor = st.checkbox("Rebuild RAPTOR index (ignore cache)", value=False)

# Decide URL
md_text: Optional[str] = None
url_to_fetch = remote_url_override.strip() or DEFAULT_CORPUS_URL

if not url_to_fetch or "github.com/username/repo" in url_to_fetch:
    st.error("🚨 Please update `DEFAULT_CORPUS_URL` in the script to your real corpus URL.")
    st.info("Edit this file and set DEFAULT_CORPUS_URL to your raw GitHub link.")
    st.stop()

# Fetch
try:
    import requests
    with st.spinner(f"Fetching corpus from {url_to_fetch[:70]}..."):
        r = requests.get(url_to_fetch, timeout=30)
        if r.ok:
            md_text = r.text
        else:
            st.error(f"Failed to fetch corpus from {url_to_fetch}. HTTP Status: {r.status_code}")
except Exception as e:
    st.error(f"Error fetching data from {url_to_fetch}: {e}")

if not md_text:
    st.error("Corpus text could not be loaded. App cannot continue.")
    st.stop()

# Parse products
with st.spinner("Parsing combined corpus…"):
    products = parse_combined_products_from_md(md_text)

if not products:
    st.error("No products detected. Ensure entries look like: "
             "`## <Title> **DocID:** `<id>` **Source:** <...> **Category:** <...> **Price:** <...> ---`")
    st.stop()

# Build BM25
with st.spinner("Chunking (Chonkie) & building BM25 index…"):
    bm25, chunk_table, tokenized_corpus = build_or_load_bm25(products, lang=lang)

# Facets
all_sources = sorted({p.source for p in products if p.source})
all_categories = sorted({p.category for p in products if p.category})

st.success(f"Parsed **{len(products):,}** products → **{len(chunk_table):,}** chunks. BM25 index ready.")

st.markdown("#### Filters")
c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1.2, 1.2, 1.2])
with c1:
    sel_sources = st.multiselect("Source", options=all_sources, default=[])
with c2:
    sel_categories = st.multiselect("Category (exact)", options=all_categories, default=[])
with c3:
    cat_contains = st.text_input("Category contains", "")
with c4:
    price_max_ui = st.text_input("Max price (BDT)", "")
with c5:
    rating_min_ui = st.text_input("Min rating (0–5)", "")

def _to_float(x: str) -> Optional[float]:
    x = x.strip().replace(",", "")
    if not x:
        return None
    m = re.match(r"^\d+(?:\.\d+)?$", x)
    return float(x) if m else None

price_max_filter = _to_float(price_max_ui)
rating_min_filter = _to_float(rating_min_ui)

# Query UI
st.markdown("---")
query = st.text_input(
    "Ask about products (e.g., 'best wireless gamepad under 1500 startech only')",
    ""
)
go = st.button("Search")

with st.expander("Corpus breakdown", expanded=False):
    from collections import Counter
    source_counts = Counter(p.source or "Unknown" for p in products)
    st.write(dict(source_counts))
    category_counts = Counter(p.category or "Unknown" for p in products)
    st.write(dict(category_counts))

if go and query.strip():
    constraints = _parse_query_constraints(query)
    allowed_sources = set(sel_sources) if sel_sources else ({constraints["source_hint"]} if constraints["source_hint"] else None)
    allowed_categories = set(sel_categories) if sel_categories else None
    price_min = constraints["price_min"]
    price_max = price_max_filter if price_max_filter is not None else constraints["price_max"]
    rating_min = rating_min_filter if rating_min_filter is not None else constraints["rating_min"]

    # Always compute BM25 if needed by the selected mode
    results_bm25: List[Tuple[ChunkRec, float]] = []
    results_raptor: List[Tuple[ChunkRec, float]] = []

    if retrieval_mode in ["BM25 only", "Hybrid (BM25 + RAPTOR)"]:
        with st.spinner("Retrieving with BM25…"):
            results_bm25 = bm25_search(
                bm25, chunk_table, tokenized_corpus, query,
                top_k=top_k,
                allowed_sources=allowed_sources,
                allowed_categories=allowed_categories,
                category_contains=cat_contains.strip() or None,
                price_min=price_min,
                price_max=price_max,
                rating_min=rating_min,
                diversify=diversify,
            )

    if retrieval_mode in ["RAPTOR only (semantic)", "Hybrid (BM25 + RAPTOR)"]:
        with st.spinner("Preparing RAPTOR index… (first build is cached)"):
            raptor_index = build_or_load_raptor(
                chunk_table,
                embed_model=embed_model,
                summarizer_model=summarizer_model,
                cluster_size=raptor_cluster_size,
                max_depth=raptor_max_depth,
                rebuild=rebuild_raptor,
            )
        with st.spinner("Retrieving with RAPTOR…"):
            results_raptor = raptor_search(
                raptor_index,
                chunk_table,
                query,
                top_k=top_k,
                allowed_sources=allowed_sources,
                allowed_categories=allowed_categories,
                category_contains=cat_contains.strip() or None,
                price_min=price_min,
                price_max=price_max,
                rating_min=rating_min,
                topn_per_level=raptor_topn_per_level,
                diversify=diversify,
            )

    if retrieval_mode == "BM25 only":
        results = results_bm25
    elif retrieval_mode == "RAPTOR only (semantic)":
        results = results_raptor
    else:
        # Hybrid: RRF fusion on doc_id with best chunk
        results = _rrf_fusion(results_bm25, results_raptor, top_k=top_k, rrf_k=rrf_k, diversify=diversify)

    if not results:
        st.warning("No results matched your query/filters.")
        st.stop()

    # --- Answer
    st.subheader("Answer")
    messages = _build_messages(query, results)
    try:
        st.write_stream(stream_answer(model, messages, temperature=temperature))
    except Exception as e:
        st.error(f"OpenAI error: {e}")

    # --- Top matches (Context Used)
    with st.expander("View Top Matches (Context Used)", expanded=False):
        st.subheader("Top matches")
        for i, (chunk, score) in enumerate(results, 1):
            meta_bits = []
            if chunk.source: meta_bits.append(f"**Source:** {chunk.source}")
            if chunk.category: meta_bits.append(f"**Category:** {chunk.category}")
            if chunk.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(chunk.price_value)}")
            if chunk.rating_avg is not None:
                rc = f" ({chunk.rating_cnt} ratings)" if chunk.rating_cnt is not None else ""
                meta_bits.append(f"**Rating:** {chunk.rating_avg}/5{rc}")

            st.markdown(
                f"**[{i}] {chunk.title}** \n"
                f"DocID: `{chunk.doc_id}` • Score: `{score:.3f}`  \n"
                f"{'URL: ' + chunk.url if chunk.url else ''}  \n"
                + ("  \n".join(meta_bits) if meta_bits else "")
            )
            with st.expander("View chunk"):
                st.write(chunk.text)

    # Export matched items as JSON
    export_rows = []
    for i, (c, s) in enumerate(results, 1):
        export_rows.append({
            "rank": i,
            "score": s,
            "doc_id": c.doc_id,
            "title": c.title,
            "source": c.source or "",
            "url": c.url or "",
            "category": c.category or "",
            "price_value": c.price_value if c.price_value is not None else "",
            "rating_avg": c.rating_avg if c.rating_avg is not None else "",
            "rating_cnt": c.rating_cnt if c.rating_cnt is not None else "",
            "chunk_text": c.text[:2000],
        })
    export_bytes = io.BytesIO()
    export_bytes.write(json.dumps(export_rows, ensure_ascii=False, indent=2).encode("utf-8"))
    export_bytes.seek(0)
    st.download_button("Download results (JSON)", data=export_bytes, file_name="results.json", mime="application/json")
