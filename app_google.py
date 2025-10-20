# -*- coding: utf-8 -*-
"""
Daraz + StarTech Products RAG — Chonkie + BM25 (Streamlit App)
==============================================================
*** GEMINI VERSION (Fixed) ***

What this file does (high level):
- Loads your combined corpus (Daraz + StarTech) from a default raw GitHub URL.
- ...
- Streams a strictly grounded LLM answer (Google Gemini). Citations reference DocIDs.

Dependencies (pip):
    streamlit, python-dotenv, google-generativeai, rank_bm25, chonkie, requests

Environment:
    export GEMINI_API_KEY="your-api-key..."
"""

import os
import re
import io
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# App Config
# ----------------------------
INDEX_DIR = "index"                             # local cache folder for BM25 + metadata
os.makedirs(INDEX_DIR, exist_ok=True)

# --- !!! IMPORTANT: SET YOUR DEFAULT URL HERE !!! ---
DEFAULT_CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/refs/heads/main/out/combined_corpus.md"
# ---

DEFAULT_MODEL = "gemini-2.5-flash"              # <-- MODIFIED (Removed -latest)
DEFAULT_TOPK = 10                               # how many chunks for LLM grounding
DEFAULT_LANG = "en"                             # Chonkie recipe language

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ProductDoc:
    """
    A single product parsed from the combined corpus.
    ... (rest of class is unchanged) ...
    """
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
    """A single search chunk with light metadata for filtering & display."""
    doc_id: str
    title: str
    source: Optional[str]
    category: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    rating_cnt: Optional[int]
    url: Optional[str]
    text: str

# ----------------------------
# Regex helpers
# ... (all regex and utility functions _sha1, _index_paths, _parse_price_value,
#      _clean_for_bm25, _tokenize are unchanged) ...
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


# --- MODIFIED: Gemini Client Configuration ---
def _ensure_client() -> genai.GenerativeModel:
    """Configures and returns a Gemini GenerativeModel client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    # This function will now be called *with* the model name
    return genai.GenerativeModel
# ---

# ----------------------------
# Parsing combined corpus
# ... (parse_combined_products_from_md is unchanged) ...
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
# ... (build_chunker, product_to_chunks are unchanged) ...
# ----------------------------
def build_chunker(lang: str = DEFAULT_LANG) -> RecursiveChunker:
    return RecursiveChunker.from_recipe("markdown", lang=lang)

def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    chunks = []
    try:
        chonks = chunker(product.raw_md)  # typically 1 short chunk
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
# ... (build_or_load_bm25 is unchanged) ...
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
# Retrieval + filtering
# ... (_passes_filters, _parse_query_constraints, bm25_search are unchanged) ...
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
# Gemini helpers (MODIFIED)
# ----------------------------

def _build_gemini_prompt(query: str, results: List[Tuple[ChunkRec, float]]) -> str:
    """
    Builds a single, long prompt string for Gemini.
    """
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
    
    # Combine system and user prompt into one
    system = (
        "You are a precise product assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say you don't know. Keep answers concise with bullets. "
        "Cite as [#] with DocID, and include URLs when available."
    )
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    
    return f"{system}\n\n{user}"


def stream_answer(model_name: str, prompt: str, temperature: float = 0.2):
    """
    Streams an answer from the Gemini API.
    """
    client_provider = _ensure_client()
    model = client_provider(model_name)
    
    # Configure generation
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_k=1,
        top_p=1.0
    )
    
    # Configure safety (to be less restrictive)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        resp = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=True
        )
        
        for chunk in resp:
            # Check for text in the chunk
            if chunk.parts:
                yield chunk.parts[0].text
            # Handle potential empty chunks or finish reasons
            elif chunk.candidates and chunk.candidates[0].finish_reason:
                pass # Finished
            
    except Exception as e:
        # Handle potential safety blocks or other errors
        st.error(f"Gemini API Error: {e}")
        yield f"An error occurred while generating the response: {e}"


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="RAG: Daraz + StarTech (BM25 + Gemini)", layout="wide")
st.title("Daraz + StarTech Products RAG — Recursive Chunking + BM25 (Gemini)")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    # --- MODIFIED: Model list (Removed -latest) ---
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-2.0-flash-lite",], index=0)
    lang = st.selectbox("Chunk recipe language", ["en"], index=0)
    top_k = st.slider("Top-K chunks", 1, 25, DEFAULT_TOPK)
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

# ... (Corpus loading logic is unchanged) ...
md_text: Optional[str] = None
url_to_fetch = remote_url_override.strip() or DEFAULT_CORPUS_URL
if not url_to_fetch or "github.com/username/repo" in url_to_fetch:
    st.error("🚨 Please update `DEFAULT_CORPUS_URL` in the script to your real corpus URL.")
    st.info("You must edit the Python file and set this constant to your raw GitHub link.")
    st.stop()
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


# ... (Parsing, BM25 building, Facets, and Filters logic is unchanged) ...
with st.spinner("Parsing combined corpus…"):
    products = parse_combined_products_from_md(md_text)
if not products:
    st.error("No products detected. Ensure entries look like: "
             "`## <Title> **DocID:** `<id>` **Source:** <...> **Category:** <...> **Price:** <...> ---`")
    st.stop()
with st.spinner("Chunking (Chonkie) & building BM25 index…"):
    bm25, chunk_table, tokenized_corpus = build_or_load_bm25(products, lang=lang)
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

# ... (Query UI and Corpus breakdown is unchanged) ...
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'best wireless gamepad under 1500 startech only')", "")
go = st.button("Search")
with st.expander("Corpus breakdown", expanded=False):
    from collections import Counter
    source_counts = Counter(p.source or "Unknown" for p in products)
    st.write(dict(source_counts))
    category_counts = Counter(p.category or "Unknown" for p in products)
    st.write(dict(category_counts))


if go and query.strip():
    # ... (Search logic is unchanged) ...
    constraints = _parse_query_constraints(query)
    allowed_sources = set(sel_sources) if sel_sources else ( {constraints["source_hint"]} if constraints["source_hint"] else None )
    allowed_categories = set(sel_categories) if sel_categories else None
    price_min = constraints["price_min"]
    price_max = price_max_filter if price_max_filter is not None else constraints["price_max"]
    rating_min = rating_min_filter if rating_min_filter is not None else constraints["rating_min"]
    with st.spinner("Retrieving with BM25…"):
        results = bm25_search(
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
    if not results:
        st.warning("No results matched your query/filters.")
        st.stop()

    # --- MODIFIED: Swapped OpenAI logic for Gemini logic ---
    
    st.subheader("Answer")
    # 1. Build the single Gemini prompt
    prompt = _build_gemini_prompt(query, results)
    
    try:
        # 2. Call the new Gemini stream_answer function
        st.write_stream(stream_answer(model, prompt, temperature=temperature))
    except Exception as e:
        st.error(f"Gemini error: {e}")

    # The "View Top Matches" expander is unchanged, as requested
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
    
    # --- END OF MODIFICATION ---

    # ... (Export logic is unchanged) ...
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