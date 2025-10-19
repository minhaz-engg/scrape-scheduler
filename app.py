import os
import re
import io
import json
import pickle
import hashlib
import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv
load_dotenv()

# --- App Config ---
CORPUS_PATH = Path("out/daraz_products_corpus.md")
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 8
DEFAULT_LANG = "en"

# --- Data Structures (same as your original) ---
@dataclass
class ProductDoc:
    doc_id: str; title: str; url: Optional[str]; category: Optional[str]
    brand: Optional[str]; price_value: Optional[float]; rating_avg: Optional[float]
    rating_cnt: Optional[int]; raw_md: str

@dataclass
class ChunkRec:
    doc_id: str; title: str; url: Optional[str]; category: Optional[str]
    brand: Optional[str]; price_value: Optional[float]; rating_avg: Optional[float]
    rating_cnt: Optional[int]; text: str

# --- Regex & Parsing (same as your original) ---
DOC_BLOCK_RE = re.compile(r"(?P<body>.*?)", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*(.+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)\s*/\s*5(?:\s*\((\d+)\s*ratings\))?", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    out = {};
    for kv in attrs.strip().split():
        if "=" in kv: k, v = kv.split("=", 1); out[k.strip()] = v.strip()
    return out

def _parse_price_value(s: str) -> Optional[float]:
    s = s.replace(",", ""); nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try: vals = [float(x) for x in nums]; return min(vals) if vals else None
    except Exception: return None

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""; body = (m.group("body") or "").strip()
        meta = _meta_from_header(attrs)
        doc_id = meta.get("id") or f"doc_{len(products)+1}"
        category = meta.get("category")
        title_m = TITLE_RE.search(body); title = (title_m.group(1).strip() if title_m else f"Product {doc_id}")
        url_m = URL_LINE_RE.search(body); url = url_m.group(1).strip() if url_m else None
        brand_m = BRAND_RE.search(body); brand = brand_m.group(1).strip() if brand_m else None
        price_m = PRICE_RE.search(body); price_value = _parse_price_value(price_m.group(1)) if price_m else None
        rating_m = RATING_RE.search(body); rating_avg, rating_cnt = None, None
        if rating_m:
            try: rating_avg = float(rating_m.group(1))
            except Exception: pass
            try: rating_cnt = int(rating_m.group(2)) if rating_m.group(2) else None
            except Exception: pass
        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url, category=category, brand=brand, 
            price_value=price_value, rating_avg=rating_avg, rating_cnt=rating_cnt, raw_md=body
        ))
    return products

# --- ⭐️ REQUIREMENT 1: Auto-refreshing data loader ---
@st.cache_data(ttl=3600, show_spinner="Loading fresh product data...") # 1-hour cache
def load_data_and_time(path: Path) -> Tuple[List[ProductDoc], datetime.datetime]:
    """
    Reads the corpus from disk and gets its modification time.
    Streamlit caches this for 1 hour (3600s).
    """
    try:
        md_text = path.read_text(encoding="utf-8")
        products = parse_products_from_md(md_text)
        mod_time_stamp = os.path.getmtime(path)
        last_updated = datetime.datetime.fromtimestamp(mod_time_stamp)
        return products, last_updated
    except FileNotFoundError:
        st.error(f"Corpus file not found at {path}. Waiting for pipeline to run...")
        return [], datetime.datetime.now()
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return [], datetime.datetime.now()

# --- Chunking & BM25 (no changes needed) ---
def build_chunker(lang: str) -> RecursiveChunker:
    return RecursiveChunker.from_recipe("markdown", lang=lang)

def _clean_for_bm25(text: str) -> str:
    clean_lines = []
    for line in text.splitlines():
        ll = line.strip()
        if not ll or ll.lower().startswith("**images"): continue
        if "http://" in ll or "https://" in ll:
            ll = " ".join([p for p in re.split(r"\s+https?://\S+", ll) if p.strip()])
            if not ll: continue
        clean_lines.append(ll)
    return "\n".join(clean_lines)

def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    chunks = []
    try: chonks = chunker(product.raw_md)
    except Exception: chonks = [{"text": s} for s in re.split(r"\n{2,}", product.raw_md) if s.strip()]
    for c in chonks:
        text = (getattr(c, "text", None) or (c["text"] if isinstance(c, dict) else "")).strip()
        if not text: continue
        indexed_text = _clean_for_bm25(text)
        if not indexed_text: continue
        chunks.append(ChunkRec(
            doc_id=product.doc_id, title=product.title, url=product.url,
            category=product.category, brand=product.brand, price_value=product.price_value,
            rating_avg=product.rating_avg, rating_cnt=product.rating_cnt, text=indexed_text
        ))
    return chunks

STOPWORDS = set(["the","a","an","and","or","of","for","on","in","to","from","with","by","at","is","are","was","were"])
def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower()); return [t for t in toks if t not in STOPWORDS]
def _sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()
def _index_paths(sig: str) -> Tuple[str, str]:
    return (os.path.join(INDEX_DIR, f"bm25_{sig}.pkl"), os.path.join(INDEX_DIR, f"meta_{sig}.pkl"))

@st.cache_data(show_spinner="Building search index...")
def build_bm25_index(products: List[ProductDoc], lang: str) -> Tuple[BM25Okapi, List[ChunkRec]]:
    """Builds BM25 index, cached based on the product list."""
    chunker = build_chunker(lang=lang)
    all_chunks: List[ChunkRec] = []
    for p in products:
        all_chunks.extend(product_to_chunks(p, chunker))
    
    content_sig = _sha1(json.dumps([p.doc_id for p in products])) # Simple hash
    sig = _sha1(f"v1|lang={lang}|{content_sig}")
    bm25_pkl, meta_pkl = _index_paths(sig)

    if os.path.exists(bm25_pkl) and os.path.exists(meta_pkl):
        with open(bm25_pkl, "rb") as f: bm25 = pickle.load(f)
        with open(meta_pkl, "rb") as f: meta = pickle.load(f)
        return bm25, meta["chunks"]
        
    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(bm25_pkl, "wb") as f: pickle.dump(bm25, f)
    with open(meta_pkl, "wb") as f: pickle.dump({"chunks": all_chunks}, f)
    return bm25, all_chunks

# --- Search & OpenAI (no changes needed) ---
def _passes_filters(chunk: ChunkRec, **filters) -> bool:
    if filters.get("allowed_categories") and (chunk.category not in filters["allowed_categories"]): return False
    if filters.get("brand_filter") and (filters["brand_filter"].lower() not in (chunk.brand or "").lower()): return False
    if filters.get("price_min") is not None and (chunk.price_value is not None) and (chunk.price_value < filters["price_min"]): return False
    if filters.get("price_max") is not None and (chunk.price_value is not None) and (chunk.price_value > filters["price_max"]): return False
    if filters.get("rating_min") is not None and (chunk.rating_avg is not None) and (chunk.rating_avg < filters["rating_min"]): return False
    return True

def _parse_query_constraints(q: str) -> Dict[str, Optional[float]]:
    qn = q.lower().replace(",", ""); price_min, price_max, rating_min = None, None, None
    m = re.search(r"between\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)", qn)
    if m: price_min, price_max = min(float(m.group(1)), float(m.group(2))), max(float(m.group(1)), float(m.group(2)))
    m = re.search(r"(?:under|below|<=|less than)\s*(\d+(?:\.\d+)?)", qn);
    if m: price_max = float(m.group(1))
    m = re.search(r"(?:>=|at least)\s*(\d+(?:\.\d+)?)\s*(?:bdt|৳|tk|taka)?", qn)
    if m: price_min = max(price_min or 0.0, float(m.group(1)))
    m = re.search(r"rating\s*(?:>=|at least|of at least)?\s*([0-5](?:\.\d+)?)", qn);
    if m: rating_min = float(m.group(1))
    else:
        m = re.search(r"([0-5](?:\.\d+)?)\s*\+\s*rating", qn);
        if m: rating_min = float(m.group(1))
        else:
            m = re.search(r"(?:at least|minimum|min)\s*([0-5](?:\.\d+)?)\s*(?:stars|rating)", qn);
            if m: rating_min = float(m.group(1))
    return {"price_min": price_min, "price_max": price_max, "rating_min": rating_min}

def bm25_search(bm25: BM25Okapi, chunks: List[ChunkRec], query: str, top_k: int, **filters) -> List[Tuple[ChunkRec, float]]:
    q_tokens = _tokenize(query); scores = bm25.get_scores(q_tokens)
    pairs: List[Tuple[int, float]] = []
    for i, sc in enumerate(scores):
        if _passes_filters(chunks[i], **filters): pairs.append((i, float(sc)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    if not filters.get("diversify", True): return [(chunks[i], s) for i, s in pairs[:top_k]]
    
    seen_docs = set(); diversified: List[Tuple[ChunkRec, float]] = []
    for i, s in pairs:
        c = chunks[i]
        if c.doc_id not in seen_docs:
            diversified.append((c, s)); seen_docs.add(c.doc_id)
            if len(diversified) >= top_k: return diversified
    return diversified

def _ensure_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("Missing OPENAI_API_KEY env variable.")
    return OpenAI()

def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    ctx_blocks = []
    for i, (c, s) in enumerate(results, 1):
        head = f"[{i}] {c.title} — DocID: {c.doc_id}" + (f" — {c.url}" if c.url else "")
        fields = [f"Brand: {c.brand}" if c.brand else None, f"Category: {c.category}" if c.category else None,
                  f"PriceValue: {int(c.price_value)}" if c.price_value is not None else None,
                  f"Rating: {c.rating_avg}/5" if c.rating_avg is not None else None]
        meta_line = " | ".join(filter(None, fields))
        ctx_blocks.append(f"{head}\n{meta_line}\n---\n{c.text}\n")
    system = ("You are a precise product assistant. Answer ONLY from the provided context. "
              "If the answer isn't present, say 'I do not have that information.' "
              "Cite as [#] with DocID and include URL when available.")
    user = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def stream_answer(model: str, messages: List[Dict[str, str]], temp: float):
    client = _ensure_client()
    resp = client.chat.completions.create(model=model, temperature=temp, messages=messages, stream=True)
    for chunk in resp:
        delta = chunk.choices[0].delta.content or ""
        if delta: yield delta

# --- Streamlit UI ---
st.set_page_config(page_title="Daraz RAG (Auto-Refresh)", layout="wide")
st.title("Daraz Products RAG  ऑटो-रिफ्रेश 🔄") # Hindi "Auto-Refresh"

# 1. Load data and last-updated time
products, last_updated = load_data_and_time(CORPUS_PATH)

if not products:
    st.stop() # Wait for data to appear

# 2. Build the search index
bm25, chunk_table = build_bm25_index(products, DEFAULT_LANG)

# 3. ⭐️ REQUIREMENT 2: Get categories and show them
all_categories = sorted({p.category for p in products if p.category})
all_brands = sorted({(p.brand or "").strip() for p in products if p.brand})

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    top_k = st.slider("Top-K results", 1, 20, DEFAULT_TOPK)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    diversify = st.checkbox("Diversify results", value=True)
    
    st.markdown("---")
    st.markdown(f"**Status:** `Data updated`")
    # ⭐️ REQUIREMENT 1: Show last updated time
    st.markdown(f"`{last_updated.strftime('%Y-%m-%d %I:%M %p')}`")

st.success(f"App is ready. Loaded **{len(products):,}** products from {len(all_categories)} categories.")

# --- Main Page Filters ---
st.markdown("#### Filters")
ncols = st.columns([1.7, 1.3, 1.2, 1.2])
with ncols[0]:
    # ⭐️ REQUIREMENT 2: Show available categories
    sel_categories = st.multiselect("Available Categories", options=all_categories, default=[])
with ncols[1]: brand_filter = st.text_input("Brand contains", "")
with ncols[2]: price_max_ui = st.text_input("Max price (BDT)", "")
with ncols[3]: rating_min_ui = st.text_input("Min rating (0-5)", "")

def _to_float(x: str) -> Optional[float]:
    x = x.strip().replace(",", "");
    if not x: return None
    m = re.match(r"^\d+(?:\.\d+)?$", x); return float(x) if m else None

# --- Search Box ---
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'best bedding set under 2000')", "")
go = st.button("Search")

if go and query.strip():
    constraints = _parse_query_constraints(query)
    filters = {
        "allowed_categories": set(sel_categories) if sel_categories else None,
        "brand_filter": brand_filter if brand_filter.strip() else None,
        "price_min": constraints["price_min"],
        "price_max": _to_float(price_max_ui) if _to_float(price_max_ui) is not None else constraints["price_max"],
        "rating_min": _to_float(rating_min_ui) if _to_float(rating_min_ui) is not None else constraints["rating_min"],
        "diversify": diversify
    }
    
    with st.spinner("Searching..."):
        results = bm25_search(bm25, chunk_table, query, top_k=top_k, **filters)

    if not results:
        st.warning("No results matched your query/filters."); st.stop()

    # --- Display Results ---
    colL, colR = st.columns([0.55, 0.45], gap="large")
    with colL:
        st.subheader("Top Matches")
        for i, (chunk, score) in enumerate(results, 1):
            meta_bits = []
            if chunk.brand: meta_bits.append(f"**Brand:** {chunk.brand}")
            if chunk.category: meta_bits.append(f"**Category:** {chunk.category}")
            if chunk.price_value is not None: meta_bits.append(f"**Price:** ~৳{int(chunk.price_value)}")
            if chunk.rating_avg is not None:
                rc = f" ({chunk.rating_cnt} ratings)" if chunk.rating_cnt else ""
                meta_bits.append(f"**Rating:** {chunk.rating_avg}/5{rc}")
            
            st.markdown(f"**[{i}] {chunk.title}** \n"
                        f"DocID: `{chunk.doc_id}` • Score: `{score:.3f}`  \n"
                        f"{'URL: ' + chunk.url if chunk.url else ''}  \n" +
                        ("  \n".join(filter(None, meta_bits))))
            with st.expander("View chunk"): st.write(chunk.text)
    
    with colR:
        st.subheader("Answer (from OpenAI)")
        messages = _build_messages(query, results)
        try:
            st.write_stream(stream_answer(model, messages, temp=temperature))
        except Exception as e:
            st.error(f"OpenAI error: {e}")