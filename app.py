"""
Unified RAG Application for Daraz & StarTech
==============================================
This Streamlit app reads a pre-built, combined corpus file.
It assumes a separate, scheduled process (GitHub Actions) is
running scrapers for Daraz and StarTech and then building
a single corpus file named 'combined_corpus.md'.

Features:
- Auto-refreshes data every hour (ttl=3600).
- Filters by Source (Daraz, StarTech), Category, Price, and Rating.
- Uses OpenAI to answer natural language queries based on retrieved data.
"""
import os
import re
import io
import json
import pickle
import hashlib
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# --- App Configuration ---
# This app reads the final output of your build_combined_corpus.py script
CORPUS_PATH = Path("out/combined_corpus.md")
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 8
DEFAULT_LANG = "en"

# --- Data Structures for Unified Products ---
@dataclass
class ProductDoc:
    """Represents a single product from any source after parsing."""
    doc_id: str
    title: str
    url: Optional[str]
    category: Optional[str]
    brand: Optional[str]
    price_value: Optional[float]
    rating_avg: Optional[float]
    raw_md: str
    source: Optional[str] = None

@dataclass
class ChunkRec:
    """Represents a single searchable chunk derived from a ProductDoc."""
    doc_id: str
    title: str
    text: str
    url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price_value: Optional[float] = None
    rating_avg: Optional[float] = None
    source: Optional[str] = None

# --- Regex & Parsing Logic ---
# These patterns parse the structured markdown created by build_combined_corpus.py
DOC_BLOCK_RE = re.compile(r"<!--DOC:START(?P<attrs>[^>]*)-->(?P<body>.*?)<!--DOC:END-->", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*([^\s\n]+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    """Parses key=value attributes from the DOC:START comment."""
    out = {}
    for kv in attrs.strip().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip().replace("'", "").replace('"', '')
    return out

def _parse_price_value(s: str) -> Optional[float]:
    """Extracts a numeric value from a price string (e.g., '৳1,999' -> 1999.0)."""
    if not s: return None
    s = s.replace(",", "").replace("৳", "").strip()
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try:
        return float(nums[0])
    except (ValueError, IndexError):
        return None

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    """Parses the entire combined_corpus.md file into a list of ProductDoc objects."""
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""
        body = (m.group("body") or "").strip()
        meta = _meta_from_header(attrs)
        
        doc_id = meta.get("id") or f"doc_{len(products)+1}"
        
        # Extract fields using regex from the markdown body
        title_m = TITLE_RE.search(body)
        title = title_m.group(1).strip() if title_m else f"Product {doc_id}"
        
        url_m = URL_LINE_RE.search(body)
        url = url_m.group(1).strip() if url_m else None
        
        brand_m = BRAND_RE.search(body)
        brand = brand_m.group(1).strip() if brand_m else None
        
        price_m = PRICE_RE.search(body)
        price_str = price_m.group(1).strip() if price_m else None
        price_value = _parse_price_value(price_str)
        
        rating_m = RATING_RE.search(body)
        rating_avg = float(rating_m.group(1)) if rating_m else None
        
        products.append(ProductDoc(
            doc_id=doc_id,
            title=title,
            url=url,
            category=meta.get("category"),
            brand=brand,
            price_value=price_value,
            rating_avg=rating_avg,
            source=meta.get("source"),
            raw_md=body
        ))
    return products

# --- Data Loading with Auto-Refresh ---
@st.cache_data(ttl=3600, show_spinner="Loading fresh product data...")
def load_data_and_time(path: Path) -> Tuple[List[ProductDoc], datetime.datetime]:
    """
    Reads the corpus file from disk and gets its modification time.
    Streamlit caches this function's result for 1 hour (3600s).
    """
    try:
        md_text = path.read_text(encoding="utf-8")
        products = parse_products_from_md(md_text)
        mod_time_stamp = os.path.getmtime(path)
        last_updated = datetime.datetime.fromtimestamp(mod_time_stamp)
        return products, last_updated
    except FileNotFoundError:
        st.error(f"Corpus file not found: {path}. Waiting for the automated pipeline to run...")
        return [], datetime.datetime.now()
    except Exception as e:
        st.error(f"Error loading corpus: {e}")
        return [], datetime.datetime.now()

# --- Chunking & BM25 Indexing ---
def _tokenize(text: str) -> List[str]:
    """Minimal tokenizer for BM25."""
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    # You can expand stopwords if you wish
    stopwords = set(["the", "a", "an", "is", "in", "of", "for"])
    return [t for t in toks if t not in stopwords]

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

@st.cache_data(show_spinner="Building search index...")
def build_bm25_index(products: List[ProductDoc]) -> Tuple[BM25Okapi, List[ChunkRec]]:
    """Builds the BM25 index. This is cached and only re-runs if `products` changes."""
    chunker = RecursiveChunker.from_recipe("markdown", lang=DEFAULT_LANG)
    all_chunks: List[ChunkRec] = []
    
    for product in products:
        # We chunk the raw markdown body of each product
        chonks = chunker(product.raw_md)
        for c in chonks:
            text = getattr(c, 'text', '').strip()
            if text:
                all_chunks.append(ChunkRec(
                    doc_id=product.doc_id,
                    title=product.title,
                    text=text,
                    url=product.url,
                    category=product.category,
                    brand=product.brand,
                    price_value=product.price_value,
                    rating_avg=product.rating_avg,
                    source=product.source
                ))

    # Build the BM25 index from the tokenized text of each chunk
    tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_chunks

# --- Search Logic ---
def _passes_filters(chunk: ChunkRec, **filters) -> bool:
    """Checks if a chunk passes all active user filters."""
    if filters.get("allowed_sources") and (chunk.source not in filters["allowed_sources"]):
        return False
    if filters.get("allowed_categories") and (chunk.category not in filters["allowed_categories"]):
        return False
    if filters.get("brand_filter") and (filters["brand_filter"].lower() not in (chunk.brand or "").lower()):
        return False
    price_max = filters.get("price_max")
    if price_max is not None and chunk.price_value is not None and chunk.price_value > price_max:
        return False
    rating_min = filters.get("rating_min")
    if rating_min is not None and chunk.rating_avg is not None and chunk.rating_avg < rating_min:
        return False
    return True

def bm25_search(bm25: BM25Okapi, chunks: List[ChunkRec], query: str, top_k: int, **filters) -> List[Tuple[ChunkRec, float]]:
    """Performs BM25 search, applies filters, and returns top_k results."""
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    
    pairs: List[Tuple[int, float]] = []
    for i, score in enumerate(scores):
        if _passes_filters(chunks[i], **filters):
            pairs.append((i, float(score)))
            
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Simple diversification: ensure we don't show too many chunks from the same product
    diversified_results = []
    seen_doc_ids = set()
    for i, score in pairs:
        if len(diversified_results) >= top_k:
            break
        if chunks[i].doc_id not in seen_doc_ids:
            diversified_results.append((chunks[i], score))
            seen_doc_ids.add(chunks[i].doc_id)
            
    return diversified_results

# --- OpenAI Integration ---
def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    """Builds the prompt for the OpenAI API with retrieved context."""
    ctx_blocks = []
    for i, (chunk, score) in enumerate(results, 1):
        header = f"[{i}] ({chunk.source}) {chunk.title} — DocID: {chunk.doc_id}"
        url_line = f"URL: {chunk.url}" if chunk.url else ""
        ctx_blocks.append(f"{header}\n{url_line}\n---\n{chunk.text}\n")
    
    system_prompt = (
        "You are a helpful e-commerce assistant for users in Bangladesh. "
        "Answer the user's question based *only* on the provided context blocks. "
        "If the information is not in the context, say 'I do not have that information in the provided data.' "
        "Present your answer clearly, using bullet points for lists of products. "
        "Cite your sources at the end of relevant sentences using the format `[#]`."
    )
    user_prompt = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks)
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float):
    """Streams the OpenAI response to the Streamlit UI."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY secret.")
        
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception as e:
        yield f"\n\n**Error communicating with OpenAI:** {e}"

# --- Main Streamlit UI ---
st.set_page_config(page_title="Daraz & StarTech RAG", layout="wide")
st.title("Daraz & StarTech Product Search 🛒")

# 1. Load data and get last updated time
products, last_updated = load_data_and_time(CORPUS_PATH)
if not products:
    st.warning("No product data found. The application will be ready once the data pipeline completes its first run.")
    st.stop()

# 2. Build the search index (will be cached)
bm25, chunk_table = build_bm25_index(products)

# 3. Get unique values for filters from the loaded data
all_sources = sorted({p.source for p in products if p.source})
all_categories = sorted({p.category for p in products if p.category})

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4-turbo"], index=0)
    top_k = st.slider("Number of Results", 1, 15, DEFAULT_TOPK)
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.1, 0.1)
    
    st.markdown("---")
    st.markdown(f"**Data Status**")
    st.info(f"Updated: {last_updated.strftime('%b %d, %Y %I:%M %p')}")

st.success(f"**Ready!** Loaded {len(products):,} products from {len(all_sources)} sources.")

# --- Main Page Filters ---
st.markdown("#### Filters")
filter_cols = st.columns(5)
with filter_cols[0]:
    sel_sources = st.multiselect("Source", options=all_sources, default=all_sources)
with filter_cols[1]:
    sel_categories = st.multiselect("Category", options=all_categories, default=[])
with filter_cols[2]:
    brand_filter = st.text_input("Brand contains", "")
with filter_cols[3]:
    price_max_ui = st.text_input("Max Price (৳)", "")
with filter_cols[4]:
    rating_min_ui = st.text_input("Min Rating (1-5)", "")

def _to_float(x: str) -> Optional[float]:
    x = x.strip().replace(",", "");
    if not x: return None
    try: return float(x)
    except ValueError: return None

# --- Search Box & Execution ---
st.markdown("---")
query = st.text_input("Ask about products (e.g., 'gaming laptop under 1 lakh with good graphics card')", "")
go = st.button("Search")

if go and query.strip():
    filters = {
        "allowed_sources": set(sel_sources),
        "allowed_categories": set(sel_categories),
        "brand_filter": brand_filter.strip(),
        "price_max": _to_float(price_max_ui),
        "rating_min": _to_float(rating_min_ui),
    }
    
    with st.spinner("Searching through products..."):
        results = bm25_search(bm25, chunk_table, query, top_k=top_k, **filters)

    if not results:
        st.warning("No products matched your search criteria.")
        st.stop()

    # --- Display Results ---
    colL, colR = st.columns([0.6, 0.4], gap="large")
    with colL:
        st.subheader(f"Top {len(results)} Matches")
        for i, (chunk, score) in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. ({chunk.source}) {chunk.title}**")
                meta_parts = [
                    f"**Category:** {chunk.category}" if chunk.category else None,
                    f"**Brand:** {chunk.brand}" if chunk.brand else None,
                    f"**Price:** ৳{int(chunk.price_value)}" if chunk.price_value is not None else None,
                ]
                st.caption(" | ".join(filter(None, meta_parts)))
                if chunk.url:
                    st.markdown(f"[View Product]({chunk.url})")
                with st.expander("View Retrieved Text"):
                    st.markdown(chunk.text)
    
    with colR:
        st.subheader("🤖 AI Generated Answer")
        with st.container(border=True):
            messages = _build_messages(query, results)
            st.write_stream(stream_answer(model, messages, temperature))
