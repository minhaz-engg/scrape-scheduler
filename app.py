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
import requests # <--- ADD THIS IMPORT

import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chonkie import RecursiveChunker
from dotenv import load_dotenv

load_dotenv()

# --- App Configuration ---
# --- 👇 USE THE RAW GITHUB URL ---
CORPUS_URL = "https://raw.githubusercontent.com/minhaz-engg/scrape-scheduler/main/out/combined_corpus.md" # PASTE YOUR RAW URL HERE
# --- 👆 ---
INDEX_DIR = "index" # Local cache for BM25 index
os.makedirs(INDEX_DIR, exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOPK = 8
DEFAULT_LANG = "en"

# --- Data Structures (No changes needed) ---
@dataclass
class ProductDoc:
    doc_id: str; title: str; url: Optional[str]; category: Optional[str]
    brand: Optional[str]; price_value: Optional[float]; rating_avg: Optional[float]
    raw_md: str; source: Optional[str] = None

@dataclass
class ChunkRec:
    doc_id: str; title: str; text: str; url: Optional[str] = None
    category: Optional[str] = None; brand: Optional[str] = None
    price_value: Optional[float] = None; rating_avg: Optional[float] = None
    source: Optional[str] = None

# --- Regex & Parsing Logic (No changes needed) ---
# ... (Keep all your parsing functions: DOC_BLOCK_RE, _meta_from_header, _parse_price_value, parse_products_from_md, etc.) ...
DOC_BLOCK_RE = re.compile(r"(?P<body>.*?)", re.DOTALL|re.IGNORECASE)
TITLE_RE = re.compile(r"^##\s+(.+?)\s*(?:\s{2,}\n|\n|$)", re.MULTILINE)
URL_LINE_RE = re.compile(r"\*\*URL:\*\*\s*(\S+)", re.IGNORECASE)
BRAND_RE = re.compile(r"\*\*Brand:\*\*\s*(.+)", re.IGNORECASE)
PRICE_RE = re.compile(r"\*\*Price:\*\*\s*([^\s\n]+)", re.IGNORECASE)
RATING_RE = re.compile(r"\*\*Rating:\*\*\s*([0-9.]+)", re.IGNORECASE)

def _meta_from_header(attrs: str) -> Dict[str,str]:
    out = {};
    for kv in attrs.strip().split():
        if "=" in kv: k, v = kv.split("=", 1); out[k.strip()] = v.strip().replace("'", "").replace('"', '')
    return out

def _parse_price_value(s: str) -> Optional[float]:
    if not s: return None
    s = s.replace(",", "").replace("৳", "").strip()
    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums: return None
    try: return float(nums[0])
    except (ValueError, IndexError): return None

def parse_products_from_md(md_text: str) -> List[ProductDoc]:
    products: List[ProductDoc] = []
    for m in DOC_BLOCK_RE.finditer(md_text):
        attrs = m.group("attrs") or ""; body = (m.group("body") or "").strip()
        meta = _meta_from_header(attrs)
        doc_id = meta.get("id") or f"doc_{len(products)+1}"
        source = meta.get("source")
        title_m = TITLE_RE.search(body); title = (title_m.group(1).strip() if title_m else f"Product {doc_id}")
        url_m = URL_LINE_RE.search(body); url = url_m.group(1).strip() if url_m else None
        brand_m = BRAND_RE.search(body); brand = brand_m.group(1).strip() if brand_m else None
        price_m = PRICE_RE.search(body); price_str = price_m.group(1).strip() if price_m else None
        price_value = _parse_price_value(price_str)
        rating_m = RATING_RE.search(body); rating_avg = float(rating_m.group(1)) if rating_m else None
        products.append(ProductDoc(
            doc_id=doc_id, title=title, url=url, category=meta.get("category"), brand=brand,
            price_value=price_value, rating_avg=rating_avg, source=source, raw_md=body
        ))
    return products


# --- Data Loading with Auto-Refresh from URL ---
# --- 👇 MODIFIED TO FETCH FROM URL ---
@st.cache_data(ttl=3600, show_spinner="Fetching latest product data from GitHub...") # 1-hour cache
def load_data_from_url(url: str) -> Tuple[List[ProductDoc], datetime.datetime]:
    """
    Downloads the corpus file from the specified URL and parses it.
    Caches the result for 1 hour.
    """
    try:
        headers = {'Cache-Control': 'no-cache'} # Try to bypass intermediate caches
        response = requests.get(url, headers=headers, timeout=30) # 30 second timeout
        response.raise_for_status() # Raise an error for bad status codes (404, 500, etc.)
        
        md_text = response.text
        products = parse_products_from_md(md_text)
        
        # We use the current time as "last updated" since we just fetched it
        last_updated = datetime.datetime.now()
        
        print(f"--- Successfully fetched and parsed corpus from {url} ---")
        return products, last_updated
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch corpus from GitHub: {e}")
        return [], datetime.datetime.now() # Return empty on error
    except Exception as e:
        st.error(f"Error processing corpus data: {e}")
        return [], datetime.datetime.now() # Return empty on error
# --- 👆 ---


# --- Chunking & BM25 Indexing (No changes needed) ---
# ... (Keep build_bm25_index, _tokenize, _sha1, product_to_chunks, etc. exactly as before) ...
def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower()); stopwords = set(["the","a","an","is","in","of","for"]); return [t for t in toks if t not in stopwords]
def _sha1(s: str) -> str: return hashlib.sha1(s.encode("utf-8")).hexdigest()
def build_chunker(lang: str) -> RecursiveChunker: return RecursiveChunker.from_recipe("markdown", lang=lang)
def _clean_for_bm25(text: str) -> str:
    lines = [];
    for line in text.splitlines():
        ll = line.strip();
        if not ll or ll.lower().startswith("**images"): continue;
        if "http" in ll: ll = " ".join([p for p in re.split(r"\s+https?://\S+", ll) if p.strip()]);
        if not ll: continue;
        lines.append(ll);
    return "\n".join(lines)
def product_to_chunks(product: ProductDoc, chunker: RecursiveChunker) -> List[ChunkRec]:
    chunks = [];
    try: chonks = chunker(product.raw_md)
    except Exception: chonks = [{"text": s} for s in re.split(r"\n{2,}", product.raw_md) if s.strip()]
    for c in chonks:
        text = getattr(c, 'text', '').strip();
        if text:
            cleaned_text = _clean_for_bm25(text) # Clean before adding
            if cleaned_text:
                 chunks.append(ChunkRec(doc_id=product.doc_id, title=product.title, text=cleaned_text, url=product.url, category=product.category, brand=product.brand, price_value=product.price_value, rating_avg=product.rating_avg, source=product.source))
    return chunks

@st.cache_data(show_spinner="Building search index...")
def build_bm25_index(products: List[ProductDoc]) -> Tuple[BM25Okapi, List[ChunkRec]]:
    chunker = build_chunker(DEFAULT_LANG); all_chunks: List[ChunkRec] = [];
    for p in products: all_chunks.extend(product_to_chunks(p, chunker));
    # Note: Using a simpler hash for caching BM25 based on product IDs only
    content_sig = _sha1(json.dumps(sorted([p.doc_id for p in products])));
    # Ensure cache directory exists
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    index_file = Path(INDEX_DIR) / f"bm25_index_{content_sig}.pkl"
    chunks_file = Path(INDEX_DIR) / f"chunks_{content_sig}.pkl"

    if index_file.exists() and chunks_file.exists():
        print(f"--- Loading cached BM25 index ({content_sig}) ---")
        with open(index_file, "rb") as f_idx, open(chunks_file, "rb") as f_chnk:
            bm25 = pickle.load(f_idx); cached_chunks = pickle.load(f_chnk);
            return bm25, cached_chunks

    print(f"--- Building new BM25 index ({content_sig}) ---")
    tokenized_corpus = [_tokenize(c.text) for c in all_chunks];
    bm25 = BM25Okapi(tokenized_corpus);
    with open(index_file, "wb") as f_idx, open(chunks_file, "wb") as f_chnk:
        pickle.dump(bm25, f_idx); pickle.dump(all_chunks, f_chnk);
    return bm25, all_chunks

# --- Search Logic (No changes needed) ---
# ... (Keep _passes_filters, bm25_search exactly as before) ...
def _passes_filters(chunk: ChunkRec, **filters) -> bool:
    if filters.get("allowed_sources") and (chunk.source not in filters["allowed_sources"]): return False
    if filters.get("allowed_categories") and (chunk.category not in filters["allowed_categories"]): return False
    if filters.get("brand_filter") and (filters["brand_filter"].lower() not in (chunk.brand or "").lower()): return False
    price_max = filters.get("price_max")
    if price_max is not None and chunk.price_value is not None and chunk.price_value > price_max: return False
    rating_min = filters.get("rating_min")
    if rating_min is not None and chunk.rating_avg is not None and chunk.rating_avg < rating_min: return False
    return True

def bm25_search(bm25: BM25Okapi, chunks: List[ChunkRec], query: str, top_k: int, **filters) -> List[Tuple[ChunkRec, float]]:
    q_tokens = _tokenize(query); scores = bm25.get_scores(q_tokens); pairs: List[Tuple[int, float]] = []
    for i, score in enumerate(scores):
        if _passes_filters(chunks[i], **filters): pairs.append((i, float(score)));
    pairs.sort(key=lambda x: x[1], reverse=True); diversified_results = []; seen_doc_ids = set()
    for i, score in pairs:
        if len(diversified_results) >= top_k: break
        # Prioritize unique products
        if chunks[i].doc_id not in seen_doc_ids:
            diversified_results.append((chunks[i], score)); seen_doc_ids.add(chunks[i].doc_id);
    # If still not enough, add duplicates (less diverse)
    if len(diversified_results) < top_k:
         for i, score in pairs:
             if len(diversified_results) >= top_k: break
             # Add chunk if it's not already the *exact* same chunk object
             if not any(dr[0] is chunks[i] for dr in diversified_results):
                  diversified_results.append((chunks[i], score))
    return diversified_results[:top_k] # Ensure only top_k are returned


# --- OpenAI Integration (No changes needed) ---
# ... (Keep _build_messages, stream_answer exactly as before) ...
def _build_messages(query: str, results: List[Tuple[ChunkRec, float]]) -> List[Dict[str, str]]:
    ctx_blocks = [];
    for i, (chunk, score) in enumerate(results, 1):
        header = f"[{i}] ({chunk.source}) {chunk.title} — DocID: {chunk.doc_id}";
        url_line = f"URL: {chunk.url}" if chunk.url else "";
        ctx_blocks.append(f"{header}\n{url_line}\n---\n{chunk.text}\n");
    system_prompt = ("You are a helpful e-commerce assistant for users in Bangladesh. Answer the user's question based *only* on the provided context blocks. If the information is not in the context, say 'I do not have that information in the provided data.' Present your answer clearly, using bullet points for lists of products. Cite your sources at the end of relevant sentences using the format `[#]`.");
    user_prompt = f"Question:\n{query}\n\nContext:\n" + "\n\n".join(ctx_blocks);
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def stream_answer(model: str, messages: List[Dict[str, str]], temperature: float):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"));
        if not client.api_key: raise ValueError("OpenAI API key not found.");
        response = client.chat.completions.create(model=model, temperature=temperature, messages=messages, stream=True);
        for chunk in response:
            delta = chunk.choices[0].delta.content or "";
            if delta: yield delta
    except Exception as e: yield f"\n\n**Error communicating with OpenAI:** {e}"


# --- Main Streamlit UI ---
st.set_page_config(page_title="Daraz & StarTech RAG", layout="wide")
st.title("Daraz & StarTech Product Search 🛒")

# --- 👇 CALL THE MODIFIED DATA LOADER ---
products, last_updated = load_data_from_url(CORPUS_URL)
# --- 👆 ---

if not products:
    st.warning("Could not load product data from the source. Please check the GitHub repository or wait for the next pipeline run.")
    st.stop()

# Build index (will be cached locally based on fetched data)
bm25, chunk_table = build_bm25_index(products)

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
    st.info(f"Source Updated: {last_updated.strftime('%b %d, %Y %I:%M %p')}") # Display fetch time

st.success(f"**Ready!** Using data from {len(products):,} products ({len(all_sources)} sources).")

# --- Filters ---
st.markdown("#### Filters")
filter_cols = st.columns(5)
with filter_cols[0]: sel_sources = st.multiselect("Source", options=all_sources, default=all_sources)
with filter_cols[1]: sel_categories = st.multiselect("Category", options=all_categories, default=[])
with filter_cols[2]: brand_filter = st.text_input("Brand contains", "")
with filter_cols[3]: price_max_ui = st.text_input("Max Price (৳)", "")
with filter_cols[4]: rating_min_ui = st.text_input("Min Rating (1-5)", "")

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
        st.warning("No products matched your search criteria."); st.stop()

    # --- Display Results ---
    colL, colR = st.columns([0.6, 0.4], gap="large")
    with colL:
        st.subheader(f"Top {len(results)} Matches")
        for i, (chunk, score) in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. ({chunk.source}) {chunk.title}**")
                meta_parts = [
                    f"Cat: {chunk.category}" if chunk.category else None,
                    f"Brand: {chunk.brand}" if chunk.brand else None,
                    f"Price: ৳{int(chunk.price_value)}" if chunk.price_value is not None else None,
                    f"Rating: {chunk.rating_avg}/5" if chunk.rating_avg is not None else None,
                ]
                st.caption(" | ".join(filter(None, meta_parts)) + f" (Score: {score:.2f})")
                if chunk.url: st.markdown(f"[View Product]({chunk.url})", unsafe_allow_html=True)
                with st.expander("Retrieved Text"): st.markdown(chunk.text)
    
    with colR:
        st.subheader("🤖 AI Generated Answer")
        with st.container(border=True):
            messages = _build_messages(query, results)
            st.write_stream(stream_answer(model, messages, temperature))