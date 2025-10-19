#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- Config ----------
INPUT_ROOT = Path("./result") # ⭐️ Reads from scrape.py's output
OUT_DIR = Path("out")
MAX_IMAGES = 5
MAX_DESC_CHARS = 1500
PRINT_EVERY = 500
# ----------------------------

# Output file names
MD_PATH = OUT_DIR / "daraz_products_corpus.md"
TXT_PATH = OUT_DIR / "daraz_products_corpus.txt"
JSONL_PATH = OUT_DIR / "daraz_products_corpus.jsonl"

# --- All your helper functions (normalize_url, safe_get, clean_text, etc.) ---
# (No changes needed, they are perfect)

def normalize_url(u: Optional[str]) -> Optional[str]:
    if not u: return None; u = u.strip()
    if u.startswith("//"): return "https:" + u
    if u.startswith("http://") or u.startswith("https://"): return u
    return u

def category_readable_name(folder: str) -> str:
    prefix = "www_daraz_com_bd_"
    if folder.startswith(prefix): folder = folder[len(prefix):]
    return folder.replace("_", " ").strip()

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def clean_text(t: Optional[str]) -> Optional[str]:
    if not t: return t
    t = re.sub(r"\s+\n", "\n", t); t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t); t = t.strip()
    if len(t) > MAX_DESC_CHARS: t = t[:MAX_DESC_CHARS].rstrip() + " …"
    return t

def list_preview(items: List[str], max_n: int) -> List[str]:
    out = []
    for s in items[:max_n]:
        if isinstance(s, str):
            s = s.strip()
            if s: out.append(s)
    return out

def unique_stable_key(prod: Dict[str, Any]) -> str:
    candidates = [
        prod.get("data_item_id"), safe_get(prod, "detail", "url"),
        prod.get("detail_url"), prod.get("product_detail_url"),
        prod.get("data_sku_simple"),
    ]
    for c in candidates:
        if c: return str(c)
    return (prod.get("product_title") or "unknown") + "||" + (prod.get("image_url") or "unknown")

def unify_product(prod: Dict[str, Any], cat_folder: str, pfile: Path) -> Dict[str, Any]:
    detail = prod.get("detail") or {}
    title = detail.get("name") or prod.get("product_title") or "Unknown"
    brand = detail.get("brand") or None
    price_disp = safe_get(detail, "price", "display") or prod.get("product_price")
    rating_avg = safe_get(detail, "rating", "average")
    rating_count = safe_get(detail, "rating", "count")
    url = normalize_url(detail.get("url") or prod.get("detail_url") or prod.get("product_detail_url"))
    image_url = prod.get("image_url")
    images = detail.get("images") or ([image_url] if image_url else [])
    images = [normalize_url(i) for i in images if isinstance(i, str) and i.startswith(("http", "//"))]
    images = list_preview([i for i in images if i], max_n=MAX_IMAGES)
    
    return {
        "id": str(prod.get("data_item_id") or unique_stable_key(prod)),
        "title": str(title),
        "brand": brand,
        "category": category_readable_name(Path(cat_folder).name),
        "url": url,
        "sku": prod.get("data_sku_simple"),
        "images": images,
        "price_display": price_disp,
        "original_price_display": safe_get(detail, "price", "original_display"),
        "discount_display": safe_get(detail, "price", "discount_display"),
        "rating_average": rating_avg,
        "rating_count": rating_count,
        "sold_text": prod.get("location"),
        "seller_name": safe_get(detail, "seller", "name"),
        "description": clean_text(safe_get(detail, "details", "description_text")),
        # Metadata for verification
        "_source_category_dir": str(Path(cat_folder).name),
        "_source_file": str(pfile.relative_to(INPUT_ROOT)),
    }

# --- Writers (streaming) ---
def write_markdown_header(fmd):
    fmd.write("# Daraz Product Corpus\n\n")

def product_to_markdown_block(p: Dict[str, Any]) -> str:
    lines = [f""]
    lines.append(f"## {p['title']}  \n**DocID:** {p['id']}")
    meta = [f"**Category:** {p['category']}" if p.get("category") else None,
            f"**Brand:** {p['brand']}" if p.get("brand") else None,
            f"**SKU:** {p['sku']}" if p.get("sku") else None,
            f"**URL:** {p['url']}" if p.get("url") else None]
    lines.append("  \n".join(filter(None, meta)))
    price_bits = [f"**Price:** {p['price_display']}" if p.get('price_display') else None,
                  f"**Original:** {p['original_price_display']}" if p.get('original_price_display') else None]
    rating_bits = []
    if p.get("rating_average") is not None:
        rc = p.get('rating_count')
        rating_bits.append(f"**Rating:** {p['rating_average']}/5" + (f" ({rc} ratings)" if rc else ""))
    lines.append("  \n".join(filter(None, price_bits + rating_bits)))
    if p.get("description"):
        lines.append(f"\n**Description:**\n{p['description']}")
    lines.append("\n---")
    lines.append("\n")
    return "\n".join(lines)

def product_to_jsonl_record(p: Dict[str, Any]) -> Dict[str, Any]:
    text_lines = [f"{p['title']} (ID: {p['id']})"]
    if p.get("brand"): text_lines.append(f"Brand: {p['brand']}")
    if p.get("category"): text_lines.append(f"Category: {p['category']}")
    if p.get("price_display"): text_lines.append(f"Price: {p['price_display']}")
    if p.get("rating_average"): text_lines.append(f"Rating: {p['rating_average']}/5")
    if p.get("description"): text_lines.append(f"Description: {p['description']}")
    
    metadata = p.copy() # Keep all unified data as metadata
    return {"id": p["id"], "text": "\n".join(text_lines), "metadata": metadata}

# --- Main execution ---
def main():
    if not INPUT_ROOT.exists():
        print(f"Input folder not found: {INPUT_ROOT}. Run scrape.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with MD_PATH.open("w", encoding="utf-8") as fmd, \
         TXT_PATH.open("w", encoding="utf-8") as ftxt, \
         JSONL_PATH.open("w", encoding="utf-8") as fjsonl:

        write_markdown_header(fmd)
        ftxt.write("DARAZ PRODUCT CORPUS (TEXT)\n\n")

        products_files = sorted(INPUT_ROOT.glob("*/products.json"))
        if not products_files:
            print(f"No products.json files found under: {INPUT_ROOT}")
            return

        seen_ids = set()
        total_written = 0
        categories = set()

        for pfile in products_files:
            cat_dir = pfile.parent
            categories.add(cat_dir.name)
            
            try: data = json.loads(pfile.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] Failed to parse {pfile}: {e}"); continue
            
            if not isinstance(data, list): continue
            
            for raw in data:
                uid = unique_stable_key(raw)
                if uid in seen_ids: continue
                seen_ids.add(uid)

                unified = unify_product(raw, str(cat_dir), pfile)
                
                fmd.write(product_to_markdown_block(unified))
                ftxt.write(product_to_markdown_block(unified)) # Using same for text
                
                rec = product_to_jsonl_record(unified)
                fjsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

                total_written += 1
                if total_written % PRINT_EVERY == 0:
                    print(f"[INFO] Processed {total_written:,} products...")

    print(f"[OK] Wrote {total_written} unique products to {OUT_DIR}/")

if __name__ == "__main__":
    main()