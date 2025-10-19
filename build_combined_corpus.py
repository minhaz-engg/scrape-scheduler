#!/usr/bin/env python3
import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- Config ----------
DARAZ_INPUT_ROOT = Path("./result")
STARTECH_CSV_PATH = Path("./startech_products.csv")
OUT_DIR = Path("out")
MAX_DESC_CHARS = 1500
PRINT_EVERY = 500
# ----------------------------

# Output file names
MD_PATH = OUT_DIR / "combined_corpus.md"
TXT_PATH = OUT_DIR / "combined_corpus.txt" # Optional
JSONL_PATH = OUT_DIR / "combined_corpus.jsonl" # Optional

# --- Helper Functions (mostly from your original build_corpus.py) ---
def normalize_url(u: Optional[str]) -> Optional[str]:
    # (Same as before)
    if not u: return None; u = u.strip()
    if u.startswith("//"): return "https:" + u
    if u.startswith("http://") or u.startswith("https://"): return u
    return u # Assume valid if not starting with http/https/ //

def category_readable_name(folder: str) -> str:
    # (Same as before)
    prefix = "www_daraz_com_bd_"
    if folder.startswith(prefix): folder = folder[len(prefix):]
    return folder.replace("_", " ").strip()

def safe_get(d: Dict[str, Any], *keys, default=None):
    # (Same as before)
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def clean_text(t: Optional[str]) -> Optional[str]:
    # (Same as before)
    if not t: return t
    t = re.sub(r"\s+\n", "\n", t); t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t); t = t.strip()
    if len(t) > MAX_DESC_CHARS: t = t[:MAX_DESC_CHARS].rstrip() + " …"
    return t

def unique_stable_key(prod: Dict[str, Any], source: str) -> str:
    # Use different keys based on source, prefix with source
    if source == 'Daraz':
        candidates = [
            prod.get("data_item_id"), safe_get(prod, "detail", "url"),
            prod.get("detail_url"), prod.get("product_detail_url"),
            prod.get("data_sku_simple"),
        ]
        for c in candidates:
            if c: return f"daraz_{str(c)}"
        # Fallback for Daraz
        key_part = (prod.get("product_title") or "unknown") + "||" + (prod.get("image_url") or "unknown")
        return f"daraz_{hash(key_part)}" # Use hash for fallback
    elif source == 'StarTech':
        # StarTech URL seems like a good unique key
        if prod.get("url"):
            return f"startech_{prod['url']}"
        # Fallback for StarTech
        key_part = (prod.get("name") or "unknown") + "||" + (prod.get("price") or "unknown")
        return f"startech_{hash(key_part)}" # Use hash for fallback
    else:
        return f"unknown_{hash(str(prod))}" # Generic fallback


def unify_daraz_product(prod: Dict[str, Any], cat_folder: str, pfile: Path) -> Dict[str, Any]:
    """Converts raw Daraz JSON object to a unified structure."""
    detail = prod.get("detail") or {}
    title = detail.get("name") or prod.get("product_title") or "Unknown Daraz Product"
    
    return {
        "source": "Daraz", # Add source field
        "id": unique_stable_key(prod, "Daraz"),
        "title": str(title),
        "brand": detail.get("brand") or prod.get("brand"), # Try getting brand from main scrape too
        "category": category_readable_name(Path(cat_folder).name),
        "url": normalize_url(detail.get("url") or prod.get("detail_url") or prod.get("product_detail_url")),
        "price_display": safe_get(detail, "price", "display") or prod.get("product_price"),
        "rating_average": safe_get(detail, "rating", "average"),
        "status": "N/A", # Daraz structure doesn't easily provide stock status here
        "description": clean_text(safe_get(detail, "details", "description_text")),
        "_raw_data": prod # Optional: keep original for debugging
    }

def unify_startech_product(row: Dict[str, str]) -> Dict[str, Any]:
    """Converts StarTech CSV row (as dict) to a unified structure."""
    return {
        "source": "StarTech", # Add source field
        "id": unique_stable_key(row, "StarTech"),
        "title": row.get("name", "Unknown StarTech Product"),
        "brand": None, # StarTech CSV doesn't seem to have brand easily available
        "category": row.get("category", "Unknown"),
        "url": normalize_url(row.get("url")),
        "price_display": row.get("price", "N/A"),
        "rating_average": None, # StarTech CSV doesn't have ratings
        "status": row.get("status", "N/A"),
        "description": None, # StarTech CSV doesn't have description
        "_raw_data": row # Optional: keep original
    }

# --- Markdown Writer ---
def product_to_markdown_block(p: Dict[str, Any]) -> str:
    """Creates a markdown block for a unified product."""
    lines = [f""]
    lines.append(f"## {p['title']}  \n**DocID:** `{p['id']}`")
    meta = [f"**Source:** {p['source']}" if p.get('source') else None,
            f"**Category:** {p['category']}" if p.get("category") else None,
            f"**Brand:** {p['brand']}" if p.get("brand") else None,
            f"**Status:** {p['status']}" if p.get('status') and p['status'] != 'N/A' else None,
            f"**URL:** {p['url']}" if p.get("url") else None]
    lines.append("  \n".join(filter(None, meta)))
    
    price_bits = [f"**Price:** {p['price_display']}" if p.get('price_display') and p['price_display'] != 'N/A' else None]
    rating_bits = []
    if p.get("rating_average") is not None:
        rating_bits.append(f"**Rating:** {p['rating_average']}/5") # Assuming 5 is max
        
    lines.append("  \n".join(filter(None, price_bits + rating_bits)))
    
    if p.get("description"):
        lines.append(f"\n**Description:**\n{p['description']}")
        
    lines.append("\n---")
    lines.append("\n")
    return "\n".join(lines)


# --- Main execution ---
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_unified_products = []
    seen_ids = set()
    
    # --- 1. Process Daraz Data ---
    print("--- Processing Daraz JSON files ---")
    if DARAZ_INPUT_ROOT.exists():
        daraz_files = sorted(DARAZ_INPUT_ROOT.glob("*/products.json"))
        print(f"Found {len(daraz_files)} Daraz product files.")
        for pfile in daraz_files:
            cat_dir = pfile.parent
            try: 
                data = json.loads(pfile.read_text(encoding="utf-8"))
                if not isinstance(data, list): continue
                
                for raw in data:
                    unified = unify_daraz_product(raw, str(cat_dir), pfile)
                    if unified['id'] in seen_ids: continue
                    seen_ids.add(unified['id'])
                    all_unified_products.append(unified)
            except Exception as e:
                print(f"[WARN] Failed to process {pfile}: {e}")
    else:
        print(f"[WARN] Daraz input directory not found: {DARAZ_INPUT_ROOT}")

    # --- 2. Process StarTech Data ---
    print("\n--- Processing StarTech CSV file ---")
    if STARTECH_CSV_PATH.exists():
        try:
            with open(STARTECH_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                startech_rows = list(reader)
                print(f"Found {len(startech_rows)} rows in StarTech CSV.")
                for row in startech_rows:
                    unified = unify_startech_product(row)
                    if unified['id'] in seen_ids: continue
                    seen_ids.add(unified['id'])
                    all_unified_products.append(unified)
        except Exception as e:
            print(f"[WARN] Failed to process {STARTECH_CSV_PATH}: {e}")
    else:
        print(f"[WARN] StarTech CSV file not found: {STARTECH_CSV_PATH}")

    # --- 3. Write Combined Output ---
    print(f"\n--- Writing combined corpus ({len(all_unified_products)} unique products) ---")
    if not all_unified_products:
        print("[ERROR] No products found from either source. Cannot write corpus.")
        return
        
    with MD_PATH.open("w", encoding="utf-8") as fmd: # Add TXT/JSONL writers if needed
        fmd.write("# Combined Daraz & StarTech Product Corpus\n\n")
        
        count = 0
        for product in all_unified_products:
            fmd.write(product_to_markdown_block(product))
            count += 1
            if count % PRINT_EVERY == 0:
                print(f"[INFO] Wrote {count:,} products to corpus...")
                
    print(f"[OK] Wrote {len(all_unified_products)} products to {MD_PATH}")

if __name__ == "__main__":
    main()