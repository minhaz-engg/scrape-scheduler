# import json

# from orchestra import scrape_product



# def scrape_product_detail(url: str) -> str:
#     # Put your product page URL here
#     # url = "https://www.daraz.com.bd/products/high-quality-mens-shoes-imported-i378023244-s1895600858.html"

#     result = scrape_product(url, render_if_missing=True, debug=True)

#     js = json.dumps(result, ensure_ascii=False, indent=2)
#     return js
#     # print(js)
#     # with open("product.json", "w", encoding="utf-8") as f:
#     #     f.write(js)
#     # print("Saved to product.json")

# # url = "https://www.daraz.com.bd/products/understated-craftsmanship-and-trendy-new-indian-vichitra-silk-saree-high-quality-embroidery-work-on-blouse-progressively-better-i305660146-s1365577801.html"
# # scrape_product_detail(url=url)


# detail_main.py
import json
import time
import random
from typing import Optional, Dict, Any
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Add any other imports detail_main needs, e.g., helpers

# --- Tunables for Detail Scraping ---
DETAIL_PAGE_TIMEOUT = 25000  # Increased timeout for detail pages (25 seconds)
DETAIL_WAIT_UNTIL = 'domcontentloaded' # Faster than 'load' or 'networkidle' for simple data
DETAIL_SLEEP_AFTER_LOAD = 1.5 # Extra sleep after page claims to load
RETRY_DELAY_RANGE = (2.0, 5.0)
MAX_DETAIL_RETRIES = 2 # Limit retries to avoid long runs

def extract_product_details_from_page(page, url: str) -> Optional[Dict[str, Any]]:
    """
    Your actual logic to extract details from the Playwright page object.
    Focus on selecting elements and getting text/attributes.
    Add print statements here too if needed.
    Return a dictionary with the data, or None if extraction fails.
    """
    print(f"   [Detail] Attempting to extract data from: {url}")
    details = {"url": url} # Start with the URL

    try:
        # Example: Extract product name (Update selector for Daraz)
        name_element = page.query_selector('h1.pdp-mod-product-badge-title') # Adjust selector
        if name_element:
            details["name"] = name_element.text_content().strip()
            print(f"      [Detail] Found name: {details['name'][:50]}...")
        else:
            print(f"      [Detail] ⚠️ Product name element not found.")

        # Example: Extract price (Update selector for Daraz)
        price_element = page.query_selector('.pdp-price .price') # Adjust selector
        if price_element:
            details["price"] = {"display": price_element.text_content().strip()}
            print(f"      [Detail] Found price: {details['price']['display']}")
        else:
            print(f"      [Detail] ⚠️ Price element not found.")

        # Example: Extract description (Update selector for Daraz)
        desc_element = page.query_selector('.pdp-product-desc') # Adjust selector
        if desc_element:
            details["description_text"] = desc_element.text_content().strip()[:1000] # Limit length
            print(f"      [Detail] Found description: {details['description_text'][:50]}...")
        else:
             print(f"      [Detail] ⚠️ Description element not found.")

        # Add extraction logic for other fields (brand, ratings, images, specs etc.)
        # ... your selectors and extraction code ...

        print(f"   [Detail] Extraction successful for: {url}")
        return details

    except Exception as e:
        print(f"   [Detail] ❌ Error during data extraction on page {url}: {e}")
        return None


def scrape_product_detail(url: str) -> Optional[str]:
    """
    Fetches a single product detail page using Playwright (sync) and extracts data.
    Includes retries and enhanced logging.
    """
    print(f"-> [Detail] Starting scrape for URL: {url}")
    if not url or not url.startswith("http"):
        print(f"   [Detail] ⚠️ Invalid URL: {url}")
        return json.dumps({"_error": "Invalid URL"})

    for attempt in range(1, MAX_DETAIL_RETRIES + 1):
        print(f"   [Detail] Attempt {attempt}/{MAX_DETAIL_RETRIES}...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # Use a specific user agent if needed
                context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
                page = context.new_page()

                # print(f"      [Detail] Navigating to page...")
                # page.goto(url, timeout=DETAIL_PAGE_TIMEOUT, wait_until=DETAIL_WAIT_UNTIL)
                # print(f"      [Detail] Page navigation triggered for {url}. Waiting {DETAIL_SLEEP_AFTER_LOAD}s...")
                # time.sleep(DETAIL_SLEEP_AFTER_LOAD + random.uniform(0, 0.5)) # Add random jitter
                
                # ---------------------------------------------------
                
                print(f"      [Detail] Navigating to page...")
                # 1. Use 'load' to wait for more resources (images, etc.)
                page.goto(url, timeout=DETAIL_PAGE_TIMEOUT, wait_until='load')
                print(f"      [Detail] Page loaded for {url}. Waiting for dynamic content...")

                # 2. Wait for a *specific element* that is loaded by JavaScript
                # This is the most reliable way. We wait for the product title.
                try:
                    page.wait_for_selector('h1.pdp-mod-product-badge-title', timeout=10000) # Wait 10s
                    print(f"      [Detail] Dynamic content (title) appeared.")
                except Exception as e:
                    print(f"      [Detail] ⚠️ Waited 10s, but key element 'h1.pdp-mod-product-badge-title' not found. Page might be empty.")
                    # We can let it continue, extract_product_details_from_page will just find nothing
                
                # You can even remove the time.sleep() now, or keep a very short one.
                time.sleep(0.5 + random.uniform(0, 0.5))
                # ---------------------------------------------------

                # Optional: Add a specific wait after load if needed
                # page.wait_for_selector('YOUR_RELIABLE_SELECTOR', timeout=10000)

                product_data = extract_product_details_from_page(page, url)

                browser.close()

                if product_data:
                    print(f"<- [Detail] Successfully scraped: {url}")
                    return json.dumps(product_data, ensure_ascii=False)
                else:
                    # Extraction failed, maybe retry
                    print(f"   [Detail] ⚠️ Extraction logic failed on attempt {attempt} for {url}.")
                    # Optional: Add screenshot on failure
                    # page.screenshot(path=f'error_screenshot_{attempt}.png')

        except PlaywrightTimeoutError:
            print(f"   [Detail] ❌ TimeoutError on attempt {attempt} for {url} after {DETAIL_PAGE_TIMEOUT/1000}s.")
            if browser: browser.close() # Ensure browser is closed
        except Exception as e:
            print(f"   [Detail] ❌ Unexpected Error on attempt {attempt} for {url}: {e}")
            if 'browser' in locals() and browser.is_connected(): browser.close() # Ensure browser is closed

        # If loop continues, it means an error occurred or extraction failed
        if attempt < MAX_DETAIL_RETRIES:
            delay = random.uniform(*RETRY_DELAY_RANGE)
            print(f"   [Detail] Retrying after {delay:.1f} seconds...")
            time.sleep(delay)

    # If all retries failed
    print(f"<- [Detail] ❌ Failed to scrape detail after {MAX_DETAIL_RETRIES} attempts: {url}")
    return json.dumps({"_error": f"Failed after {MAX_DETAIL_RETRIES} attempts", "url": url})

# # --- Simple test block ---
# if __name__ == "__main__":
#     test_url = "https://www.daraz.com.bd/products/-i472283182-s2175210955.html" # Replace with a valid Daraz product URL
#     print(f"Testing detail scrape for: {test_url}")
#     result_json = scrape_product_detail(test_url)
#     print("\nResult:")
#     if result_json:
#         print(json.dumps(json.loads(result_json), indent=2))
#     else:
#         print("No result returned.")