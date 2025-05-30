# multi_scraper.py

import os
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def extract_content_and_title(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(2000)
        content = page.content()
        browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # Remove unnecessary tags
        for tag in soup(['nav', 'footer', 'script', 'style', 'aside']):
            tag.decompose()

        # Get the main heading
        title_tag = soup.find("h1")
        title = title_tag.get_text() if title_tag else "untitled"

        # Clean title to be filename-safe
        filename = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_') + ".txt"

        # Extract full text
        text = soup.get_text(separator="\n").strip()
        return text, filename

def save_to_file(text, filename):
    os.makedirs("lightcast_scraped_docs", exist_ok=True)
    path = os.path.join("lightcast_scraped_docs", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    urls = [
        "https://kb.lightcast.io/en/articles/7215917-lightcast-occupation-taxonomy-lot",
        "https://kb.lightcast.io/en/articles/10344528-lightcast-nations-and-dataruns"
        # Add more doc URLs here...
    ]

    for url in urls:
        print(f"🔍 Scraping: {url}")
        try:
            text, filename = extract_content_and_title(url)
            save_to_file(text, filename)
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")
