"""Scrape BBC News articles and store them in one CSV file per day.

Each stored article has: unique ID, URL, date scraped, headline, body.
Run:  python scraper_news.py   (stops automatically at TARGET_ARTICLES)
"""

import csv
import os
import time
import uuid
from datetime import date

import requests
from bs4 import BeautifulSoup

SITEMAP_INDEX = "https://www.bbc.com/sitemaps/https-index-com-news.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nlp-scraper student project)"}
DATA_DIR = "data"
TARGET_ARTICLES = 300
FIELDNAMES = ["unique_id", "url", "date", "headline", "body"]


def get_sitemap_urls():
    """Return the list of news sub-sitemap URLs from BBC's sitemap index."""
    response = requests.get(SITEMAP_INDEX, headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml-xml")
    return [loc.text for loc in soup.find_all("loc")]


def get_article_urls():
    """Collect candidate article URLs from all news sub-sitemaps."""
    urls = []
    for sitemap_url in get_sitemap_urls():
        response = requests.get(sitemap_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml-xml")
        for loc in soup.find_all("loc"):
            url = loc.text
            # Keep standard English news articles; skip live blogs and videos,
            # which have no normal article body to extract.
            if "/news/articles/" in url and "/live/" not in url:
                urls.append(url)
    return urls


def parse_article(url):
    """Download one article page and extract headline and body.

    Returns a dict with the five required fields, or None if the page
    does not contain a usable article.
    """
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    headline_tag = soup.find("h1")
    paragraphs = soup.select('article div[data-component="text-block"] p')
    if headline_tag is None or not paragraphs:
        return None

    body = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    if not body.strip():
        return None

    return {
        "unique_id": str(uuid.uuid4()),
        "url": url,
        "date": date.today().isoformat(),
        "headline": headline_tag.get_text(strip=True),
        "body": body,
    }


def load_seen_urls():
    """Return URLs already stored, so re-running the scraper never duplicates."""
    seen = set()
    if not os.path.isdir(DATA_DIR):
        return seen
    for filename in os.listdir(DATA_DIR):
        if filename.startswith("news_") and filename.endswith(".csv"):
            with open(os.path.join(DATA_DIR, filename), newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seen.add(row["url"])
    return seen


def count_stored_articles():
    """Return how many articles are already stored across all daily files."""
    return len(load_seen_urls())


def save_article(article):
    """Append one article to today's CSV file and return the file path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"news_{date.today().isoformat()}.csv")
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(article)
    return path


def main():
    seen_urls = load_seen_urls()
    stored = len(seen_urls)
    print(f"Already stored: {stored} articles. Target: {TARGET_ARTICLES}.")

    counter = 0
    for url in get_article_urls():
        if stored >= TARGET_ARTICLES:
            break
        if url in seen_urls:
            continue

        counter += 1
        print(f"{counter}. scraping {url}")
        try:
            print("\trequesting ...")
            article = parse_article(url)
            print("\tparsing ...")
            if article is None:
                print("\tskipped (no article body found)\n")
                continue
            path = save_article(article)
            print(f"\tsaved in {path}\n")
            seen_urls.add(url)
            stored += 1
        except requests.RequestException as error:
            print(f"\tskipped (request failed: {error})\n")
        time.sleep(0.5)

    print(f"Done. {stored} articles stored in '{DATA_DIR}/'.")


if __name__ == "__main__":
    main()
    