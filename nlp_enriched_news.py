"""NLP engine: enrich the scraped news articles.

Reads all articles stored by scraper_news.py from data/news_*.csv, then for
each article: detects ORG entities (spaCy NER), predicts the topic
(topic_classifier.pkl) and analyses sentiment (NLTK VADER). Results are
saved in results/enhanced_news.csv.

Run from the project root:  python nlp_enriched_news.py
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

DATA_GLOB = os.path.join("data", "news_*.csv")
MODEL_PATH = "topic_classifier.pkl"
OUTPUT_PATH = os.path.join("results", "enhanced_news.csv")

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

TOP_N = 10

# Environmental-disaster keywords. Multi-word, unambiguous phrases only, to
# avoid false positives from words like "spill" or "plant" used in other
# contexts (see subject warning).
SCANDAL_KEYWORDS = (
    "oil spill, toxic waste dumping, chemical leak, deforestation, "
    "water contamination, air pollution, radioactive leak, "
    "greenhouse gas emissions, environmental disaster, ecological damage, "
    "illegal dumping, groundwater pollution"
)

def load_articles():
    """Load every scraped daily CSV into a single DataFrame."""
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError("No scraped data found. Run scraper_news.py first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def detect_entities(doc):
    """Return the list of unique ORG entities in a spaCy doc, in order."""
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG" and ent.text not in orgs:
            orgs.append(ent.text)
    return orgs


def sentiment_label(compound):
    """Map a VADER compound score to a human-readable label."""
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"

def scandal_similarity(doc, orgs, keywords_doc):
    """Return the max cosine similarity between the disaster keywords and
    the sentences of the article that mention a detected ORG entity.

    Returns 0.0 if no sentence mentions an entity or vectors are empty.
    """
    best = 0.0
    for sentence in doc.sents:
        if not any(org in sentence.text for org in orgs):
            continue
        if sentence.vector_norm == 0:
            continue
        similarity = keywords_doc.similarity(sentence)
        best = max(best, float(similarity))
    return best

def main():
    articles = load_articles()
    nlp = spacy.load("en_core_web_md")
    analyzer = SentimentIntensityAnalyzer()
    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)
    keywords_doc = nlp(SCANDAL_KEYWORDS)

    rows = []
    for _, article in articles.iterrows():
        print(f"Enriching {article['url']}:\n")
        doc = nlp(article["body"])

        print("---------- Detect entities ----------\n")
        orgs = detect_entities(doc)
        if orgs:
            print(f"Detected {len(orgs)} companies which are {', '.join(orgs)}\n")
        else:
            print("Detected 0 companies\n")

        print("---------- Topic detection ----------\n")
        print("Text preprocessing ...\n")
        topic = classifier.predict([article["body"]])[0]
        print(f"The topic of the article is: {topic}\n")

        print("---------- Sentiment analysis ----------\n")
        compound = analyzer.polarity_scores(article["body"])["compound"]
        label = sentiment_label(compound)
        print(f"The article {article['headline']} has a {label} sentiment\n")
        print("---------- Scandal detection ----------\n")
        print("Computing embeddings and distance ...\n")
        distance = scandal_similarity(doc, orgs, keywords_doc)

        rows.append({
            "Unique ID": article["unique_id"],
            "URL": article["url"],
            "Date scraped": article["date"],
            "Headline": article["headline"],
            "Body": article["body"],
            "Org": orgs,
            "Topics": [topic],
            "Sentiment": compound,
            "Scandal_distance": distance,
        })

    result = pd.DataFrame(rows)
    threshold = result["Scandal_distance"].nlargest(TOP_N).min()
    result["Top_10"] = result["Scandal_distance"] >= threshold

    for _, row in result[result["Top_10"]].iterrows():
        entities = ", ".join(row["Org"]) if row["Org"] else "unknown entity"
        print(f"Environmental scandal detected for {entities}")

    os.makedirs("results", exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result)} enriched articles in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()