# NLP Scraper - News Intelligence Platform

An end-to-end NLP pipeline that scrapes live news articles and enriches them
with entity recognition, topic classification, sentiment analysis and
environmental-scandal detection.

Built as part of the 01Edu AI Specialization curriculum.

## What it does

The platform runs in two independent stages:

**Stage 1 - Scraper** (`scraper_news.py`)
Discovers fresh BBC News articles through the site's XML news sitemap,
downloads and parses each page, and stores 300+ articles (unique ID, URL,
date, headline, body) in one CSV file per day. Idempotent: re-running it
tops up the dataset without ever duplicating an article.

**Stage 2 - NLP engine** (`nlp_enriched_news.py`)
Processes every stored article through four analyses:

| Analysis | Method | Output |
|---|---|---|
| Entity detection | spaCy NER (`en_core_web_md`) | List of `ORG` entities |
| Topic detection | TF-IDF + LinearSVC (custom-trained) | tech / sport / business / entertainment / politics |
| Sentiment analysis | NLTK VADER (pre-trained) | Compound score in [-1, 1] |
| Scandal detection | Word embeddings + cosine similarity | Per-article score + Top-10 flag |

Results are consolidated into `results/enhanced_news.csv` (10 columns,
one row per article).

## Pipeline

```
BBC news sitemap ──> scraper_news.py ──> data/news_YYYY-MM-DD.csv
                                              │
bbc_news_train.csv ──> results/training_model.py ──> topic_classifier.pkl
                                              │              │
                                              ▼              ▼
                                        nlp_enriched_news.py
                                              │
                                              ▼
                                   results/enhanced_news.csv
```

## Results

- Topic classifier accuracy on the held-out test set: **98.5%**
  (required: > 95%), with per-class F1 between 0.97 and 0.99.
- Learning curves (`results/learning_curves.png`) show training and
  cross-validation accuracy converging as data grows - the model
  generalizes rather than memorizes.
- Scandal detection surfaces genuinely environmental stories (water
  contamination, oil terminals, reservoir incidents) from a general
  news feed with no labeled scandal data at all.

## Project structure

```
.
├── data/
│   ├── bbc_news_train.csv       # labeled BBC dataset (training)
│   ├── bbc_news_tests.csv       # labeled BBC dataset (evaluation)
│   └── news_YYYY-MM-DD.csv      # scraped articles, one file per day
├── results/
│   ├── training_model.py        # trains + evaluates the topic classifier
│   ├── learning_curves.png      # overfitting diagnostic
│   └── enhanced_news.csv        # final enriched output
├── scraper_news.py              # stage 1: article scraper
├── nlp_enriched_news.py         # stage 2: NLP engine
├── topic_classifier.pkl         # trained TF-IDF + LinearSVC pipeline
└── requirements.txt             # pinned dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

On Windows, use `python` instead of `python3`.

## Usage

```bash
python scraper_news.py             # 1. fetch >= 300 articles into data/
python results/training_model.py   # 2. train and save topic_classifier.pkl
python nlp_enriched_news.py        # 3. produce results/enhanced_news.csv
```

## Design decisions

### Scandal detection: embeddings and distance choice

**Embeddings:** spaCy `en_core_web_md` word vectors (300-dimensional,
GloVe-style, trained on large web corpora). Sentence vectors are the average
of word vectors. Chosen because: (1) the same library already performs NER,
so one model serves two tasks; (2) unlike the small model, `md` ships real
pre-trained vectors, so semantically related words (pollution/contamination)
are close in vector space; (3) averaging gives robust sentence-level
representations without training anything.

**Similarity measure:** cosine similarity between the embedded disaster
keywords and every article sentence containing a detected ORG entity.
Cosine compares the *direction* of vectors, not their magnitude, so short
and long sentences are compared fairly - meaning is captured by direction.
Values range from 0 (unrelated) to 1 (same meaning).

**Per-article metric:** the maximum sentence similarity. A scandal is
typically expressed in one strong sentence inside an otherwise neutral
article; the mean would dilute that signal, the max preserves it.
The 10 articles with the highest score are flagged `Top_10 = True`.

**Keywords:** multi-word, unambiguous phrases only (e.g. "oil spill",
"toxic waste dumping"), avoiding single ambiguous words such as "spill" or
"plant" that also occur in unrelated contexts and would produce false
positives.

### Other choices

- **BBC as the news source** - its XML news sitemaps make discovery clean
  and reliable, and the topic classifier is trained on BBC text, so scraped
  articles match the training distribution.
- **One CSV per day** instead of a SQL database - simpler, transparent,
  directly inspectable, and sufficient for batch processing.
- **Pre-trained sentiment model (VADER)** - labeled news sentiment data is
  expensive; reusing a validated lexicon-based model is the pragmatic
  industry approach.
- **Fixed random seeds** - training is fully reproducible run to run.

## Known limitations

- spaCy NER occasionally tags non-companies (e.g. sports teams, government
  bodies) as `ORG`; these are still organizations, but not all are
  companies.
- The scandal score is unsupervised - it ranks likelihood, it does not
  verify that a scandal occurred.
- BBC's news sitemap covers roughly the last 48 hours, so building a
  multi-day dataset requires running the scraper on several days.

## Tech stack

Python · requests · BeautifulSoup · pandas · scikit-learn · spaCy · NLTK · matplotlib