# NLP Scraper

NLP-enriched news intelligence platform: scrapes BBC News articles, then
detects organizations (spaCy NER), classifies topics (TF-IDF + scikit-learn),
analyses sentiment (NLTK VADER), and flags potential environmental scandals
using word embeddings.

## Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

## Usage

python scraper_news.py        # fetch >= 300 articles into data/
python results/training_model.py   # train and save topic_classifier.pkl
python nlp_enriched_news.py   # produce results/enhanced_news.csv

## Scandal detection: embeddings and distance choice

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