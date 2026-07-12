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