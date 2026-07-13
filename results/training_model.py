"""Train the BBC topic classifier and save it as topic_classifier.pkl.

Trains a TF-IDF + LinearSVC pipeline on data/bbc_news_train.csv, evaluates
accuracy on data/bbc_news_tests.csv (required: > 95%), and saves learning
curves to results/learning_curves.png to show the model does not overfit.

Run from the project root:  python results/training_model.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # save plots to file; no display window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "bbc_news_train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "bbc_news_tests.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "topic_classifier.pkl")
CURVES_PATH = os.path.join(PROJECT_ROOT, "results", "learning_curves.png")

RANDOM_STATE = 42


def build_pipeline():
    """Return the text-classification pipeline: TF-IDF -> Linear SVM."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", min_df=2,
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
    ])


def plot_learning_curves(pipeline, texts, labels):
    """Compute and save learning curves (training vs cross-validation score)."""
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, texts, labels,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Training accuracy")
    plt.plot(train_sizes, val_mean, "o-", label="Cross-validation accuracy")
    plt.xlabel("Number of training articles")
    plt.ylabel("Accuracy")
    plt.title("Learning curves - BBC topic classifier")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(CURVES_PATH, dpi=150, bbox_inches="tight")
    print(f"Learning curves saved in {CURVES_PATH}")


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH, index_col=0)

    pipeline = build_pipeline()

    print("Computing learning curves ...")
    plot_learning_curves(build_pipeline(), train_df["Text"], train_df["Category"])

    print("Training on the full training set ...")
    pipeline.fit(train_df["Text"], train_df["Category"])

    predictions = pipeline.predict(test_df["Text"])
    accuracy = accuracy_score(test_df["Category"], predictions)
    print(f"Accuracy on test set: {accuracy:.4f}")
    print(classification_report(test_df["Category"], predictions))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved in {MODEL_PATH}")


if __name__ == "__main__":
    main()