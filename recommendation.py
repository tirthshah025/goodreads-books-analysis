from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).resolve().parent / "books_cleaned.csv"


def load_books() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["title"]).copy()
    return df.reset_index(drop=True)


def _build_similarity_model(df: pd.DataFrame):
    df = df.copy()
    df["content_text"] = (
        df["title"].fillna("")
        + " "
        + df["authors"].fillna("")
        + " "
        + df["publisher"].fillna("")
        + " "
        + df["language_code"].fillna("")
    ).str.lower()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["content_text"])
    similarity = cosine_similarity(tfidf_matrix)
    return df, similarity


def get_recommendations(book_title: str, top_n: int = 10) -> pd.DataFrame:
    """Return up to top_n similar books based on TF-IDF + cosine similarity."""
    df = load_books()
    normalized = df["title"].str.lower().str.strip()
    match_hits = df.loc[normalized == book_title.lower().strip()]

    if match_hits.empty:
        match_hits = df.loc[df["title"].str.lower().str.contains(book_title.lower().strip(), na=False)].head(1)

    if match_hits.empty:
        raise ValueError(f"Book '{book_title}' not found in the catalog.")

    selected = match_hits.iloc[0]
    df_model, similarity = _build_similarity_model(df)
    selected_idx = df_model.index[df_model["title"] == selected["title"]].tolist()[0]

    similarity_scores = list(enumerate(similarity[selected_idx]))
    similarity_scores = [
        (idx, score)
        for idx, score in similarity_scores
        if idx != selected_idx and score > 0
    ]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]

    recommendation_rows = []
    for idx, score in similarity_scores:
        row = df_model.iloc[idx].copy()
        row["similarity_score"] = round(float(score), 4)
        row["popularity_score"] = round(float(row["ratings_count"]) / float(df_model["ratings_count"].max()) * 100, 2)
        recommendation_rows.append(row)

    results = pd.DataFrame(recommendation_rows)
    if results.empty:
        return pd.DataFrame(columns=["title", "authors", "average_rating", "ratings_count", "similarity_score", "popularity_score"])

    results = results[["title", "authors", "average_rating", "ratings_count", "similarity_score", "popularity_score"]]
    results = results.sort_values(["similarity_score", "ratings_count"], ascending=[False, False]).reset_index(drop=True)
    return results.head(top_n)


if __name__ == "__main__":
    sample = get_recommendations("Harry Potter and the Prisoner of Azkaban", top_n=10)
    print(sample.head(10).to_string(index=False))
