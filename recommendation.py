from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).resolve().parent / "books_cleaned.csv"

_CACHED_DF: Optional[pd.DataFrame] = None
_CACHED_TFIDF = None
_CACHED_VECTORIZER: Optional[TfidfVectorizer] = None


def load_books() -> pd.DataFrame:
    """Load cleaned Goodreads catalog with proper column types and indexing."""
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["authors"] = df["authors"].fillna("Unknown").astype(str).str.strip()
    df["publisher"] = df["publisher"].fillna("Unknown").astype(str).str.strip()
    df["language_code"] = df["language_code"].fillna("Unknown").astype(str).str.strip()
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0.0)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)
    _CACHED_DF = df.reset_index(drop=True)
    return _CACHED_DF


def _get_or_build_model() -> Tuple[pd.DataFrame, any]:
    """Build or retrieve cached TF-IDF feature matrix over catalog metadata."""
    global _CACHED_TFIDF, _CACHED_VECTORIZER
    df = load_books()

    if _CACHED_TFIDF is not None:
        return df, _CACHED_TFIDF

    # Content text combines available metadata: title, author, publisher, and language
    content_text = (
        df["title"]
        + " "
        + df["authors"]
        + " "
        + df["publisher"]
        + " "
        + df["language_code"]
    ).str.lower()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25000)
    _CACHED_TFIDF = vectorizer.fit_transform(content_text)
    _CACHED_VECTORIZER = vectorizer
    return df, _CACHED_TFIDF


def get_recommendations(
    book_title: str,
    top_n: int = 10,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return top_n content-based recommendations using TF-IDF and cosine similarity.

    Similarity is computed directly from metadata fields present in the catalog:
    book title, author(s), publisher, and language code.

    Parameters:
        book_title: Query string (exact or partial title).
        top_n: Maximum number of recommendations to return (default: 10).
        df: Optional pre-loaded dataframe.

    Returns:
        pd.DataFrame with columns:
        ['title', 'authors', 'average_rating', 'ratings_count', 'publisher', 'similarity_score']
    """
    if not book_title or not str(book_title).strip():
        return pd.DataFrame(
            columns=["title", "authors", "average_rating", "ratings_count", "publisher", "similarity_score"]
        )

    clean_query = str(book_title).lower().strip()
    catalog, tfidf_matrix = _get_or_build_model()

    # Step 1: Exact match (case-insensitive)
    normalized = catalog["title"].str.lower().str.strip()
    match_mask = normalized == clean_query
    if not match_mask.any():
        # Step 2: Substring match
        match_mask = catalog["title"].str.lower().str.contains(clean_query, regex=False, na=False)

    if not match_mask.any():
        raise ValueError(f"Book '{book_title}' not found in the catalog.")

    selected_idx = catalog[match_mask].index[0]

    # Efficient single-row cosine similarity against entire TF-IDF matrix
    query_vec = tfidf_matrix[selected_idx]
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Zero out the selected book itself
    scores[selected_idx] = 0.0

    # Filter out non-positive similarity
    positive_indices = np.where(scores > 0.0)[0]
    if len(positive_indices) == 0:
        return pd.DataFrame(
            columns=["title", "authors", "average_rating", "ratings_count", "publisher", "similarity_score"]
        )

    # Sort top candidates by similarity score descending, then ratings_count descending
    top_candidate_indices = positive_indices[np.argsort(-scores[positive_indices])[: top_n * 2]]

    results_list = []
    for idx in top_candidate_indices:
        row = catalog.iloc[idx]
        sim = float(scores[idx])
        results_list.append({
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": round(float(row["average_rating"]), 2),
            "ratings_count": int(row["ratings_count"]),
            "publisher": row["publisher"],
            "similarity_score": round(sim, 4),
        })

    results_df = pd.DataFrame(results_list)
    # Deduplicate titles if identical editions exist
    results_df = results_df.drop_duplicates(subset=["title"]).head(top_n)
    return results_df.reset_index(drop=True)


if __name__ == "__main__":
    sample_title = "Harry Potter and the Prisoner of Azkaban"
    print(f"Testing recommendations for '{sample_title}':\n")
    recs = get_recommendations(sample_title, top_n=5)
    print(recs.to_string(index=False))
