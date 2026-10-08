from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from recommendation import get_recommendations


DATA_PATH = Path(__file__).resolve().parent / "books_cleaned.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["publication_year"] = df["publication_date"].dt.year
    df["author_count"] = df["authors"].fillna("").str.split("/").str.len()
    return df


def render_metric_card(title: str, value: str, delta: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent};">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home(df: pd.DataFrame) -> None:
    total_books = len(df)
    avg_rating = df["average_rating"].mean()
    total_ratings = df["ratings_count"].sum()
    top_author = (
        df.assign(authors=df["authors"].str.split("/")).explode("authors")
        .groupby("authors")["ratings_count"].sum()
        .sort_values(ascending=False)
        .index[0]
    )

    st.markdown("## Home Dashboard")
    st.caption("Portfolio-grade book analytics for the Goodreads catalog")

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Books in catalog", f"{total_books:,}", "+0.0% vs baseline", "#7c3aed")
    with cols[1]:
        render_metric_card("Avg. rating", f"{avg_rating:.2f}/5", "Strong reader sentiment", "#22c55e")
    with cols[2]:
        render_metric_card("Total ratings", f"{total_ratings:,.0f}", "Community engagement", "#06b6d4")
    with cols[3]:
        render_metric_card("Top author", top_author, "Largest audience reach", "#f59e0b")

    top_books = df.nlargest(10, "ratings_count")[["title", "authors", "average_rating", "ratings_count"]].copy()
    top_books["ratings_count"] = top_books["ratings_count"].map(lambda x: f"{x:,.0f}")

    col1, col2 = st.columns([1.7, 1.3])
    with col1:
        st.markdown("### Top 10 books by readership")
        st.dataframe(top_books.reset_index(drop=True), use_container_width=True, hide_index=True)
    with col2:
        rating_hist = px.histogram(
            df,
            x="average_rating",
            nbins=30,
            color_discrete_sequence=["#8b5cf6"],
            title="Rating distribution",
            range_x=[2.5, 5],
        )
        rating_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            title_font={"size": 18},
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(rating_hist, use_container_width=True)


def render_analytics(df: pd.DataFrame) -> None:
    st.markdown("## Analytics Dashboard")

    year_counts = df.groupby("publication_year").size().reset_index(name="book_count")
    year_counts = year_counts[year_counts["publication_year"].notna()]
    year_chart = px.line(
        year_counts,
        x="publication_year",
        y="book_count",
        markers=True,
        title="Publication volume over time",
        color_discrete_sequence=["#7c3aed"],
    )
    year_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        margin=dict(l=20, r=20, t=30, b=20),
    )

    reviews_scatter = px.scatter(
        df,
        x="ratings_count",
        y="text_reviews_count",
        size="average_rating",
        hover_name="title",
        title="Reviews vs ratings",
        color="average_rating",
        color_continuous_scale="Viridis",
        log_x=True,
        log_y=True,
        range_color=[3.0, 5.0],
    )
    reviews_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        margin=dict(l=20, r=20, t=30, b=20),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(year_chart, use_container_width=True)
    with col2:
        st.plotly_chart(reviews_scatter, use_container_width=True)

    publisher_summary = (
        df.groupby("publisher", as_index=False)["ratings_count"]
        .sum()
        .sort_values("ratings_count", ascending=False)
        .head(10)
    )
    pub_chart = px.bar(
        publisher_summary,
        x="publisher",
        y="ratings_count",
        color="ratings_count",
        title="Top publishers by audience reach",
        color_continuous_scale="Plasma",
    )
    pub_chart.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=30, b=60),
    )
    st.plotly_chart(pub_chart, use_container_width=True)


def render_book_explorer(df: pd.DataFrame) -> None:
    st.markdown("## Book Explorer")
    filters = st.columns(3)
    min_rating = filters[0].slider("Minimum rating", 0.0, 5.0, 3.5, step=0.1)
    min_ratings = filters[1].slider("Minimum ratings", 0, 5000000, 100, step=100)
    selected_publisher = filters[2].selectbox("Publisher", ["All"] + sorted(df["publisher"].dropna().unique().tolist()))

    filtered = df.copy()
    filtered = filtered[filtered["average_rating"] >= min_rating]
    filtered = filtered[filtered["ratings_count"] >= min_ratings]
    if selected_publisher != "All":
        filtered = filtered[filtered["publisher"] == selected_publisher]

    st.caption(f"Showing {len(filtered):,} books")
    st.dataframe(
        filtered[["title", "authors", "average_rating", "ratings_count", "publisher", "language_code"]]
        .sort_values(["ratings_count", "average_rating"], ascending=[False, False])
        .head(200)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    csv = filtered.to_csv(index=False)
    st.download_button("Download filtered table", csv, "goodreads_filtered_books.csv", "text/csv")


def render_author_explorer(df: pd.DataFrame) -> None:
    st.markdown("## Author Explorer")
    exploded = df.assign(authors=df["authors"].str.split("/")).explode("authors")
    author_summary = (
        exploded.groupby("authors", as_index=False)
        .agg(total_books=("title", "count"), total_ratings=("ratings_count", "sum"), avg_rating=("average_rating", "mean"))
        .sort_values(["total_ratings", "avg_rating"], ascending=[False, False])
        .head(15)
    )

    fig = px.bar(
        author_summary,
        x="authors",
        y="total_ratings",
        color="avg_rating",
        title="Top authors by total audience reach",
        color_continuous_scale="Sunset",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        xaxis_tickangle=-35,
        margin=dict(l=20, r=20, t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(author_summary, use_container_width=True, hide_index=True)


def render_publisher_analytics(df: pd.DataFrame) -> None:
    st.markdown("## Publisher Analytics")
    publisher_stats = (
        df.groupby("publisher", as_index=False)
        .agg(book_count=("title", "count"), avg_rating=("average_rating", "mean"), total_ratings=("ratings_count", "sum"))
        .sort_values(["book_count", "avg_rating"], ascending=[False, False])
        .head(15)
    )
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(publisher_stats, x="publisher", y="book_count", color="avg_rating", title="Books published by publisher")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(publisher_stats, x="publisher", y="avg_rating", color="total_ratings", title="Average rating by publisher", color_continuous_scale="Magma")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"}, xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(publisher_stats, use_container_width=True, hide_index=True)


def render_language_analytics(df: pd.DataFrame) -> None:
    st.markdown("## Language Analytics")
    lang_summary = (
        df.groupby("language_code", as_index=False)
        .agg(book_count=("title", "count"), avg_rating=("average_rating", "mean"), total_ratings=("ratings_count", "sum"))
        .sort_values("book_count", ascending=False)
        .head(12)
    )
    pie = px.pie(lang_summary, names="language_code", values="book_count", hole=0.45, title="Language mix")
    pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})

    bar = px.bar(lang_summary, x="language_code", y="avg_rating", color="total_ratings", title="Average rating by language")
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie, use_container_width=True)
    with col2:
        st.plotly_chart(bar, use_container_width=True)

    st.dataframe(lang_summary, use_container_width=True, hide_index=True)


def render_recommendation_engine(df: pd.DataFrame) -> None:
    st.markdown("## Recommendation Engine")
    st.caption("Content-based recommendations using TF-IDF and cosine similarity")

    book_titles = sorted(df["title"].dropna().unique().tolist())
    selected_title = st.selectbox("Choose a book", book_titles)
    top_n = st.slider("Number of recommendations", 5, 10, 10)

    if selected_title:
        recs = get_recommendations(selected_title, top_n=top_n)
        st.markdown(f"### Similar books for: **{selected_title}**")
        st.dataframe(recs[["title", "authors", "average_rating", "ratings_count"]], use_container_width=True, hide_index=True)

        csv = recs.to_csv(index=False)
        st.download_button("Download recommendations", csv, f"{selected_title.lower().replace(' ', '_')}_recommendations.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="Goodreads Insight Lab", page_icon="📚", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-strong: rgba(15, 23, 42, 0.96);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #a5b4cf;
            --accent: #8b5cf6;
            --accent-2: #22d3ee;
        }
        html, body, [data-testid="stApp"] {
            background: radial-gradient(circle at top left, rgba(139,92,246,0.18), transparent 30%),
                        radial-gradient(circle at top right, rgba(34,211,238,0.12), transparent 25%),
                        #07111f;
            color: var(--text);
        }
        .stApp {
            background: transparent;
        }
        .stSidebar {
            background: rgba(15, 23, 42, 0.8);
            border-right: 1px solid var(--border);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 15px 35px rgba(15, 23, 42, 0.3);
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.7rem;
            font-weight: 800;
            color: #fff;
            margin-bottom: 0.3rem;
        }
        .metric-delta {
            color: #b3c2d9;
            font-size: 0.85rem;
        }
        h1, h2, h3 {
            color: #f8fafc;
        }
        .stDataFrame, .stTable {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 14px;
            overflow: hidden;
        }
        .stSelectbox, .stSlider, .stDownloadButton > button {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    nav = st.sidebar.radio(
        "Navigation",
        ["Home Dashboard", "Analytics Dashboard", "Book Explorer", "Author Explorer", "Publisher Analytics", "Language Analytics", "Recommendation Engine"],
        index=0,
    )

    if nav == "Home Dashboard":
        render_home(df)
    elif nav == "Analytics Dashboard":
        render_analytics(df)
    elif nav == "Book Explorer":
        render_book_explorer(df)
    elif nav == "Author Explorer":
        render_author_explorer(df)
    elif nav == "Publisher Analytics":
        render_publisher_analytics(df)
    elif nav == "Language Analytics":
        render_language_analytics(df)
    else:
        render_recommendation_engine(df)


if __name__ == "__main__":
    main()
