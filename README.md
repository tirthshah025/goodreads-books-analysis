# Goodreads Books Analysis & Recommendation Insights

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-Goodreads%20Books-purple)]()

<p align="center">
  <img src="https://img.shields.io/badge/Books-10%2C803-8b5cf6" alt="Books analyzed" />
  <img src="https://img.shields.io/badge/Visualizations-10-22d3ee" alt="Visualizations" />
  <img src="https://img.shields.io/badge/Recommendations-TF-IDF%20%2B%20Cosine-34d399" alt="Recommendations" />
</p>

<p align="center">
  <img src="visualizations/10_top_books_dashboard.png" alt="Goodreads dashboard preview" width="1000" />
</p>

## Overview

This project analyzes the Goodreads Books dataset to uncover reader behavior, publishing trends, author performance, and market patterns across a large catalog of books. The work combines exploratory data analysis, custom visual storytelling, and a content-based recommendation engine to create a polished, portfolio-ready analytics experience.

The repository includes:

- A structured data-cleaning workflow
- A premium-quality exploratory notebook
- Ten publication-ready visualizations
- An interactive Streamlit dashboard
- A TF-IDF and cosine similarity recommendation engine
- A clean GitHub-ready project presentation

## Features

- Data profiling and cleaning on the Goodreads catalog
- Author, publisher, and language analysis
- Publication-year trend analysis
- Ratings vs reviews insight mining
- Top-book and audience reach analysis
- Recommendation engine for similar books
- Streamlit dashboard for interactive exploration
- Exportable filtered tables and recommendation outputs

## Dashboard Preview

Run the dashboard locally:

```bash
streamlit run app.py
```

The dashboard includes the following sections:

- Home Dashboard
- Analytics Dashboard
- Book Explorer
- Author Explorer
- Publisher Analytics
- Language Analytics
- Recommendation Engine

## Visualizations Gallery

<p align="center">
  <img src="visualizations/01_rating_distribution.png" alt="Rating distribution" width="900" />
</p>

<p align="center">
  <img src="visualizations/02_top_20_highest_rated_books.png" alt="Top rated books" width="900" />
</p>

<p align="center">
  <img src="visualizations/03_most_popular_authors.png" alt="Popular authors" width="900" />
</p>

<p align="center">
  <img src="visualizations/04_publication_year_trend.png" alt="Publication trend" width="900" />
</p>

<p align="center">
  <img src="visualizations/05_language_distribution.png" alt="Language distribution" width="900" />
</p>

<p align="center">
  <img src="visualizations/06_ratings_vs_reviews.png" alt="Ratings vs reviews" width="900" />
</p>

<p align="center">
  <img src="visualizations/07_correlation_heatmap.png" alt="Correlation heatmap" width="900" />
</p>

<p align="center">
  <img src="visualizations/08_publisher_analysis.png" alt="Publisher analysis" width="900" />
</p>

<p align="center">
  <img src="visualizations/09_avg_ratings_by_language.png" alt="Average ratings by language" width="900" />
</p>

<p align="center">
  <img src="visualizations/10_top_books_dashboard.png" alt="Executive dashboard" width="900" />
</p>

## Recommendation System

The recommendation system uses a content-based approach built with:

- TF-IDF vectorization over title, author, publisher, and language metadata
- Cosine similarity scoring between books
- Top 10 similar-book recommendations based on a selected title

Example:

```python
from recommendation import get_recommendations

recs = get_recommendations("Harry Potter and the Prisoner of Azkaban", top_n=10)
print(recs.head())
```

## Project Structure

```text
goodreads-books-analysis/
├── app.py                          # Streamlit dashboard
├── recommendation.py              # TF-IDF + cosine similarity engine
├── analysis.ipynb                 # Exploratory data analysis and visual summaries
├── data_cleaning.ipynb            # Cleaning and preprocessing notebook
├── books.csv                      # Raw Goodreads dataset
├── books_cleaned.csv              # Cleaned analysis dataset
├── requirements.txt               # Python dependency list
├── project_report.md              # Detailed project report
├── project_report.pdf             # Exported academic report
├── README.md                      # Portfolio documentation
├── visualizations/                # Generated charts and dashboard images
│   ├── 01_rating_distribution.png
│   ├── 02_top_20_highest_rated_books.png
│   ├── 03_most_popular_authors.png
│   ├── 04_publication_year_trend.png
│   ├── 05_language_distribution.png
│   ├── 06_ratings_vs_reviews.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_publisher_analysis.png
│   ├── 09_avg_ratings_by_language.png
│   └── 10_top_books_dashboard.png
└── archive.zip                    # Legacy backup/archive
```

## Tech Stack

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- Streamlit
- Jupyter Notebook

## Installation

```bash
git clone https://github.com/tirthshah025/goodreads-books-analysis.git
cd goodreads-books-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Results

Key insights from the analysis include:

- Ratings cluster heavily in the 3.7–4.2 range
- Major book franchises dominate popularity metrics
- Popularity and quality are weakly correlated
- English-language content dominates the catalog
- Publishing activity peaks in the late 2000s to early 2010s

## Team Contributions

### Tirth Shah
- Data cleaning
- Dataset preparation
- Preprocessing
- Data validation

### Kavish
- Exploratory analysis
- Visualization design
- Dashboard development
- Recommendation engine

## Acknowledgements

This project is built on the Goodreads Books dataset and designed as a practical analytics portfolio project for data exploration, visual communication, and machine learning-driven recommendations.

---

<p align="center">
  <strong>Goodreads Insight Lab</strong><br>
  A data storytelling project for modern book analytics and personalization.
</p>
