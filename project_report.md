# Goodreads Books Analysis and Recommendation Insights Using Python
## Final Academic Project Report

**Course:** Data Analysis and Visualization  
**Academic Year:** 2026–2027  
**Submission Date:** October 2026

---

| Field | Details |
|---|---|
| **Project Title** | Goodreads Books Analysis and Recommendation Insights Using Python |
| **Repository** | https://github.com/tirthshah025/goodreads-books-analysis |
| **Contributors** | **Tirth Shah** (Data Cleaning & Verification) &bull; **Kavish Patel** (EDA & Visualization) |
| **Dataset** | Goodreads Books Dataset — 10,803 cleaned records, 12 features |
| **Language & Environment** | Python 3.11, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit |
| **Status** | Complete & Verified |

---

## Table of Contents

1. [Project Definition](#1-project-definition)
2. [Dataset Use Case & Audit](#2-dataset-use-case--audit)
3. [Methodology](#3-methodology)
4. [Libraries & Environment](#4-libraries--environment)
5. [Exploratory Data Analysis (Core Visualizations)](#5-exploratory-data-analysis)
6. [Visualization Simplification Rationale](#6-visualization-simplification-rationale)
7. [Statistical Findings](#7-statistical-findings)
8. [Recommendation System Design](#8-recommendation-system-design)
9. [Streamlit Interactive Application](#9-streamlit-interactive-application)
10. [Dataset Limitations](#10-dataset-limitations)
11. [Team Contributions](#11-team-contributions)
12. [Conclusion](#12-conclusion)

---

## 1. Project Definition

### Background
Goodreads, owned by Amazon, is the world's largest community for book readers and reviewers. Its catalog provides an empirical lens into reader evaluation patterns, literary reach, publisher output, and temporal archiving dynamics. This project conducts an academic, data-driven analysis using a verified snapshot of the Goodreads catalog.

### Problem Statement
With thousands of titles published annually, data science can address key empirical questions:
- What does the true distribution of reader scores look like, and is there evidence of rating bias?
- Which authors and publishers dominate catalog volume?
- Does high community readership (ratings count) correlate with higher appraisal (average rating)?
- How have cataloged publications evolved over time, and what archival limitations exist?
- How can catalog metadata drive transparent, content-based book recommendations without fabricating user history?

### Goals
1. Execute a verified data-cleaning pipeline with automated assertion testing.
2. Produce a simplified, academically rigorous exploratory visualization suite.
3. Replace unnecessary heatmaps and unsupported visualizations with clear, interpretable charts.
4. Deploy a self-explanatory Streamlit dashboard with restrained academic aesthetics.
5. Provide a fast, transparent TF-IDF and cosine similarity recommendation tool.

---

## 2. Dataset Use Case & Audit

### Dataset Overview
- **Raw Input:** `books.csv` with 11,128 lines (1 header + 11,127 data rows).
- **Loaded Records:** 11,123 records initially loaded after 4 malformed lines with delimiter shifting were skipped (`on_bad_lines='skip'`).
- **Cleaned Records:** 10,803 records across 12 standardized columns in `books_cleaned.csv`.
- **Integrity Checks:** 6/6 automated assertion tests successfully passed in `data_cleaning.ipynb`.

### Verified Feature Schema

| Feature | Type | Description | Analytical Role |
|---|---|---|---|
| `bookID` | int64 | Unique book identifier | Primary key / Uniqueness check |
| `title` | str | Title of the book edition | Identification, labels, TF-IDF feature |
| `authors` | str | Author(s), delimited by `/` | Volume and co-author analysis, TF-IDF feature |
| `average_rating` | float64 | Mean Goodreads rating (0.00–5.00) | Primary quality metric |
| `isbn` / `isbn13` | str | Standard book identifiers | Integrity / deduplication |
| `language_code` | str | ISO language code (e.g. `eng`, `spa`) | Linguistic distribution, TF-IDF feature |
| `num_pages` | int64 | Number of pages | Book length analysis |
| `ratings_count` | int64 | Total community ratings received | Popularity / audience reach |
| `text_reviews_count` | int64 | Total written reviews submitted | Engagement depth |
| `publication_date` | str | Standardized ISO date (`YYYY-MM-DD`) | Temporal trend extraction |
| `publisher` | str | Publishing house | Market volume analysis, TF-IDF feature |

---

## 3. Methodology

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Data Ingestion, Cleaning & Verification       │
│ Lead: Tirth Shah                                       │
│ • Skip 4 malformed delimiter-shifted records           │
│ • Strip header whitespace ('  num_pages' -> 'num_pages')│
│ • Resolve invalid calendar dates (e.g., Nov 31)        │
│ • Enforce relational validity: reviews <= ratings      │
│ • Execute 6 automated assertion checks -> books_cleaned│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Exploratory Data Analysis & Visualization     │
│ Lead: Kavish Patel                                     │
│ • Distribution analysis: Mean=3.94, Median=3.96        │
│ • Author & Publisher volume ranking                    │
│ • Popularity vs Quality hypothesis test (r = +0.038)   │
│ • Temporal trend & archival truncation identification  │
│ • Generation of 6 core publication-grade charts        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Interactive Dashboard & Recommendation Engine │
│ Team Collaboration: Tirth Shah & Kavish Patel          │
│ • Streamlit web application with academic styling      │
│ • Self-explanatory chart captions & dynamic KPIs       │
│ • Instantaneous TF-IDF + Cosine Similarity Recommender │
└────────────────────────────────────────────────────────┘
```

---

## 4. Libraries & Environment

- **Python 3.11**
- **pandas (2.x):** Ingestion, schema audit, aggregation, datetime parsing.
- **NumPy (1.x):** Numeric operations, log transformations, vector indexing.
- **Matplotlib (3.x):** Core academic figures and publication charts.
- **Seaborn (0.x):** Supplemental styling utilities.
- **Plotly (5.x):** Interactive web dashboard visualizations.
- **scikit-learn (1.x):** `TfidfVectorizer` and `cosine_similarity`.
- **Streamlit (1.x):** University project dashboard interface.

---

## 5. Exploratory Data Analysis

### 5.1 Visualization 1: Distribution of Average Book Ratings (Histogram)
- **Objective:** Examine the distribution of reader appraisal scores.
- **Method:** 40-bin frequency histogram for books with ratings (excluding 75 unrated titles).
- **Statistics:** Mean = 3.94, Median = 3.96, Std Dev = 0.35.
- **Interpretation:** Ratings cluster heavily between 3.50 and 4.50. Ratings below 2.50 are rare due to platform self-selection bias.

### 5.2 Visualization 2: Top 10 Authors by Number of Books (Horizontal Bar Chart)
- **Objective:** Identify the most prolific authors by distinct cataloged editions.
- **Method:** Split multi-author entries on `/`, aggregate counts, and rank the top 10.
- **Findings:** William Shakespeare (91 books), Stephen King (83 books), and J.R.R. Tolkien (50 books) lead catalog volume.

### 5.3 Visualization 3: Top 10 Most-Rated Books (Popularity by Ratings Count)
- **Objective:** Identify books with the greatest community reach.
- **Method:** Ranked by `ratings_count`, displaying count in millions alongside average rating.
- **Findings:** *Twilight* leads with 4.60M ratings (3.59 average rating), followed by *The Hobbit* (2.53M, 4.27 rating) and *The Catcher in the Rye* (2.46M, 3.80 rating). This highlights that popularity is distinct from critical acclaim.

### 5.4 Visualization 4: Publication Year Trend (1950–2020) with Coverage Limitation
- **Objective:** Assess publishing volume over time while documenting archival boundaries.
- **Method:** Annual book count from 1950 to 2020 with explicit warning banner for post-2007.
- **Findings:** Catalog volume peaked in 2006 (1,665 books). Post-2007 records drop sharply (only 103 titles between 2011 and 2020) due to archival snapshot timing, not an industry decline.

### 5.5 Visualization 5: Average Rating vs. Ratings Count (Scatter Plot)
- **Objective:** Empirically test whether popular books receive systematically higher ratings.
- **Method:** Scatter plot of average rating vs. log-transformed ratings count with Pearson correlation.
- **Findings:** Pearson r = **+0.038** (near zero). Quality and popularity are statistically independent dimensions.

### 5.6 Visualization 6: Top 10 Publishers by Book Volume (Horizontal Bar Chart)
- **Objective:** Determine publishing houses providing the foundation of the catalog.
- **Method:** Count titles per publisher and display top 10.
- **Findings:** *Vintage* (310 books), *Penguin Books* (252 books), and *Penguin Classics* (180 books) lead the catalog.

---

## 6. Visualization Simplification Rationale

Following professor feedback that the heatmap and an excessive chart/map looked unnecessary, the visualization suite was revised:

1. **Removal of Correlation Heatmap:**
   The numeric correlation matrix had only 4 variables (`average_rating`, `num_pages`, `ratings_count`, `text_reviews_count`). Three correlations were near zero and one was redundant with the scatter plot. Replacing the heatmap with the direct scatter plot of **Average Rating vs. Ratings Count** communicates the key empirical finding (r = +0.038) far more clearly.

2. **Removal of Dashboard Summary Grid & Unsupported Language Ranking:**
   The artificial 10-subplot matplotlib dashboard collage (`10_top_books_dashboard.png`) was redundant with the actual interactive Streamlit dashboard. Additionally, the previous "Average Ratings by Language" chart compared languages with 8–11 titles (Latin, Greek) against 8,600+ English titles, which was statistically ungrounded. These were removed in favor of the clean 6-chart core suite.

3. **Correction of "Highest-Rated Books" Chart:**
   The former chart ranked complete series box-sets (Calvin & Hobbes, Harry Potter boxed sets) as "highest rated books". This was replaced by **Most-Rated Books**, clearly labeled as audience reach/popularity.

---

## 7. Statistical Findings

1. **Compressed Rating Scale:** Over 90% of rated books fall between 3.50 and 4.50 (sample mean: 3.94, median: 3.96).
2. **Quality-Popularity Independence:** Pearson r = +0.038 proves that commercial reach does not imply higher reader appraisal.
3. **Engagement Linearity:** Ratings count and review count correlate strongly (r = +0.866).
4. **Power-Law Readership:** The top title (*Twilight*) commands 4.60M ratings, dominating catalog attention.
5. **Archival Truncation:** 68% of titles date from 2000–2007; sparse coverage post-2007 must be treated as a collection limitation.
6. **Anglo-Centricity:** 94.7% of titles are in English, limiting global generalizations.

---

## 8. Recommendation System Design

The project implements a content-based recommendation engine in `recommendation.py`:
- **Features Used:** `title`, `authors`, `publisher`, `language_code`.
- **Method:** `TfidfVectorizer` (unigrams and bigrams) + single-row `cosine_similarity`.
- **Optimization:** Caches the fitted TF-IDF matrix, achieving **<0.01s** query latency rather than computing dense NxN similarity matrices on each call.
- **Transparency:** Clearly discloses that recommendations are based on bibliographic metadata without fabricating genres or user profiles.

---

## 9. Streamlit Interactive Application

The Streamlit app (`app.py`) follows a restrained academic layout:
- Neutral background (`#f8fafc`) with clean card borders (`#e2e8f0`).
- Self-explanatory structure: Overview -> Data Cleaning Audit -> EDA with "What is shown" and "What the data suggests" explanations -> Key Findings -> Recommendation Tool -> Methodology & Contributions.
- Dynamic KPIs and responsive Plotly charts.

---

## 10. Dataset Limitations

- **Language Bias:** 94.7% English editions; minimal coverage of non-English literary traditions.
- **Temporal Truncation:** Snapshot dates from the mid-2000s; lacks contemporary 2015–2026 publishing representation.
- **Self-Selection Bias:** Voluntary user ratings create an inflated rating baseline (mean 3.94).
- **Metadata Scope:** Lacks synopsis text and genre taxonomy; recommendations reflect bibliographic similarity.

---

## 11. Team Contributions

- **Tirth Shah:** Phase 1 Data Cleaning & Engineering, parsing anomaly resolution, whitespace stripping, date standardization, review count consistency enforcement, 6/6 automated test assertions, and dataset export (`books_cleaned.csv`).
- **Kavish Patel:** Phase 2 Exploratory Data Analysis, statistical hypothesis formulation, academic visualization design, insight extraction, and documentation.
- **Joint Collaboration:** Streamlit dashboard architecture (`app.py`), recommendation engine optimization (`recommendation.py`), and final project reporting.

---

## 12. Conclusion

This project delivers a clear, credible, and academically rigorous data science analysis. By simplifying visualizations, removing the unnecessary heatmap and excessive dashboard collage, grounding every claim in verified data calculations, and deploying a responsive Streamlit tool, the project satisfies all university submission standards.
