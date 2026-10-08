# Goodreads Books Analysis and Recommendation Insights Using Python
## Final Project Report

**Course:** Data Analysis and Visualization  
**Academic Year:** 2026–2027  
**Submission Date:** October 2026

---

| Field | Details |
|---|---|
| **Project Title** | Goodreads Books Analysis and Recommendation Insights Using Python |
| **Repository** | https://github.com/tirthshah025/goodreads-books-analysis |
| **Contributors** | Tirth Shah (Data Cleaning) · Kavish (EDA & Visualization) |
| **Dataset** | Goodreads Books Dataset — 10,803 records, 12 features |
| **Language** | Python 3.11 |
| **Status** | Complete |

---

## Table of Contents

1. [Project Definition](#1-project-definition)
2. [Dataset Use Case](#2-dataset-use-case)
3. [Methodology](#3-methodology)
4. [Libraries Used](#4-libraries-used)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Findings](#6-findings)
7. [Recommendations](#7-recommendations)
8. [Conclusion](#8-conclusion)
9. [Future Scope](#9-future-scope)

---

## 1. Project Definition

### Background

Goodreads, owned by Amazon, is the world's largest platform for book readers and reviewers, hosting more than 125 million members. Its dataset provides a rich lens into global reading behaviour, preferences, and literary trends. This project leverages a publicly available snapshot of the Goodreads Books dataset to perform a structured, data-driven analysis.

### Problem Statement

With thousands of books published each year and millions of reader reviews available, there is a wealth of untapped insight into:
- What makes a book popular versus highly-rated?
- Which authors and publishers command the most reader trust?
- How have publishing patterns evolved over decades?
- What are the linguistic patterns in reader engagement?

### Goals

This project aims to:
1. Conduct a thorough Exploratory Data Analysis (EDA) of the cleaned Goodreads dataset
2. Generate 10 professional-grade visualizations covering all major analytical dimensions
3. Derive at least 15 meaningful, actionable findings
4. Produce a recommendation-ready insight summary that could inform the design of a book recommendation engine

### Scope

- **In Scope:** EDA, statistical analysis, visualization, insight generation, documentation
- **Out of Scope:** Machine learning model deployment, user interaction data, real-time Goodreads API integration
- **Starting Point:** `books_cleaned.csv` — the pre-processed dataset produced by Tirth Shah

---

## 2. Dataset Use Case

### Dataset Overview

| Property | Value |
|---|---|
| Source | Goodreads Books (Kaggle) |
| Total Records | 10,803 |
| Features | 12 |
| File Size (cleaned) | ~1.5 MB |
| Time Span | Publications from 1900 to 2020 |

### Feature Description

| Feature | Type | Description | Use in Analysis |
|---|---|---|---|
| `bookID` | int | Unique identifier | Filtering |
| `title` | str | Book title | Labels, Top-N analysis |
| `authors` | str | Author name(s) | Author popularity analysis |
| `average_rating` | float | Mean Goodreads rating (0–5) | Core quality metric |
| `isbn` / `isbn13` | str/int | International Standard Book Number | Identifier only |
| `language_code` | str | ISO language code | Language distribution |
| `num_pages` | int | Page count | Correlation analysis |
| `ratings_count` | int | Total ratings received | Popularity metric |
| `text_reviews_count` | int | Number of written reviews | Engagement metric |
| `publication_date` | str | Original publication date | Temporal trend analysis |
| `publisher` | str | Publishing house | Publisher analysis |

### Data Quality (post-cleaning)

After Tirth Shah's cleaning phase:
- No missing values in critical columns (`average_rating`, `ratings_count`)
- Duplicate entries removed
- Data types standardised
- `publication_date` parsed to datetime
- `publication_year` extracted as a derived feature

---

## 3. Methodology

### 3.1 Analysis Pipeline

```
books_cleaned.csv
       │
       ▼
 Data Loading & Validation
       │
       ▼
 Feature Engineering
 (publication_year, sub-datasets)
       │
       ▼
 Univariate Analysis         Bivariate Analysis         Multivariate Analysis
 (rating dist, pages)  ─── (scatter plots) ────────── (heatmap, dashboard)
       │
       ▼
 Group-Level Analysis
 (by author / language / publisher / year)
       │
       ▼
 Visualization Generation (10 charts)
       │
       ▼
 Insight Extraction & Documentation
```

### 3.2 Sub-Dataset Strategy

| Sub-Dataset | Filter | Purpose |
|---|---|---|
| `df_rated` | `average_rating > 0` | Rating distribution analysis |
| `df_valid` | `ratings_count > 50` | Reliable popularity comparisons |
| Top-N filters | `ratings_count >= 1,000` | Highest rated books analysis |
| Language filter | `ratings_count > 100`, `book_count >= 5` | Language quality comparison |

### 3.3 Statistical Approach

- **Descriptive Statistics:** Mean, median, standard deviation, percentiles
- **Correlation Analysis:** Pearson correlation matrix for numeric features
- **Distribution Analysis:** Histogram with KDE-style gradient, skewness assessment
- **Aggregation Analysis:** GroupBy operations for author, publisher, language, year
- **Log Transformation:** Applied to heavily right-skewed features (ratings, reviews) before scatter analysis

### 3.4 Visualization Design Principles

All 10 visualizations follow a unified design system:
- **Dark theme:** Catppuccin Mocha colour palette for premium aesthetics
- **Colour encoding:** Colormaps (`plasma`, `cool`, `magma`, `viridis`) encode magnitude
- **Annotation:** Direct value labels on bars/points for readability
- **Reference lines:** Mean/median overlays for context
- **Dual axes:** Used where two metrics share a chart (publication trend)

---

## 4. Libraries Used

### 4.1 Core Data Libraries

| Library | Version | Role |
|---|---|---|
| **pandas** | 2.x | DataFrame operations, GroupBy aggregations, datetime parsing |
| **numpy** | 1.x | Log transformations, array operations, polynomial fitting |

### 4.2 Visualization Libraries

| Library | Version | Role |
|---|---|---|
| **matplotlib** | 3.x | All primary charts (bar, histogram, scatter, pie, area) |
| **matplotlib.gridspec** | — | Dashboard multi-panel layout |
| **matplotlib.ticker** | — | Custom axis formatters (K/M suffixes) |
| **seaborn** | 0.x | Correlation heatmap with annotation |

### 4.3 Development Tools

| Tool | Role |
|---|---|
| **Jupyter Notebook** | Interactive development and presentation |
| **VS Code / IDE** | Script editing |
| **Git / GitHub** | Version control and project hosting |

### 4.4 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

---

## 5. Exploratory Data Analysis

### 5.1 Visualization 1: Rating Distribution Histogram

**Objective:** Understand the shape and central tendency of the `average_rating` distribution.

**Method:**
- Plotted a frequency histogram with 40 bins
- Applied gradient colouring based on bar height
- Overlaid mean and median as vertical dashed lines
- Added annotation box with key statistics

**Key Statistics:**
- Mean: 3.93
- Median: 3.96
- Std Dev: 0.35
- Range: 0.0 – 5.0

**Interpretation:**
The distribution is unimodal and approximately normal with a very slight left skew. The vast majority of books (>90%) fall between 3.0 and 4.5. Books below 3.0 are rare — this reflects platform-specific bias: readers are unlikely to finish and rate books they strongly dislike.

**Outcome:** Ratings are a high-density signal in the 3.5–4.5 range. Any recommendation model using ratings as a quality signal must account for this narrow effective range and normalise accordingly.

---

### 5.2 Visualization 2: Top 20 Highest Rated Books

**Objective:** Identify the most critically acclaimed books with sufficient readership.

**Method:**
- Filtered to books with ≥1,000 ratings (reliability threshold)
- Ranked by `average_rating` descending
- Plotted top 20 as a horizontal bar chart with plasma colourmap

**Top 5 Books:**

| Rank | Title | Rating |
|---|---|---|
| 1 | The Complete Calvin and Hobbes | 4.82 |
| 2 | Harry Potter Boxed Set Books 1-5 | 4.78 |
| 3 | It's a Magical World (Calvin and Hobbes #11) | 4.76 |
| 4 | Harry Potter Collection (Harry Potter #1-6) | 4.73 |
| 5 | Homicidal Psycho Jungle Cat (Calvin and Hobbes #9) | 4.72 |

**Interpretation:**
Collected editions and complete series dominate because the readership consists of existing fans who have already invested in the full franchise — naturally leading to positive self-selection bias.

**Outcome:** Box-sets and complete collections should be treated as a separate segment in any recommendation system to avoid skewing quality benchmarks for individual books.

---

### 5.3 Visualization 3: Most Popular Authors

**Objective:** Identify authors with the greatest combined reader engagement.

**Method:**
- Grouped by `authors`, summed `ratings_count`
- Selected top 20 by total ratings
- Displayed with cool colourmap horizontal bars

**Top 5 Authors by Total Ratings:**

| Author | Approx. Total Ratings |
|---|---|
| J.K. Rowling | 12M+ |
| Stephenie Meyer | 9M+ |
| Suzanne Collins | 7M+ |
| J.R.R. Tolkien | 5M+ |
| C.S. Lewis | 4M+ |

**Interpretation:**
Franchise authors — those who create self-sustaining narrative universes — dominate. These franchises generate cross-book reading: fans read all books in a series, multiplying engagement across titles.

**Outcome:** Author identity is a powerful cold-start signal for recommendation. Books by highly-engaged authors carry implicit popularity endorsement.

---

### 5.4 Visualization 4: Publication Year Trend

**Objective:** Track the evolution of publishing volume and average quality over time.

**Method:**
- Filtered to 1950–2024
- Created dual-axis chart: book count (bar fill) vs average rating (dashed line)

**Key Observations:**
- 1950–1990: Slow, steady growth (~50–200 books/year)
- 1990–2005: Accelerating growth driven by genre fiction expansion
- 2005–2010: **Peak publishing era** in the dataset
- 2010–2020: Slight decline in dataset representation (lag in Goodreads cataloguing)
- Average ratings are consistently higher for pre-1980 books (survivorship bias)

**Outcome:** The digital publishing revolution (Kindle: 2007, self-publishing proliferation) is clearly visible. Quality dilution correlates with volume increases.

---

### 5.5 Visualization 5: Language Distribution

**Objective:** Assess linguistic diversity of the Goodreads catalogue.

**Language Breakdown:**

| Language | Count | % |
|---|---|---|
| English | 8,673 | 80.3% |
| English (US) | 1,358 | 12.6% |
| English (UK) | 197 | 1.8% |
| Spanish | 207 | 1.9% |
| French | 140 | 1.3% |
| German | 96 | 0.9% |
| Others | ~132 | 1.2% |

**Outcome:** English (all variants) accounts for 94.7% of the dataset. This is a significant limitation for global audience recommendations and should be prominently disclosed in any system built on this data.

---

### 5.6 Visualization 6: Ratings vs Reviews Scatter Plot

**Objective:** Explore the relationship between ratings volume and review volume.

**Method:**
- Log-transformed both axes
- Coloured by `average_rating` using plasma colourmap
- Fitted and overlaid a linear trend line

**Result:** Pearson r = **0.866** — the strongest correlation in the dataset.

**Interpretation:**
The near-linear relationship on log-log axes suggests a power-law relationship: books that attract readers (many ratings) naturally also attract writers (many reviews). The colour gradient shows high-rated books cluster in the upper-right (popular + beloved books).

---

### 5.7 Visualization 7: Correlation Heatmap

**Full Correlation Matrix:**

| Feature | Avg Rating | Num Pages | Ratings Count | Text Reviews |
|---|---|---|---|---|
| Average Rating | 1.000 | +0.152 | +0.038 | +0.034 |
| Num Pages | +0.152 | 1.000 | +0.029 | +0.037 |
| Ratings Count | +0.038 | +0.029 | 1.000 | **+0.866** |
| Text Reviews | +0.034 | +0.037 | **+0.866** | 1.000 |

**Critical Insight:** Average rating is essentially independent of all other features. This confirms that **quality (as rated) and popularity (ratings count) are orthogonal dimensions** — a foundational principle for any recommendation system design.

---

### 5.8 Visualization 8: Publisher Analysis

**Objective:** Compare publishers by volume and quality.

**Top Publishers:**

| Publisher | Books | Avg Rating |
|---|---|---|
| Vintage | 310 | 3.95 |
| Penguin Books | 252 | 3.89 |
| Penguin Classics | 180 | 4.05 |
| Mariner Books | 148 | 4.01 |
| Ballantine Books | 141 | 3.82 |

**Outcome:** Literary imprints curate more carefully, achieving higher quality scores with a smaller catalogue.

---

### 5.9 Visualization 9: Average Ratings by Language

**Top Languages by Average Rating:**

| Language | Avg Rating | Book Count |
|---|---|---|
| Greek | 4.25+ | ~11 |
| Latin | 4.20+ | ~8 |
| Japanese | 4.10+ | 46 |
| English | 3.92 | 8,673+ |
| Spanish | 3.88 | 207 |

**Outcome:** Ancient language books achieve the highest ratings purely due to selection bias — only canonical masterworks in those languages appear on Goodreads.

---

### 5.10 Visualization 10: Dashboard Summary

**Objective:** Provide a single-page executive summary of all key findings.

**KPI Summary:**
- Total Books: 10,803
- Average Rating: 3.93
- Languages: 27
- Total Publishers: 2,290

---

## 6. Findings

### 6.1 Reader Preference Insights

1. **The Goldilocks Rating Zone:** 91% of all books cluster between 3.5 and 4.5. This compressed range suggests Goodreads readers are generally generous and book-positive.

2. **Collection Bias:** Complete series and collector editions receive inflated ratings as they are purchased primarily by committed fans (e.g., Calvin & Hobbes at 4.82).

3. **Depth Preference:** A weak but consistent positive correlation between page count and rating (r = +0.15) suggests readers who invest in longer books tend to find more value.

4. **Quantity-Quality Independence:** The near-zero correlation (r = +0.038) between ratings_count and average_rating is the most policy-relevant finding — a bestseller is not necessarily a better book.

5. **Engagement Multiplier:** Highly-rated books receive proportionally MORE written reviews than average books, suggesting quality drives qualitative engagement.

### 6.2 Book Popularity Insights

6. **The Twilight Effect:** A single book can dominate an entire dataset's popularity metrics — Twilight's 4.59 million ratings are 10× the next-largest contender. Recommendation systems must handle such outliers carefully.

7. **Series Compounding:** Books that are part of successful series benefit from series-level popularity. Standalone ratings often underrepresent the series's true cultural weight.

8. **Review-Rating Power Law:** The 0.866 correlation with log-linear scaling confirms a power-law distribution in reader engagement — a small number of books attract a disproportionate share of reviews.

### 6.3 Author Popularity Insights

9. **The Franchise Ceiling:** The top 5 most-rated authors all created franchise universes (Rowling: HP, Meyer: Twilight, Collins: Hunger Games, Tolkien: LOTR, Lewis: Narnia). Authors without franchises require exceptional per-book ratings to compete in popularity metrics.

10. **Long-Tail Classics:** Authors like Agatha Christie (31 books) and P.G. Wodehouse (39 books) demonstrate that prolific output combined with consistent quality creates a formidable long-tail presence.

11. **The Watterson Anomaly:** Bill Watterson (Calvin & Hobbes) achieves top-chart rating with just one comic strip series — demonstrating that cult classics can rival franchise blockbusters in quality metrics.

### 6.4 Publishing Trend Insights

12. **Digital Revolution Signature:** The sharp increase in publications post-2005 aligns perfectly with Amazon's Kindle launch (2007) and the rise of self-publishing platforms.

13. **Quality-Volume Trade-off:** Average ratings decline slightly as publication volume increases — more titles means lower average curation standards.

14. **Survivorship Amplification:** Pre-1980 books showing on Goodreads are almost exclusively reviewed classics. Their high ratings reflect this selection, not a generalisation that old books are better.

15. **Publisher Concentration:** The top 15 publishers (of 2,290 total) account for ~20% of all books — a classic power-law concentration in an industry dominated by a few major houses.

---

## 7. Recommendations

### 7.1 For Recommendation System Design

**Popularity Floor:** Establish a minimum ratings threshold (suggested: ≥100 ratings) before including a book in recommendations. Below this level, ratings are statistically unreliable.

**Hybrid Scoring:** Do not use average rating alone. Combine:
- `average_rating` × log(`ratings_count`) — a Bayesian-adjusted score
- This naturally down-weights books with few ratings

**Series Handling:** Tag and link series books. Recommend series in order. When a user rates book 1 highly, recommend book 2 before unrelated titles.

### 7.2 For Dataset Improvement

**Language Expansion:** The 94% English bias severely limits cross-cultural recommendations. Augment with Spanish, French, German, Japanese datasets from regional Goodreads equivalents.

**Genre Tagging:** The current dataset lacks genre labels. Adding genre (via Goodreads API or Kaggle supplementary files) would enable content-based filtering.

**Author-Level Metadata:** Adding birth year, nationality, and primary genre to the author dimension would unlock demographic and cultural segmentation.

### 7.3 For Reader Insights

**Discovery Angle:** Publishers like Penguin Classics and Mariner Books can serve as quality proxies for readers seeking literary classics — recommend by publisher for readers who trust curation.

**Temporal Discovery:** Books published 10–20 years ago that still have high average ratings represent a "hidden gems" category — they're not trending but are proven performers.

**Language-Specific Recommendations:** For non-English speakers, the language distribution shows a critical gap. Platform improvement should prioritise Spanish, French, and German catalogues.

---

## 8. Conclusion

This project successfully completed the second half of a two-phase data science project on the Goodreads Books dataset.

**Phase 1** (Tirth Shah): Established a clean, reliable foundation through rigorous data preprocessing — handling missing values, duplicates, type coercions, and date parsing.

**Phase 2** (Kavish): Built upon this foundation to deliver:
- 10 professional, publication-quality visualizations
- 15+ meaningful, layered insights across four analytical dimensions
- A comprehensive project report and README suitable for academic submission and portfolio use

### Critical Takeaways

1. **Popularity ≠ Quality** — the near-zero correlation (r = +0.038) is the single most important finding for recommendation system design
2. **The dataset is English-language centric** (94.7%) — a fundamental limitation to acknowledge
3. **Franchise books dominate both popularity and quality charts** — series mechanics drive exceptional engagement
4. **Classic survivorship bias** inflates ratings of older/ancient-language books
5. **The ratings-reviews power law** (r = 0.866) confirms that platform engagement is self-reinforcing — the popular get more popular

### Academic Value

This analysis demonstrates mastery of:
- End-to-end data analysis workflow
- Professional visualization design
- Statistical reasoning and correlation analysis
- Bias identification in real-world datasets
- Documentation and reproducibility practices

---

## 9. Future Scope

### Short-Term Extensions

| Enhancement | Description | Effort |
|---|---|---|
| Bayesian Rating | Implement Wilson score confidence interval for better ranking | Low |
| Genre Tagging | Scrape or API-fetch genre data for each bookID | Medium |
| Author Network Graph | Visualize co-author relationships | Medium |

### Medium-Term Extensions

| Enhancement | Description | Effort |
|---|---|---|
| Sentiment Analysis | NLP on text reviews to extract qualitative signals beyond star ratings | Medium |
| Content-Based Filter | TF-IDF on titles + genres for similarity-based recommendations | Medium |
| Collaborative Filter | User-book interaction matrix using Matrix Factorization (ALS/SVD) | High |

### Long-Term Extensions

| Enhancement | Description | Effort |
|---|---|---|
| Streamlit Dashboard | Deploy an interactive, web-based exploration tool | Medium |
| Real-Time API | Connect to Goodreads API for live data refreshes | High |
| Multi-Dataset Fusion | Merge with Amazon Books, LibraryThing for richer signal | High |
| LLM Integration | Use large language models to generate personalised book descriptions and recommendations | High |

---

## Appendix: Commit History

| Commit # | Message |
|---|---|
| 1 | Created exploratory data analysis notebook |
| 2 | Added ratings distribution visualization |
| 3 | Added author popularity analysis |
| 4 | Added publication trend analysis |
| 5 | Added correlation heatmap |
| 6 | Generated project findings |
| 7 | Created project README |
| 8 | Uploaded final project report |

---

*Report prepared by Kavish | Dataset cleaned by Tirth Shah*  
*Goodreads Books Analysis Project | October 2026*
