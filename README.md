# Amazon Product Review Sentiment Analysis & Recommendation

MSc Data Science with Advanced Research thesis project — University of Hertfordshire, 2024.
Supervisor: Thiago Matos Pinto.

## Overview

An empirical comparison of text vectorisation techniques and classical ML classifiers
for sentiment classification of Amazon product reviews, with the trained classifier then
used as the sentiment-scoring step in a rule-based product recommendation pipeline.

## Dataset

- **Source:** [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) (Datafiniti, via Kaggle)
- **Size:** ~28,000 reviews across Amazon products (Kindle, Fire TV Stick, batteries, etc.)
- **Not redistributed in this repo** — download from the Kaggle link above.

## Methodology

**Sentiment labels were generated using TextBlob polarity scores** (positive > 0,
negative < 0, neutral = 0). This is important context — see the caveats section below.

**Text preprocessing:**
- Lowercasing, punctuation removal, stopword removal (NLTK)
- Tokenisation + lemmatisation
- Title + body merged into a single review field

**Feature extraction (compared):**
- Bag-of-Words (CountVectorizer) — Unigram and Bigram
- TF-IDF (TfidfVectorizer) — Unigram and Bigram

**Classifiers (compared):**
- Logistic Regression
- Decision Tree
- Multinomial Naive Bayes
- Random Forest
- K-Nearest Neighbours

**Evaluation:** 80/20 train/test split, evaluated on accuracy, precision, recall, F1
(weighted average for multi-class).

## Results

Best combination on the held-out test set:

| Vectoriser | N-gram | Classifier | Accuracy | F1 |
|---|---|---|---|---|
| Bag-of-Words | Unigram | **Logistic Regression** | **97.76%** | 97.74% |
| Bag-of-Words | Unigram | Decision Tree | 96.82% | 96.80% |
| Bag-of-Words | Unigram | Random Forest | 96.45% | 96.32% |
| TF-IDF | Unigram | Random Forest | 96.29% | 96.05% |
| TF-IDF | Unigram | Logistic Regression | 95.94% | 95.63% |

Bag-of-Words with Unigrams consistently outperformed TF-IDF for the top models on
this dataset. Bigrams hurt performance across the board, likely due to sparsity.

## Recommendation pipeline

Rule-based filter applied over reviews scored by the trained classifier:
1. Filter to user's chosen primary category
2. Keep only reviews predicted as Positive sentiment
3. Keep only reviews with polarity > 0.8
4. Keep only reviews with rating > 4.2
5. Return top 5 by rating

## Honest caveats

A few things worth flagging openly, in the interest of not overselling the result:

- **The 97.76% accuracy measures agreement with TextBlob's lexicon-based labels,
  not with human-annotated ground truth.** The classifier is effectively learning to
  reproduce TextBlob's rules from text features. A more rigorous evaluation would
  require a hand-labelled test set, which was out of scope for this MSc.
- **Severe class imbalance.** Of ~28,000 reviews, ~25,200 are Positive vs. ~1,600
  Negative and ~1,500 Neutral. The accuracy number is dominated by performance on
  the positive class; per-class precision/recall on Negative and Neutral are lower
  (Negative recall = 0.84, Neutral recall = 0.91).
- **Recommendation step is rule-based, not a learned recommender.** No collaborative
  filtering, embedding-based similarity, or learned ranking — it's threshold filters
  over the sentiment classifier's output.
- **Classical ML only.** No LLMs, transformers, or deep learning. This work predates
  my later production work with LLMs on an EdTech platform.

## Stack

Python · scikit-learn · NLTK · TextBlob · Pandas · NumPy · Matplotlib · Seaborn

## Files

- `thesis.pdf` — full writeup with methodology, literature review, and results
- `code/` — Python source from the appendix of the thesis
- `README.md` — this file

## Citation

If referencing this work:
> Ponnambalam, S. (2024). *Amazon Product Review Analysis and Recommendations.*
> MSc Data Science with Advanced Research thesis, University of Hertfordshire.
