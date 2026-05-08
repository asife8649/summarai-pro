"""
core.py
-------
Pure TF-IDF extractive summarisation — no external API needed.

Algorithm steps
───────────────
1. Tokenise text into sentences (NLTK punkt).
2. Build TF-IDF matrix (sentences × terms) via scikit-learn.
3. Score each sentence = sum of its TF-IDF values.
4. Rank sentences; pick top-n preserving original order.
5. Return joined summary.
"""

from __future__ import annotations
import re, math
from collections import Counter
from typing import List

import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_sw

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = set(nltk_sw.words("english"))


# ── Main summariser ───────────────────────────────────────────────────────────

def tfidf_summarize(text: str, n: int = 5) -> str:
    """
    Extractive summarisation using TF-IDF sentence scoring.

    Parameters
    ----------
    text : Cleaned input text (≤ 10 000 words enforced in app.py).
    n    : Number of sentences to return.

    Returns
    -------
    str  : Summary (sentences joined with space).
    """
    text = _clean(text)
    sentences = sent_tokenize(text)

    if not sentences:
        return ""
    if len(sentences) <= n:
        return " ".join(sentences)

    # Cap n to sentence count
    n = min(n, len(sentences))

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            max_features=1000,
            sublinear_tf=True,          # log TF dampening
        )
        mat = vec.fit_transform(sentences)          # shape: (S, T)
        scores = np.asarray(mat.sum(axis=1)).flatten()   # sentence score = row sum

        top_idx = sorted(np.argsort(scores)[-n:].tolist())   # preserve order
        return " ".join(sentences[i] for i in top_idx)

    except Exception:
        # Fallback: first n sentences
        return " ".join(sentences[:n])


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    """
    TF × log(N/df) keyword scoring per sentence-corpus.
    """
    sentences = sent_tokenize(text)
    N = max(len(sentences), 1)

    words = [
        w.lower() for w in word_tokenize(text)
        if w.isalpha() and len(w) > 3 and w.lower() not in STOPWORDS
    ]
    if not words:
        return []

    tf = Counter(words)
    df: Counter = Counter()
    for s in sentences:
        seen = {w.lower() for w in word_tokenize(s) if w.isalpha()}
        for w in seen:
            df[w] += 1

    scored = {
        w: cnt * math.log((N + 1) / (df.get(w, 0) + 1))
        for w, cnt in tf.items()
    }
    return sorted(scored, key=scored.get, reverse=True)[:top_n]


# ── Readability ───────────────────────────────────────────────────────────────

def _syllables(word: str) -> int:
    word = word.lower().strip(".:;?!")
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    return max(len(re.findall(r'[aeiouy]{1,2}', word)), 1)


def readability_score(text: str) -> float:
    """Flesch Reading Ease (0–100)."""
    sents = sent_tokenize(text)
    words = [w for w in word_tokenize(text) if w.isalpha()]
    ns = max(len(sents), 1)
    nw = max(len(words), 1)
    nsy = sum(_syllables(w) for w in words)
    score = 206.835 - 1.015 * (nw / ns) - 84.6 * (nsy / nw)
    return round(max(0.0, min(100.0, score)), 1)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
