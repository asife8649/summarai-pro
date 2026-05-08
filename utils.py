"""
utils.py
────────
Helper functions for text preprocessing, metrics, and keyword extraction.
"""

from __future__ import annotations
import re
import math
from collections import Counter
from typing import List

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

STOPWORDS = set(stopwords.words("english"))


# ── Text cleaning ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Light cleaning:
    - Collapse whitespace
    - Remove non-printable characters
    - Normalise quotes and dashes
    """
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


# ── Basic stats ───────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Return word count (simple whitespace split)."""
    return len(text.split()) if text.strip() else 0


def count_sentences(text: str) -> int:
    """Return sentence count using NLTK tokenizer."""
    return len(sent_tokenize(text)) if text.strip() else 0


# ── Keyword extraction (TF-IDF-like heuristic) ───────────────────────────────

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top-n keywords using a simple TF × log(N/df) heuristic.

    Steps
    -----
    1. Tokenize into words, lowercase, filter stopwords & short words.
    2. Compute term frequency (TF) across the whole document.
    3. Treat each sentence as a "document" for IDF estimation.
    4. Score = TF × IDF; return top-n.
    """
    sentences = sent_tokenize(text)
    N = max(len(sentences), 1)

    words = [
        w.lower()
        for w in word_tokenize(text)
        if w.isalpha() and len(w) > 3 and w.lower() not in STOPWORDS
    ]

    if not words:
        return []

    tf = Counter(words)

    # Document frequency (sentence-level)
    df: Counter = Counter()
    for sent in sentences:
        sent_words = {
            w.lower()
            for w in word_tokenize(sent)
            if w.isalpha()
        }
        for w in sent_words:
            df[w] += 1

    # TF-IDF score
    scores = {
        w: count * math.log((N + 1) / (df.get(w, 0) + 1))
        for w, count in tf.items()
    }

    top = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return top


# ── Readability ───────────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Approximate syllable count for a single English word."""
    word = word.lower().strip(".:;?!")
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'^y', '', word)
    count = len(re.findall(r'[aeiouy]{1,2}', word))
    return max(count, 1)


def readability_score(text: str) -> float:
    """
    Flesch Reading Ease score.

    Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)

    Range
    -----
    90–100 : Very easy
    60–70  : Standard
    0–30   : Very difficult
    """
    sentences = sent_tokenize(text)
    words = [w for w in word_tokenize(text) if w.isalpha()]

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)
    num_syllables = sum(_count_syllables(w) for w in words)

    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return round(max(0.0, min(100.0, score)), 1)
