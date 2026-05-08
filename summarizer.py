"""
summarizer.py
─────────────
Core summarization engine supporting 6 AI/ML algorithms via the Sumy library
plus a fallback TF-IDF implementation using scikit-learn.
"""

from __future__ import annotations
import re
import numpy as np

# Sumy imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# NLTK
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

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Scikit-learn (for custom TF-IDF summarizer)
from sklearn.feature_extraction.text import TfidfVectorizer

LANGUAGE = "english"


class TextSummarizer:
    """
    Unified interface for multiple extractive summarization algorithms.

    Supported algorithms
    --------------------
    - TF-IDF      : Custom scikit-learn–based TF-IDF scoring
    - TextRank    : Graph-based ranking (Sumy)
    - LexRank     : Stochastic graph (Sumy)
    - LSA         : Latent Semantic Analysis (Sumy)
    - Luhn        : Frequency heuristics (Sumy)
    - Edmundson   : Cue + title + position scoring (Sumy)
    """

    def __init__(self) -> None:
        self.stemmer = Stemmer(LANGUAGE)
        self.stop_words = set(stopwords.words("english"))
        self._sumy_summarizers: dict[str, object] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize(
        self,
        text: str,
        algorithm: str = "TF-IDF",
        ratio: float = 0.3,
        num_sentences: int = 5,
    ) -> str:
        """
        Summarize *text* with the chosen *algorithm*.

        Parameters
        ----------
        text          : Input text (already cleaned / pre-processed).
        algorithm     : One of TF-IDF | TextRank | LexRank | LSA | Luhn | Edmundson
        ratio         : Fraction of sentences to keep (0.0–1.0).
        num_sentences : Hard upper limit on output sentences.

        Returns
        -------
        str : Summary text.
        """
        sentences = sent_tokenize(text)
        n = max(1, min(num_sentences, int(len(sentences) * ratio)))

        if algorithm == "TF-IDF":
            return self._tfidf_summarize(text, sentences, n)
        else:
            return self._sumy_summarize(text, algorithm, n)

    # ── TF-IDF (custom sklearn) ───────────────────────────────────────────────

    def _tfidf_summarize(self, text: str, sentences: list[str], n: int) -> str:
        """
        Score each sentence by summing the TF-IDF weights of its constituent
        tokens, then return the top-n sentences in original order.

        Algorithm steps
        ───────────────
        1. Tokenize document into sentences.
        2. Build a TF-IDF matrix (sentences × unique terms).
        3. Score each sentence = mean TF-IDF of its tokens.
        4. Rank sentences by score; take top-n.
        5. Return them sorted by original position.
        """
        if len(sentences) <= n:
            return text

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 1),
                max_features=500,
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Sentence scores = mean of non-zero TF-IDF values in that row
            sentence_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()

            # Pick top-n indices, preserve original order
            top_indices = sorted(
                np.argsort(sentence_scores)[-n:].tolist()
            )
            return " ".join(sentences[i] for i in top_indices)

        except Exception as e:
            # Fallback: return first n sentences
            return " ".join(sentences[:n])

    # ── Sumy-based algorithms ─────────────────────────────────────────────────

    def _sumy_summarize(self, text: str, algorithm: str, n: int) -> str:
        """
        Use Sumy library summarizers (TextRank, LexRank, LSA, Luhn, Edmundson).
        """
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

        summarizer = self._get_sumy_summarizer(algorithm)

        try:
            summary_sentences = summarizer(parser.document, n)
            result = " ".join(str(s) for s in summary_sentences)
            return result if result.strip() else " ".join(
                text.split(".")[:n]
            )
        except Exception:
            # Graceful fallback
            return " ".join(sent_tokenize(text)[:n])

    def _get_sumy_summarizer(self, algorithm: str) -> object:
        """Return (cached) Sumy summarizer instance for *algorithm*."""
        if algorithm in self._sumy_summarizers:
            return self._sumy_summarizers[algorithm]

        algo_map = {
            "TextRank": TextRankSummarizer,
            "LexRank": LexRankSummarizer,
            "LSA": LsaSummarizer,
            "Luhn": LuhnSummarizer,
            "Edmundson": EdmundsonSummarizer,
        }

        cls = algo_map.get(algorithm)
        if cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        inst = cls(self.stemmer)
        inst.stop_words = get_stop_words(LANGUAGE)

        self._sumy_summarizers[algorithm] = inst
        return inst
