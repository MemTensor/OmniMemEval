"""Shared NLP evaluation metrics for LoCoMo and LongMemEval.

Centralises ROUGE, BLEU, METEOR, BERTScore, semantic similarity,
F1, and the combined ``calculate_nlp_metrics`` so LLM-as-Judge benchmarks
share a single implementation.

Also provides ``extract_label_json`` and the ``LLMGrade`` Pydantic model
used by grader functions.

Usage::

    from utils.nlp_metrics import (
        calculate_nlp_metrics, extract_label_json, LLMGrade,
        init_nlp, get_encoding,
    )
"""

from __future__ import annotations

import json
import re

import nltk
import tiktoken
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from pydantic import BaseModel, Field
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine

from utils.resources import ensure_sentence_model, init_eval_resources

_encoding = None
_sentence_model = None


def init_nlp() -> None:
    """Call once at module startup to download NLTK data.

    The SentenceTransformer model is lazy-loaded on first call to
    ``calculate_semantic_similarity()``, so the ~1.2 GB download is
    only triggered when ``--options semantic`` is actually used.
    """
    init_eval_resources()


def get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


# ── Pydantic model ────────────────────────────────────────────────────────────


class LLMGrade(BaseModel):
    llm_judgment: str = Field(description="CORRECT or WRONG")
    llm_reasoning: str = Field(description="Explain why the answer is correct or incorrect.")


# ── Label extraction ──────────────────────────────────────────────────────────


def extract_label_json(text: str) -> str | None:
    """Extract ``{"label": "VALUE"}`` from LLM grader output."""
    pattern = r'\{\s*"label"\s*:\s*["\']([^"\']*)["\']\s*\}'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None


# ── Individual metrics ────────────────────────────────────────────────────────


def calculate_rouge_scores(gold_answer: str, response: str) -> dict[str, float]:
    metrics = {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(gold_answer, response)
        metrics["rouge1_f"] = scores["rouge1"].fmeasure
        metrics["rouge2_f"] = scores["rouge2"].fmeasure
        metrics["rougeL_f"] = scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Failed to calculate ROUGE scores: {e}")
    return metrics


def calculate_bleu_scores(gold_tokens: list[str], response_tokens: list[str]) -> dict[str, float]:
    metrics = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    try:
        smoothing = SmoothingFunction().method1
        weights = [
            (1, 0, 0, 0), (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25),
        ]
        for i, weight in enumerate(weights, 1):
            metrics[f"bleu{i}"] = sentence_bleu(
                [gold_tokens], response_tokens,
                weights=weight, smoothing_function=smoothing,
            )
    except ZeroDivisionError:
        pass
    except Exception as e:
        print(f"Failed to calculate BLEU scores: {e}")
    return metrics


def calculate_meteor_score(gold_tokens: list[str], response_tokens: list[str]) -> float:
    try:
        return meteor_score([gold_tokens], response_tokens)
    except Exception as e:
        print(f"Failed to calculate METEOR score: {e}")
        return 0.0


def calculate_semantic_similarity(gold_answer: str, response: str) -> float:
    global _sentence_model
    try:
        if _sentence_model is None:
            _sentence_model = ensure_sentence_model()
        if _sentence_model is None:
            return 0.0
        gold_emb = _sentence_model.encode([gold_answer], show_progress_bar=False)[0]
        resp_emb = _sentence_model.encode([response], show_progress_bar=False)[0]
        return 1 - cosine(gold_emb, resp_emb)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def calculate_f1_score(gold_tokens: list[str], response_tokens: list[str]) -> float:
    try:
        gold_set = set(gold_tokens)
        response_set = set(response_tokens)
        if len(gold_set) == 0 or len(response_set) == 0:
            return 0.0
        precision = len(gold_set & response_set) / len(response_set)
        recall = len(gold_set & response_set) / len(gold_set)
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0
    except Exception as e:
        print(f"Failed to calculate F1 score: {e}")
        return 0.0


# ── Combined metric calculation ───────────────────────────────────────────────


def calculate_nlp_metrics(gold_answer: str, response: str, context: str, options: list[str] | None = None) -> dict:
    """Compute lexical + semantic metrics for a single QA pair.

    Returns a dict with ``context_tokens`` and optional ``lexical`` /
    ``semantic`` sub-dicts depending on *options*.
    """
    if options is None:
        options = ["lexical"]

    gold_answer = str(gold_answer) if gold_answer is not None else ""
    response = str(response) if response is not None else ""
    context = str(context) if context is not None else ""

    enc = get_encoding()
    metrics = {"context_tokens": len(enc.encode(context, disallowed_special=())) if context else 0}

    if "lexical" in options:
        gold_tokens = nltk.word_tokenize(gold_answer.lower())
        response_tokens = nltk.word_tokenize(response.lower())
        metrics["lexical"] = {}
        metrics["lexical"]["f1"] = calculate_f1_score(gold_tokens, response_tokens)
        metrics["lexical"].update(calculate_rouge_scores(gold_answer, response))
        metrics["lexical"].update(calculate_bleu_scores(gold_tokens, response_tokens))
        metrics["lexical"]["meteor"] = calculate_meteor_score(gold_tokens, response_tokens)

    if "semantic" in options:
        metrics["semantic"] = {}
        metrics["semantic"]["similarity"] = calculate_semantic_similarity(gold_answer, response)
        _, _, f1 = bert_score(
            [gold_answer], [response], lang="en",
            rescale_with_baseline=True, verbose=False,
        )
        metrics["semantic"]["bert_f1"] = f1.item() if f1 is not None else 0.0

    return metrics
