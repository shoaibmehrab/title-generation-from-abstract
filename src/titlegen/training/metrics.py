from __future__ import annotations

from typing import Dict, Iterable, List

import evaluate
import numpy as np


def _normalize_texts(texts: Iterable[str]) -> List[str]:
    normalized = []
    for text in texts:
        normalized.append(" ".join(str(text).split()).strip())
    return normalized


def make_trainer_compute_metrics(tokenizer):
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = _normalize_texts(decoded_predictions)
        decoded_labels = _normalize_texts(decoded_labels)

        rouge_scores = rouge_metric.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )
        bleu_score = bleu_metric.compute(
            predictions=decoded_predictions,
            references=[[label] for label in decoded_labels],
        )

        return {
            "bleu": float(bleu_score["score"]),
            "rouge1": float(rouge_scores["rouge1"]),
            "rouge2": float(rouge_scores["rouge2"]),
            "rougeL": float(rouge_scores["rougeL"]),
        }

    return compute_metrics


def compute_text_metrics(
    predictions: List[str],
    references: List[str],
    metric_names: Iterable[str],
    bertscore_lang: str = "en",
) -> Dict[str, float]:
    metric_names = {name.lower() for name in metric_names}

    predictions = _normalize_texts(predictions)
    references = _normalize_texts(references)

    output: Dict[str, float] = {}

    if "rouge" in metric_names:
        rouge_metric = evaluate.load("rouge")
        rouge_scores = rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        output.update(
            {
                "rouge1": float(rouge_scores["rouge1"]),
                "rouge2": float(rouge_scores["rouge2"]),
                "rougeL": float(rouge_scores["rougeL"]),
            }
        )

    if "bleu" in metric_names:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_score = bleu_metric.compute(
            predictions=predictions,
            references=[[reference] for reference in references],
        )
        output["bleu"] = float(bleu_score["score"])

    if "bertscore" in metric_names:
        try:
            bertscore_metric = evaluate.load("bertscore")
            try:
                # Prefer fast tokenizer path when supported by this evaluate version.
                bertscore = bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang=bertscore_lang,
                    use_fast_tokenizer=True,
                )
            except TypeError:
                # Backward compatibility for evaluate versions without this kwarg.
                bertscore = bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang=bertscore_lang,
                )

            output["bertscore_f1"] = float(np.mean(bertscore["f1"]))
            output["bertscore_precision"] = float(np.mean(bertscore["precision"]))
            output["bertscore_recall"] = float(np.mean(bertscore["recall"]))
        except Exception as exc:
            print(f"[WARN] BERTScore failed and was skipped: {type(exc).__name__}: {exc}")
            output["bertscore_f1"] = -1.0
            output["bertscore_precision"] = -1.0
            output["bertscore_recall"] = -1.0
            output["bertscore_failed"] = 1.0

    return output
