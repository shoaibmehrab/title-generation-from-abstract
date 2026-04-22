from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


def _word_count(text: str) -> int:
    return len(_tokenize(text))


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def _distinct_ngram_ratio(token_lists: List[List[str]], n: int) -> float:
    all_ngrams = []
    for tokens in token_lists:
        if len(tokens) < n:
            continue
        all_ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return float(len(set(all_ngrams)) / len(all_ngrams))


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model_device = torch.device(device)
    model.to(model_device)
    model.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(model_device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


def _semantic_cosine_mean(
    predictions: List[str],
    references: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> float:
    pred_emb = _encode_texts(predictions, model_name, batch_size, device)
    ref_emb = _encode_texts(references, model_name, batch_size, device)
    cosine = (pred_emb * ref_emb).sum(dim=1)
    return float(cosine.mean().item())


def compute_quality_metrics(
    predictions: List[str],
    references: List[str],
    abstracts: List[str],
    *,
    min_words: int = 4,
    max_words: int = 15,
    semantic_enabled: bool = True,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_batch_size: int = 64,
    semantic_device: str = "cpu",
) -> Dict[str, float]:
    preds = [str(x or "").strip() for x in predictions]
    refs = [str(x or "").strip() for x in references]
    srcs = [str(x or "").strip() for x in abstracts]

    pred_token_lists = [_tokenize(x) for x in preds]
    ref_token_lists = [_tokenize(x) for x in refs]
    src_word_counts = [max(1, _word_count(x)) for x in srcs]
    pred_word_counts = [_word_count(x) for x in preds]

    empty_rate = _safe_mean([1.0 if c == 0 else 0.0 for c in pred_word_counts])
    length_compliance_rate = _safe_mean(
        [1.0 if min_words <= c <= max_words else 0.0 for c in pred_word_counts]
    )

    unique_count = len(set(preds))
    duplicate_rate = 0.0 if not preds else float(1.0 - (unique_count / len(preds)))

    distinct_1 = _distinct_ngram_ratio(pred_token_lists, 1)
    distinct_2 = _distinct_ngram_ratio(pred_token_lists, 2)

    compression_ratio_mean = _safe_mean(
        [pred_word_counts[i] / src_word_counts[i] for i in range(len(preds))]
    )

    exact_match_rate = _safe_mean(
        [1.0 if preds[i].lower() == refs[i].lower() else 0.0 for i in range(len(preds))]
    )

    jaccard_title_reference_mean = _safe_mean(
        [_jaccard(pred_token_lists[i], ref_token_lists[i]) for i in range(len(preds))]
    )

    output: Dict[str, float] = {
        "empty_rate": float(empty_rate),
        "length_compliance_rate": float(length_compliance_rate),
        "duplicate_rate": float(duplicate_rate),
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "avg_title_words": float(_safe_mean(pred_word_counts)),
        "compression_ratio_mean": float(compression_ratio_mean),
        "exact_match_rate": float(exact_match_rate),
        "jaccard_title_reference_mean": float(jaccard_title_reference_mean),
    }

    if semantic_enabled:
        try:
            output["semantic_cosine_mean"] = _semantic_cosine_mean(
                predictions=preds,
                references=refs,
                model_name=semantic_model_name,
                batch_size=int(semantic_batch_size),
                device=str(semantic_device),
            )
            output["semantic_failed"] = 0.0
        except Exception:
            output["semantic_cosine_mean"] = -1.0
            output["semantic_failed"] = 1.0
    else:
        output["semantic_cosine_mean"] = -1.0
        output["semantic_failed"] = 1.0

    return output