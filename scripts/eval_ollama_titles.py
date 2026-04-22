from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titlegen.config import create_run_dir, load_config, save_resolved_config
from titlegen.llm import OllamaClient, build_title_prompt, compute_quality_metrics, postprocess_title
from titlegen.training import compute_text_metrics, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama LLMs for title generation (zero-shot / few-shot)."
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", default=None)
    parser.add_argument(
        "--prepared-splits-run-name",
        default=None,
        help="Run name under outputs/runs/<name>/splits containing train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--prepared-splits-dir",
        default=None,
        help="Direct path to a splits directory containing train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test", "both"],
        default="both",
        help="Use val for tuning and test for final reporting.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override configuration values with dot notation (e.g., llm.num_shots=4).",
    )
    return parser.parse_args()


def _load_prepared_splits(
    splits_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    splits_dir = splits_dir.expanduser().resolve()
    required = ["train.csv", "val.csv", "test.csv"]
    missing = [name for name in required if not (splits_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files {missing} under {splits_dir}. "
            "Expected train.csv, val.csv, test.csv."
        )

    train_frame = pd.read_csv(splits_dir / "train.csv")
    val_frame = pd.read_csv(splits_dir / "val.csv")
    test_frame = pd.read_csv(splits_dir / "test.csv")
    return train_frame, val_frame, test_frame, splits_dir


def _resolve_splits_dir(config, args: argparse.Namespace) -> Path:
    if args.prepared_splits_dir and args.prepared_splits_run_name:
        raise ValueError("Use only one of --prepared-splits-dir or --prepared-splits-run-name.")

    if args.prepared_splits_dir:
        return Path(args.prepared_splits_dir)

    if args.prepared_splits_run_name:
        return (
            Path(str(config.project.output_root))
            / "runs"
            / str(args.prepared_splits_run_name)
            / "splits"
        )

    raise ValueError(
        "Provide one of --prepared-splits-run-name or --prepared-splits-dir. "
        "LLM evaluation must use fixed prepared splits for reproducibility."
    )


def _safe_metric_names(config) -> List[str]:
    if "evaluation" not in config or "metrics" not in config.evaluation:
        return ["bleu", "rouge", "bertscore"]
    return [str(x) for x in config.evaluation.metrics]


def _build_few_shot_examples(
    train_frame: pd.DataFrame,
    num_shots: int,
    seed: int,
    max_abstract_chars: int,
) -> List[Tuple[str, str]]:
    if num_shots <= 0:
        return []

    sample_n = min(num_shots, len(train_frame))
    sampled = train_frame.sample(n=sample_n, random_state=seed)

    examples: List[Tuple[str, str]] = []
    for _, row in sampled.iterrows():
        abstract = str(row.get("abstract", ""))[:max_abstract_chars]
        title = str(row.get("title", ""))
        examples.append((abstract, title))
    return examples


def _generate_with_retry(
    client: OllamaClient,
    prompt: str,
    options: Dict[str, float | int],
    max_retries: int,
    retry_sleep_seconds: float,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            return client.generate(prompt=prompt, options=options)
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep_seconds)
    return ""


def _predict_split(
    frame: pd.DataFrame,
    split_name: str,
    client: OllamaClient,
    instruction: str,
    few_shot_examples: List[Tuple[str, str]],
    generation_options: Dict[str, float | int],
    max_words: int,
    max_retries: int,
    retry_sleep_seconds: float,
    progress_every: int,
) -> pd.DataFrame:
    records = []
    working = frame.reset_index(drop=True)

    for idx, row in working.iterrows():
        abstract = str(row.get("abstract", ""))
        prompt = build_title_prompt(
            abstract=abstract,
            instruction=instruction,
            few_shot_examples=few_shot_examples,
        )

        raw_title = _generate_with_retry(
            client=client,
            prompt=prompt,
            options=generation_options,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        predicted_title = postprocess_title(raw_title, max_words=max_words)

        item = row.to_dict()
        item["predicted_title"] = predicted_title
        item["prompt_used"] = prompt
        records.append(item)

        if progress_every > 0 and (idx + 1) % progress_every == 0:
            print(f"[{split_name}] generated {idx + 1}/{len(working)} titles")

    return pd.DataFrame(records)


def _compute_text_metrics_safe(
    predictions: List[str],
    references: List[str],
    metric_names: List[str],
    bertscore_lang: str,
) -> Dict[str, float]:
    try:
        metrics = compute_text_metrics(
            predictions=predictions,
            references=references,
            metric_names=metric_names,
            bertscore_lang=bertscore_lang,
        )
        if any(m.lower() == "bertscore" for m in metric_names):
            metrics["bertscore_failed"] = 0.0
        return metrics
    except Exception as exc:
        if not any(m.lower() == "bertscore" for m in metric_names):
            raise

        fallback_names = [m for m in metric_names if m.lower() != "bertscore"]
        metrics = compute_text_metrics(
            predictions=predictions,
            references=references,
            metric_names=fallback_names,
            bertscore_lang=bertscore_lang,
        )
        metrics["bertscore_f1"] = -1.0
        metrics["bertscore_precision"] = -1.0
        metrics["bertscore_recall"] = -1.0
        metrics["bertscore_failed"] = 1.0
        print(f"[WARN] BERTScore failed and was skipped: {type(exc).__name__}: {exc}")
        return metrics


def _save_outputs(
    run_dir: Path,
    split_name: str,
    pred_frame: pd.DataFrame,
    model_name: str,
    mode: str,
    seed: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep compatibility with existing artifacts.
    pred_frame.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)

    generated_dir = run_dir / "generated_titles"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generated_cols = ["id", "predicted_title"]
    available_generated_cols = [c for c in generated_cols if c in pred_frame.columns]
    if "predicted_title" not in available_generated_cols:
        available_generated_cols.append("predicted_title")
    pred_frame[available_generated_cols].to_csv(
        generated_dir / f"{split_name}_generated_titles.csv",
        index=False,
    )

    human_eval_dir = run_dir / "human_eval"
    human_eval_dir.mkdir(parents=True, exist_ok=True)

    human_df = pd.DataFrame(
        {
            "id": pred_frame["id"] if "id" in pred_frame.columns else "",
            "abstract": pred_frame["abstract"] if "abstract" in pred_frame.columns else "",
            "reference_title": pred_frame["title"] if "title" in pred_frame.columns else "",
            "predicted_title": pred_frame["predicted_title"],
            "split": split_name,
            "model": model_name,
            "mode": mode,
        }
    )
    human_df.to_csv(human_eval_dir / f"{split_name}_human_eval.csv", index=False)

    sample_n = min(100, len(human_df))
    if sample_n > 0:
        sample_df = human_df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        sample_df.to_csv(human_eval_dir / f"{split_name}_human_eval_sample_100.csv", index=False)


def main() -> None:
    args = parse_args()

    config = load_config(
        base_config=args.base_config,
        model_config=args.model_config,
        training_config=args.training_config,
        overrides=args.overrides,
    )

    run_dir = create_run_dir(config)
    save_resolved_config(config, run_dir)

    seed = int(config.project.seed)
    set_global_seed(seed)

    llm_cfg = config.get("llm", {})
    generation_cfg = llm_cfg.get("generation", {})
    request_cfg = llm_cfg.get("request", {})
    quality_cfg = llm_cfg.get("quality", {})
    semantic_cfg = quality_cfg.get("semantic_similarity", {})

    model_name = str(llm_cfg.get("model", config.model.hf_name))
    host = str(llm_cfg.get("host", "http://127.0.0.1:11434"))
    timeout_seconds = int(llm_cfg.get("timeout_seconds", 180))
    num_shots = int(llm_cfg.get("num_shots", 0))
    few_shot_seed = int(llm_cfg.get("few_shot_seed", seed))
    few_shot_max_abstract_chars = int(llm_cfg.get("few_shot_max_abstract_chars", 1200))
    max_eval_samples = int(llm_cfg.get("max_eval_samples", 0))
    progress_every = int(llm_cfg.get("progress_every", 50))
    max_retries = int(request_cfg.get("max_retries", 2))
    retry_sleep_seconds = float(request_cfg.get("retry_sleep_seconds", 2.0))

    generation_options: Dict[str, float | int] = {
        "temperature": float(generation_cfg.get("temperature", 0.2)),
        "top_p": float(generation_cfg.get("top_p", 0.9)),
        "num_predict": int(generation_cfg.get("max_tokens", 40)),
        "repeat_penalty": float(generation_cfg.get("repeat_penalty", 1.1)),
        "num_ctx": int(generation_cfg.get("num_ctx", 4096)),
        "seed": int(generation_cfg.get("seed", seed)),
    }

    instruction = str(config.model.prompt)
    max_title_words = int(quality_cfg.get("max_words", 15))
    mode = "few_shot" if num_shots > 0 else "zero_shot"

    splits_dir = _resolve_splits_dir(config, args)
    train_frame, val_frame, test_frame, resolved_splits_dir = _load_prepared_splits(splits_dir)
    print(f"Using prepared splits from: {resolved_splits_dir}")

    if max_eval_samples > 0:
        val_frame = val_frame.sample(n=min(max_eval_samples, len(val_frame)), random_state=seed)
        test_frame = test_frame.sample(n=min(max_eval_samples, len(test_frame)), random_state=seed)

    few_shot_examples = _build_few_shot_examples(
        train_frame=train_frame,
        num_shots=num_shots,
        seed=few_shot_seed,
        max_abstract_chars=few_shot_max_abstract_chars,
    )
    if few_shot_examples:
        pd.DataFrame(
            [{"abstract": a, "title": t} for a, t in few_shot_examples]
        ).to_csv(run_dir / "few_shot_examples.csv", index=False)

    client = OllamaClient(host=host, model=model_name, timeout_seconds=timeout_seconds)

    metric_names = _safe_metric_names(config)
    bertscore_lang = str(config.evaluation.get("bertscore_lang", "en"))

    split_map = {"val": val_frame, "test": test_frame}
    if args.eval_split == "both":
        selected_splits = ["val", "test"]
    else:
        selected_splits = [args.eval_split]

    all_metrics: Dict[str, Dict[str, float] | Dict[str, int] | Dict[str, str | int]] = {
        "model_info": {
            "provider": "ollama",
            "model": model_name,
            "mode": mode,
            "num_shots": num_shots,
        },
        "split_source": {"prepared_splits_dir": str(resolved_splits_dir)},
        "dataset_sizes": {
            "train": int(len(train_frame)),
            "val": int(len(val_frame)),
            "test": int(len(test_frame)),
        },
    }

    for split_name in selected_splits:
        source_frame = split_map[split_name]
        print(f"Running {mode} generation for split={split_name}, rows={len(source_frame)}")

        pred_frame = _predict_split(
            frame=source_frame,
            split_name=split_name,
            client=client,
            instruction=instruction,
            few_shot_examples=few_shot_examples,
            generation_options=generation_options,
            max_words=max_title_words,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            progress_every=progress_every,
        )

        _save_outputs(
            run_dir=run_dir,
            split_name=split_name,
            pred_frame=pred_frame,
            model_name=model_name,
            mode=mode,
            seed=seed,
        )

        text_metrics = _compute_text_metrics_safe(
            predictions=pred_frame["predicted_title"].astype(str).tolist(),
            references=pred_frame["title"].astype(str).tolist(),
            metric_names=metric_names,
            bertscore_lang=bertscore_lang,
        )

        quality_metrics = compute_quality_metrics(
            predictions=pred_frame["predicted_title"].astype(str).tolist(),
            references=pred_frame["title"].astype(str).tolist(),
            abstracts=pred_frame["abstract"].astype(str).tolist(),
            min_words=int(quality_cfg.get("min_words", 4)),
            max_words=int(quality_cfg.get("max_words", 15)),
            semantic_enabled=bool(semantic_cfg.get("enabled", True)),
            semantic_model_name=str(
                semantic_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            semantic_batch_size=int(semantic_cfg.get("batch_size", 64)),
            semantic_device=str(semantic_cfg.get("device", "cpu")),
        )

        all_metrics[f"{split_name}_text_metrics"] = text_metrics
        all_metrics[f"{split_name}_quality_metrics"] = quality_metrics

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("Run completed")
    print(f"Run dir: {run_dir}")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    for split_name in selected_splits:
        print(f"{split_name.upper()} text metrics:")
        for key, value in all_metrics[f"{split_name}_text_metrics"].items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()