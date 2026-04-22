from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titlegen.config import create_run_dir, load_config, save_resolved_config
from titlegen.data import (
    build_hf_datasets,
    load_and_clean_dataframe,
    save_split_artifacts,
    split_dataframe,
)
from titlegen.training import (
    compute_text_metrics,
    make_trainer_compute_metrics,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a seq2seq title generation model from config files."
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", default=None)
    parser.add_argument(
        "--prepared-splits-run-name",
        default=None,
        help="Existing run name under outputs/runs containing splits directory, e.g. prep_seed42.",
    )
    parser.add_argument(
        "--prepared-splits-dir",
        default=None,
        help="Direct path to a splits directory containing train.csv, val.csv, test.csv.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override configuration values with dot notation (e.g., training.num_train_epochs=2).",
    )
    return parser.parse_args()


def _apply_generation_config(model, config) -> None:
    generation = config.model.generation
    generation_config = model.generation_config

    generation_config.num_beams = int(generation.num_beams)
    generation_config.no_repeat_ngram_size = int(generation.no_repeat_ngram_size)
    generation_config.max_new_tokens = int(generation.max_new_tokens)

    min_new_tokens = int(generation.min_new_tokens)
    if min_new_tokens > 0:
        generation_config.min_new_tokens = min_new_tokens

    # Avoid inherited min_length conflicts from pretrained summarization configs.
    if hasattr(generation_config, "min_length"):
        generation_config.min_length = 0
    if hasattr(generation_config, "max_length"):
        generation_config.max_length = int(config.model.target_max_length) + 2


def _build_preprocess_fn(tokenizer, config):
    prompt = str(config.model.prompt)
    source_max_length = int(config.model.source_max_length)
    target_max_length = int(config.model.target_max_length)

    def preprocess(batch):
        prompts = [f"{prompt}\n\n{abstract}" for abstract in batch["abstract"]]
        model_inputs = tokenizer(
            prompts,
            max_length=source_max_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["title"],
            max_length=target_max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def _load_prepared_splits(splits_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    splits_dir = splits_dir.expanduser().resolve()

    paths = {
        "train": splits_dir / "train.csv",
        "val": splits_dir / "val.csv",
        "test": splits_dir / "test.csv",
    }

    missing_files = [name for name, path in paths.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing prepared split files in {splits_dir}: {missing_files}"
        )

    train_frame = pd.read_csv(paths["train"])
    val_frame = pd.read_csv(paths["val"])
    test_frame = pd.read_csv(paths["test"])

    required_columns = {"title", "abstract"}
    for split_name, frame in {
        "train": train_frame,
        "val": val_frame,
        "test": test_frame,
    }.items():
        missing_cols = sorted(required_columns - set(frame.columns))
        if missing_cols:
            raise ValueError(
                f"Prepared {split_name}.csv missing required columns: {missing_cols}"
            )

    return train_frame, val_frame, test_frame, str(splits_dir)


@torch.no_grad()
def generate_titles(model, tokenizer, abstracts: List[str], config) -> List[str]:
    model.eval()

    device = next(model.parameters()).device
    batch_size = int(config.training.per_device_eval_batch_size)

    prompt = str(config.model.prompt)
    source_max_length = int(config.model.source_max_length)
    generation = config.model.generation

    outputs: List[str] = []
    for start in range(0, len(abstracts), batch_size):
        chunk = abstracts[start : start + batch_size]
        prompts = [f"{prompt}\n\n{text}" for text in chunk]

        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=source_max_length,
            return_tensors="pt",
        ).to(device)

        generated = model.generate(
            **inputs,
            num_beams=int(generation.num_beams),
            max_new_tokens=int(generation.max_new_tokens),
            min_new_tokens=max(0, int(generation.min_new_tokens)),
            no_repeat_ngram_size=int(generation.no_repeat_ngram_size),
        )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend([" ".join(text.split()).strip() for text in decoded])

    return outputs


def _save_predictions(
    frame: pd.DataFrame,
    predictions: List[str],
    path: Path,
) -> None:
    output = frame.copy()
    output["predicted_title"] = predictions
    output.to_csv(path, index=False)


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

    set_global_seed(int(config.project.seed))

    if args.prepared_splits_dir and args.prepared_splits_run_name:
        raise ValueError(
            "Use only one of --prepared-splits-dir or --prepared-splits-run-name."
        )

    prepared_splits_dir: Path | None = None
    if args.prepared_splits_dir:
        prepared_splits_dir = Path(args.prepared_splits_dir)
    elif args.prepared_splits_run_name:
        prepared_splits_dir = (
            Path(str(config.project.output_root))
            / "runs"
            / str(args.prepared_splits_run_name)
            / "splits"
        )

    if prepared_splits_dir is not None:
        train_frame, val_frame, test_frame, resolved_splits_dir = _load_prepared_splits(
            prepared_splits_dir
        )
        full_size = len(train_frame) + len(val_frame) + len(test_frame)
        print(f"Using prepared splits from: {resolved_splits_dir}")
    else:
        full_frame = load_and_clean_dataframe(config)
        train_frame, val_frame, test_frame = split_dataframe(full_frame, config)
        full_size = len(full_frame)

    # Save a copy of the splits under this training run for traceability.
    # save_split_artifacts(train_frame, val_frame, test_frame, run_dir)

    train_dataset, val_dataset, _ = build_hf_datasets(train_frame, val_frame, test_frame)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.hf_name),
        use_fast=True,
        cache_dir=str(config.model.cache_dir),
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(config.model.hf_name),
        cache_dir=str(config.model.cache_dir),
    )

    _apply_generation_config(model, config)

    preprocess = _build_preprocess_fn(tokenizer, config)
    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train split",
    )
    tokenized_val = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val split",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_cuda = torch.cuda.is_available()

    training_kwargs = {
        "output_dir": str(run_dir / "trainer"),
        "num_train_epochs": float(config.training.num_train_epochs),
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "per_device_train_batch_size": int(config.training.per_device_train_batch_size),
        "per_device_eval_batch_size": int(config.training.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "warmup_ratio": float(config.training.warmup_ratio),
        "logging_steps": int(config.training.logging_steps),
        "save_strategy": str(config.training.save_strategy),
        "save_total_limit": int(config.training.save_total_limit),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(config.training.metric_for_best_model),
        "greater_is_better": bool(config.training.greater_is_better),
        "predict_with_generate": True,
        "generation_max_length": int(config.model.target_max_length),
        "generation_num_beams": int(config.model.generation.num_beams),
        "fp16": bool(config.training.fp16 and use_cuda),
        "bf16": bool(config.training.bf16 and use_cuda),
        "dataloader_num_workers": int(config.training.dataloader_num_workers),
        "report_to": "none",
        "seed": int(config.project.seed),
    }

    training_arg_params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    eval_value = str(config.training.evaluation_strategy)

    if "evaluation_strategy" in training_arg_params:
        training_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in training_arg_params:
        training_kwargs["eval_strategy"] = eval_value
    else:
        raise TypeError(
            "This transformers version supports neither evaluation_strategy nor eval_strategy."
        )

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator,
        "compute_metrics": make_trainer_compute_metrics(tokenizer),
    }

    trainer_init_params = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer_metrics = trainer.evaluate()

    best_model_dir = run_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    model = trainer.model

    val_predictions = generate_titles(model, tokenizer, val_frame["abstract"].tolist(), config)
    test_predictions = generate_titles(model, tokenizer, test_frame["abstract"].tolist(), config)

    metric_names = list(config.evaluation.metrics)
    bertscore_lang = str(config.evaluation.bertscore_lang)

    val_metrics = compute_text_metrics(
        predictions=val_predictions,
        references=val_frame["title"].tolist(),
        metric_names=metric_names,
        bertscore_lang=bertscore_lang,
    )
    test_metrics = compute_text_metrics(
        predictions=test_predictions,
        references=test_frame["title"].tolist(),
        metric_names=metric_names,
        bertscore_lang=bertscore_lang,
    )

    _save_predictions(val_frame, val_predictions, run_dir / "predictions_val.csv")
    _save_predictions(test_frame, test_predictions, run_dir / "predictions_test.csv")

    all_metrics: Dict[str, Dict[str, float]] = {
        "trainer_train": dict(train_result.metrics),
        "trainer_eval": dict(trainer_metrics),
        "val_text_metrics": val_metrics,
        "test_text_metrics": test_metrics,
        "dataset_sizes": {
            "full": full_size,
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame),
        },
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    print("Run completed")
    print(f"Run dir: {run_dir}")
    print(f"Model: {config.model.hf_name}")
    print("Validation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()