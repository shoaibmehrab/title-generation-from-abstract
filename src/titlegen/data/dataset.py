from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


CANONICAL_COLUMNS = ["id", "title", "abstract", "publication_year", "topics", "primary_topic"]
SUPPORTED_DATA_SCHEMAS = {
    "auto",
    "v2_titlegen",
    "v1_openalex_legacy",
    "qtl_reference",
}


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _detect_schema(raw: pd.DataFrame, config: DictConfig) -> str | None:
    columns = set(raw.columns)

    title_column = str(config.data.get("title_column", "title"))
    abstract_column = str(config.data.get("abstract_column", "abstract"))

    if title_column in columns and abstract_column in columns:
        return "v2_titlegen"
    if {"openalex_id", "title", "abstract"}.issubset(columns):
        return "v1_openalex_legacy"
    if {"Title", "Abstract"}.issubset(columns):
        return "qtl_reference"
    return None


def _build_canonical_frame(raw: pd.DataFrame, config: DictConfig, schema: str) -> pd.DataFrame:
    if schema == "v2_titlegen":
        id_column = str(config.data.get("id_column", "id"))
        title_column = str(config.data.get("title_column", "title"))
        abstract_column = str(config.data.get("abstract_column", "abstract"))

        required = [title_column, abstract_column]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns for v2_titlegen: {missing}")

        frame = pd.DataFrame(
            {
                "id": raw[id_column] if id_column in raw.columns else pd.Series([""] * len(raw)),
                "title": raw[title_column],
                "abstract": raw[abstract_column],
            }
        )
        for optional in ["publication_year", "topics", "primary_topic"]:
            if optional in raw.columns:
                frame[optional] = raw[optional]
        return frame

    if schema == "v1_openalex_legacy":
        required = ["openalex_id", "title", "abstract"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns for v1_openalex_legacy: {missing}")

        frame = pd.DataFrame(
            {
                "id": raw["openalex_id"],
                "title": raw["title"],
                "abstract": raw["abstract"],
            }
        )
        if "publication_year" in raw.columns:
            frame["publication_year"] = raw["publication_year"]
        if "topic_ids" in raw.columns:
            frame["topics"] = raw["topic_ids"]
        if "primary_topic" in raw.columns:
            frame["primary_topic"] = raw["primary_topic"]
        return frame

    if schema == "qtl_reference":
        required = ["Title", "Abstract"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns for qtl_reference: {missing}")

        frame = pd.DataFrame(
            {
                "id": pd.Series([""] * len(raw)),
                "title": raw["Title"],
                "abstract": raw["Abstract"],
            }
        )
        if "Category" in raw.columns:
            frame["primary_topic"] = raw["Category"]
        return frame

    raise ValueError(f"Unsupported schema handler: {schema}")


def load_and_clean_dataframe(config: DictConfig) -> pd.DataFrame:
    csv_path = Path(str(config.data.csv_path))
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    raw = pd.read_csv(csv_path)

    requested_schema = str(config.data.get("schema_version", "auto"))
    if requested_schema not in SUPPORTED_DATA_SCHEMAS:
        supported = ", ".join(sorted(SUPPORTED_DATA_SCHEMAS))
        raise ValueError(
            f"Unsupported data.schema_version={requested_schema!r}. "
            f"Supported values: {supported}."
        )

    detected_schema = _detect_schema(raw, config)
    if requested_schema == "auto":
        if detected_schema is None:
            raise ValueError(
                "Could not auto-detect dataset schema from columns. "
                "Set data.schema_version explicitly in config."
            )
        schema = detected_schema
    else:
        if detected_schema is not None and detected_schema != requested_schema:
            raise ValueError(
                f"Configured schema {requested_schema!r} does not match detected schema "
                f"{detected_schema!r}."
            )
        schema = requested_schema

    frame = _build_canonical_frame(raw, config, schema)

    frame["title"] = frame["title"].map(_normalize_text)
    frame["abstract"] = frame["abstract"].map(_normalize_text)

    frame = frame[(frame["title"].str.len() >= int(config.data.min_title_chars))]
    frame = frame[(frame["abstract"].str.len() >= int(config.data.min_abstract_chars))]

    if bool(config.data.deduplicate):
        if "id" in frame.columns and frame["id"].astype(str).str.strip().ne("").all():
            frame = frame.drop_duplicates(subset=["id"], keep="first")
        frame = frame.drop_duplicates(subset=["title", "abstract"], keep="first")

    frame = frame.reset_index(drop=True)
    return frame


def _safe_stratify(frame: pd.DataFrame, stratify_column: str) -> pd.Series | None:
    if not stratify_column or stratify_column not in frame.columns:
        return None

    series = frame[stratify_column].fillna("__NA__").astype(str)
    counts = series.value_counts(dropna=False)

    if counts.empty or (counts < 2).any() or counts.size < 2:
        return None
    return series


def split_dataframe(
    frame: pd.DataFrame,
    config: DictConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = config.data.split
    random_state = int(split_cfg.random_state)

    train_size = float(split_cfg.train_size)
    val_size = float(split_cfg.val_size)
    test_size = float(split_cfg.test_size)

    stratify_column = str(config.data.get("stratify_column", ""))
    stratify_all = _safe_stratify(frame, stratify_column)

    train_frame, temp_frame = train_test_split(
        frame,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_all,
    )

    val_ratio_inside_temp = val_size / (val_size + test_size)
    stratify_temp = _safe_stratify(temp_frame, stratify_column)

    val_frame, test_frame = train_test_split(
        temp_frame,
        train_size=val_ratio_inside_temp,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_temp,
    )

    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def save_split_artifacts(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    run_dir: Path,
) -> None:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "train": train_frame,
        "val": val_frame,
        "test": test_frame,
    }

    for split_name, split_frame in artifacts.items():
        split_frame.to_csv(split_dir / f"{split_name}.csv", index=False)
        if "id" in split_frame.columns:
            split_frame[["id"]].to_csv(split_dir / f"{split_name}_ids.csv", index=False)


def build_hf_datasets(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> Tuple[Dataset, Dataset, Dataset]:
    keep_columns = [
        c
        for c in ["id", "title", "abstract", "publication_year", "topics", "primary_topic"]
        if c in train_frame.columns
    ]

    train_dataset = Dataset.from_pandas(train_frame[keep_columns], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_frame[keep_columns], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_frame[keep_columns], preserve_index=False)

    return train_dataset, val_dataset, test_dataset
