from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titlegen.data.dataset import load_and_clean_dataframe, split_dataframe


class DatasetTests(unittest.TestCase):
    def _base_data_config(self, csv_path: Path, schema_version: str = "auto"):
        return OmegaConf.create(
            {
                "data": {
                    "csv_path": str(csv_path),
                    "schema_version": schema_version,
                    "id_column": "id",
                    "title_column": "title",
                    "abstract_column": "abstract",
                    "stratify_column": "primary_topic",
                    "min_abstract_chars": 10,
                    "min_title_chars": 3,
                    "deduplicate": True,
                    "split": {
                        "train_size": 0.8,
                        "val_size": 0.1,
                        "test_size": 0.1,
                        "random_state": 42,
                    },
                }
            }
        )

    def test_v2_schema_and_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data.csv"
            frame = pd.DataFrame(
                {
                    "id": ["A1", "A1", "A2"],
                    "title": ["T1", "T1", "T2"],
                    "abstract": [
                        "Abstract one has enough words.",
                        "Abstract one has enough words.",
                        "Abstract two has enough words.",
                    ],
                    "primary_topic": ["NLP", "NLP", "ML"],
                }
            )
            frame.to_csv(csv_path, index=False)

            config = self._base_data_config(csv_path, schema_version="v2_titlegen")
            cleaned = load_and_clean_dataframe(config)

            self.assertEqual(len(cleaned), 2)
            self.assertIn("id", cleaned.columns)
            self.assertIn("title", cleaned.columns)
            self.assertIn("abstract", cleaned.columns)

    def test_legacy_schema_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "legacy.csv"
            frame = pd.DataFrame(
                {
                    "openalex_id": ["W1", "W2"],
                    "title": ["Legacy Title 1", "Legacy Title 2"],
                    "abstract": [
                        "Legacy abstract one contains enough content.",
                        "Legacy abstract two contains enough content.",
                    ],
                    "publication_year": [2024, 2025],
                    "topic_ids": ["[T1]", "[T2]"],
                    "primary_topic": ["Topic A", "Topic B"],
                }
            )
            frame.to_csv(csv_path, index=False)

            config = self._base_data_config(csv_path, schema_version="v1_openalex_legacy")
            cleaned = load_and_clean_dataframe(config)

            self.assertEqual(list(cleaned["id"]), ["W1", "W2"])
            self.assertIn("topics", cleaned.columns)
            self.assertIn("primary_topic", cleaned.columns)

    def test_split_dataframe_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "split.csv"
            rows = []
            for i in range(100):
                rows.append(
                    {
                        "id": f"ID{i}",
                        "title": f"Title {i}",
                        "abstract": f"Abstract content for row {i} with enough length.",
                        "primary_topic": "A" if i % 2 == 0 else "B",
                    }
                )
            frame = pd.DataFrame(rows)
            frame.to_csv(csv_path, index=False)

            config = self._base_data_config(csv_path, schema_version="v2_titlegen")
            cleaned = load_and_clean_dataframe(config)
            train, val, test = split_dataframe(cleaned, config)

            self.assertEqual(len(train), 80)
            self.assertEqual(len(val), 10)
            self.assertEqual(len(test), 10)


if __name__ == "__main__":
    unittest.main()
