from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titlegen.config import load_config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.base_config = self.project_root / "configs" / "base.yaml"

    def test_base_config_paths_are_absolute_linux_paths(self) -> None:
        config = load_config(base_config=self.base_config)

        self.assertEqual(
            str(config.data.csv_path),
            "/home/shoaib/nlp-sp26/project/Data/final/scientific-article-nlp.csv",
        )
        self.assertEqual(
            str(config.project.output_root),
            "/home/shoaib/nlp-sp26/project/outputs",
        )

    def test_invalid_split_sum_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_config(
                base_config=self.base_config,
                overrides=[
                    "data.split.train_size=0.7",
                    "data.split.val_size=0.2",
                    "data.split.test_size=0.2",
                ],
            )


if __name__ == "__main__":
    unittest.main()
