from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from titlegen.config import create_run_dir, load_config, save_resolved_config
from titlegen.data import load_and_clean_dataframe, save_split_artifacts, split_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/val/test splits for title generation."
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--training-config", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dot notation (e.g., data.split.random_state=7).",
    )
    return parser.parse_args()


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

    full_frame = load_and_clean_dataframe(config)
    train_frame, val_frame, test_frame = split_dataframe(full_frame, config)
    save_split_artifacts(train_frame, val_frame, test_frame, run_dir)

    print("Prepared data splits:")
    print(f"  total: {len(full_frame)}")
    print(f"  train: {len(train_frame)}")
    print(f"  val: {len(val_frame)}")
    print(f"  test: {len(test_frame)}")
    print(f"Artifacts: {run_dir / 'splits'}")


if __name__ == "__main__":
    main()
