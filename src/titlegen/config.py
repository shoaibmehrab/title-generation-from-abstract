from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import DictConfig, OmegaConf


SUPPORTED_DATA_SCHEMAS = {
    "auto",
    "v2_titlegen",
    "v1_openalex_legacy",
    "qtl_reference",
}


def load_config(
    base_config: Path | str,
    model_config: Optional[Path | str] = None,
    training_config: Optional[Path | str] = None,
    overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    layers = [OmegaConf.load(str(base_config))]

    if model_config:
        layers.append(OmegaConf.load(str(model_config)))
    if training_config:
        layers.append(OmegaConf.load(str(training_config)))

    config = OmegaConf.merge(*layers)

    if overrides:
        dotlist = OmegaConf.from_dotlist(list(overrides))
        config = OmegaConf.merge(config, dotlist)

    _validate_config(config)
    return config


def _validate_config(config: DictConfig) -> None:
    split = config.data.split
    total = float(split.train_size) + float(split.val_size) + float(split.test_size)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"data.split values must sum to 1.0 (got {total:.6f}).")

    schema_version = str(config.data.get("schema_version", "auto"))
    if schema_version not in SUPPORTED_DATA_SCHEMAS:
        supported = ", ".join(sorted(SUPPORTED_DATA_SCHEMAS))
        raise ValueError(
            f"Unsupported data.schema_version={schema_version!r}. "
            f"Supported values: {supported}."
        )


def create_run_dir(config: DictConfig) -> Path:
    output_root = Path(str(config.project.output_root))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_name = config.project.run_name
    if run_name is None or str(run_name).strip() == "":
        run_name = f"{stamp}_{config.model.alias}"

    run_dir = output_root / "runs" / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    config.project.run_name = str(run_name)
    config.project.run_dir = str(run_dir)
    return run_dir


def save_resolved_config(config: DictConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, str(run_dir / "resolved_config.yaml"))
