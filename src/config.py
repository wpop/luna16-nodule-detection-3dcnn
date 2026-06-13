"""
Project configuration for the LUNA16 3D CNN nodule detection project.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    """Stores all important project paths."""

    root_dir: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = root_dir / "data"
    raw_data_dir: Path = data_dir / "raw" / "LUNA16"
    interim_data_dir: Path = data_dir / "interim"
    processed_data_dir: Path = data_dir / "processed"
    candidates_dir: Path = data_dir / "candidates"
    patches_dir: Path = data_dir / "patches"

    outputs_dir: Path = root_dir / "outputs"
    checkpoints_dir: Path = outputs_dir / "checkpoints"
    results_json_dir: Path = outputs_dir / "results_json"
    metrics_dir: Path = outputs_dir / "metrics"
    figures_dir: Path = outputs_dir / "figures"


@dataclass(frozen=True)
class DataConfig:
    """Stores medical image preprocessing parameters."""

    hu_min: int = -1000
    hu_max: int = 400
    patch_size: int = 64
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class TrainingConfig:
    """Stores default training parameters."""

    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-4
    num_workers: int = 4


@dataclass(frozen=True)
class ProjectConfig:
    """Groups all configuration sections in one object."""

    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()


config = ProjectConfig()

