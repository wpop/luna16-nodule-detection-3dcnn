import re
from pathlib import Path

from src.engine.experiment import ExperimentManager


def test_experiment_manager_creates_output_directories(tmp_path: Path):
    experiment = ExperimentManager(
        project_root=tmp_path,
        model_name="baseline",
    )

    assert experiment.experiment_dir.exists()
    assert experiment.checkpoint_dir.exists()
    assert experiment.metrics_dir.exists()
    assert experiment.figures_dir.exists()
    assert experiment.results_dir.exists()
    assert experiment.experiment_dir.parent == tmp_path / "outputs" / "experiments"
    assert re.match(
        r"\d{8}_\d{6}_baseline",
        experiment.experiment_dir.name,
    )
