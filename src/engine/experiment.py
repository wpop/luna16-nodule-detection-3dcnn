"""
Experiment directory management utilities.
"""

from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """
    Create and expose output directories for one training run.
    """

    def __init__(
        self,
        project_root: Path,
        model_name: str,
    ):
        """
        Initialize experiment directories.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = (
            project_root / "outputs" / "experiments" / f"{timestamp}_{model_name}"
        )
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.figures_dir = self.experiment_dir / "figures"
        self.results_dir = self.experiment_dir / "results_json"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
