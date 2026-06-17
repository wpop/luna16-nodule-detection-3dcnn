"""
Benchmark result export utilities.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Store final benchmark metrics for one model run.
    """

    model_name: str
    total_parameters: int
    trainable_parameters: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float

    def to_dict(self) -> dict[str, str | int | float]:
        """
        Return benchmark result as a dictionary.
        """

        return asdict(self)

    def save_json(self, output_path: Path) -> Path:
        """
        Save benchmark result as a JSON file.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(self.to_dict(), output_file)

        return output_path
