import json
from pathlib import Path

from src.engine.benchmark import BenchmarkResult


def test_benchmark_result_save_json(tmp_path: Path):
    result = BenchmarkResult(
        model_name="baseline",
        total_parameters=100,
        trainable_parameters=90,
        train_loss=0.5,
        val_loss=0.4,
        train_accuracy=0.8,
        val_accuracy=0.9,
        learning_rate=0.001,
    )
    output_path = tmp_path / "results_json" / "baseline_benchmark.json"

    saved_path = result.save_json(output_path)

    with saved_path.open("r", encoding="utf-8") as input_file:
        saved_result = json.load(input_file)

    assert saved_path == output_path
    assert saved_result == result.to_dict()
