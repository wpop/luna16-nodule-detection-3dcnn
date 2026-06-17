"""
Summarize benchmark JSON files from experiment directories.
"""

import csv
import json
from pathlib import Path


HEADERS = [
    "experiment",
    "model_name",
    "total_parameters",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "learning_rate",
]


def load_benchmark_rows(project_root: Path) -> list[dict[str, object]]:
    """
    Load benchmark rows sorted by experiment directory name.
    """

    benchmark_paths = sorted(
        project_root.glob("outputs/experiments/*/results_json/benchmark.json"),
        key=lambda path: path.parents[1].name,
    )

    rows = []

    for benchmark_path in benchmark_paths:
        with benchmark_path.open("r", encoding="utf-8") as input_file:
            benchmark = json.load(input_file)

        rows.append(
            {
                "experiment": benchmark_path.parents[1].name,
                "model_name": benchmark["model_name"],
                "total_parameters": benchmark["total_parameters"],
                "train_loss": benchmark["train_loss"],
                "val_loss": benchmark["val_loss"],
                "train_accuracy": benchmark["train_accuracy"],
                "val_accuracy": benchmark["val_accuracy"],
                "learning_rate": benchmark["learning_rate"],
            }
        )

    return rows


def print_table(rows: list[dict[str, object]]) -> None:
    """
    Print benchmark rows as a simple table.
    """

    widths = {
        header: max(
            len(header),
            *(len(str(row[header])) for row in rows),
        )
        for header in HEADERS
    }

    print(" | ".join(header.ljust(widths[header]) for header in HEADERS))
    print("-+-".join("-" * widths[header] for header in HEADERS))

    for row in rows:
        print(
            " | ".join(
                str(row[header]).ljust(widths[header])
                for header in HEADERS
            )
        )


def save_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    """
    Save benchmark rows as a CSV file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_markdown(rows: list[dict[str, object]], output_path: Path) -> Path:
    """
    Save benchmark rows as a GitHub-style Markdown table.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write("| " + " | ".join(HEADERS) + " |\n")
        output_file.write("| " + " | ".join("---" for _ in HEADERS) + " |\n")

        for row in rows:
            output_file.write(
                "| "
                + " | ".join(str(row[header]) for header in HEADERS)
                + " |\n"
            )

    return output_path


def main() -> None:
    """
    Summarize saved experiment benchmarks.
    """

    project_root = Path(__file__).resolve().parents[1]
    rows = load_benchmark_rows(project_root)

    if not rows:
        print("No benchmark files found.")
        return

    print_table(rows)

    csv_path = save_csv(rows, project_root / "outputs" / "benchmark_summary.csv")
    markdown_path = save_markdown(
        rows,
        project_root / "outputs" / "benchmark_summary.md",
    )

    print("Saved CSV:", csv_path)
    print("Saved Markdown:", markdown_path)


if __name__ == "__main__":
    main()
