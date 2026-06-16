from pathlib import Path

from src.data.luna_dataset import LunaDataset


def test_dataset_creation():

    dataset = LunaDataset(
        candidates_path=Path(
            "data/raw/LUNA16/data-unversioned/part2/luna/candidates.csv"
        ),
        data_dir=Path("data/raw/LUNA16"),
    )

    assert len(dataset) > 0
