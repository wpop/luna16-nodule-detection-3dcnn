"""
Utilities for loading LUNA16 annotation and candidate CSV files.
"""

from pathlib import Path

import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a LUNA16 CSV file.
    """

    return pd.read_csv(csv_path)



def filter_available_scans(
    dataframe: pd.DataFrame,
    available_series_uids: set[str],
) -> pd.DataFrame:
    """
    Keep only rows whose CT scan is available locally.
    """

    return dataframe[
        dataframe["seriesuid"].isin(available_series_uids)
    ].copy()

