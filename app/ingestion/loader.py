"""CSV data loader for shop information."""

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def load_csv(file_path: str) -> pd.DataFrame:
    """Load shop data from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with raw shop data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    logger.info("loading_csv", file_path=file_path)
    df = pd.read_csv(file_path, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    logger.info("csv_loaded", rows=len(df), columns=list(df.columns))
    return df
