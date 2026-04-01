"""Data cleaning utilities for shop information."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "mall_name",
    "shop_name",
    "category",
    "floor",
    "description",
    "open_time",
    "close_time",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "mall_name",
    "shop_name",
    "category",
    "floor",
    "description",
)

TIME_COLUMNS: tuple[str, ...] = ("open_time", "close_time")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_time(value: object) -> str:
    """Normalize a time-like value into HH:MM 24-hour format.

    Supported examples: 10:00 AM, 10am, 08:00, 0900, 9.30, 10.00, 8:00 PM.
    Empty or invalid values return an empty string.
    """
    if value is None or pd.isna(value):
        return ""

    text = _normalize_whitespace(str(value))
    if not text:
        return ""

    lowered = text.lower().replace(".", ":")
    lowered = re.sub(r"\s+", "", lowered)

    meridiem_match = re.fullmatch(r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<meridiem>am|pm)", lowered)
    if meridiem_match:
        hour = int(meridiem_match.group("hour"))
        minute = int(meridiem_match.group("minute") or "00")
        meridiem = meridiem_match.group("meridiem")
        if hour == 12:
            hour = 0
        if meridiem == "pm":
            hour += 12
        return f"{hour:02d}:{minute:02d}"

    compact_match = re.fullmatch(r"(?P<hour>\d{1,2})(?P<minute>\d{2})", lowered)
    if compact_match and ":" not in lowered:
        hour = int(compact_match.group("hour"))
        minute = int(compact_match.group("minute"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    standard_match = re.fullmatch(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})", lowered)
    if standard_match:
        hour = int(standard_match.group("hour"))
        minute = int(standard_match.group("minute"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    logger.warning("invalid_time_format", value=text)
    return ""


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    """Ensure all required columns exist in the dataframe."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return df


def clean_shop_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw shop data for downstream processing."""
    ensure_columns(df)
    cleaned_df = df.copy()

    for column in TEXT_COLUMNS:
        cleaned_df[column] = (
            cleaned_df[column]
            .fillna("")
            .astype(str)
            .map(_normalize_whitespace)
        )

    cleaned_df["description"] = cleaned_df["description"].replace("", "No description available.")

    for column in TIME_COLUMNS:
        cleaned_df[column] = cleaned_df[column].map(normalize_time)

    logger.info("shop_data_cleaned", rows=len(cleaned_df))
    return cleaned_df
