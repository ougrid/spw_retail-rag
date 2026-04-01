import pandas as pd
import pytest

from app.ingestion.cleaner import clean_shop_data, ensure_columns, normalize_time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10:00 AM", "10:00"),
        ("10am", "10:00"),
        ("08:00", "08:00"),
        ("0900", "09:00"),
        ("9.30", "09:30"),
        ("10.00", "10:00"),
        ("8:00 PM", "20:00"),
        ("", ""),
        (None, ""),
        ("invalid", ""),
    ],
)
def test_normalize_time(value, expected):
    assert normalize_time(value) == expected


def test_clean_shop_data_standardizes_text_and_times():
    raw_df = pd.DataFrame(
        {
            "mall_name": ["  Icon Siam  "],
            "shop_name": [" Nike "],
            "category": [" Sports "],
            "floor": [" 1 "],
            "description": [None],
            "open_time": ["10am"],
            "close_time": ["10:00 PM"],
        }
    )

    cleaned_df = clean_shop_data(raw_df)

    row = cleaned_df.iloc[0]
    assert row["mall_name"] == "Icon Siam"
    assert row["shop_name"] == "Nike"
    assert row["category"] == "Sports"
    assert row["floor"] == "1"
    assert row["description"] == "No description available."
    assert row["open_time"] == "10:00"
    assert row["close_time"] == "22:00"


def test_ensure_columns_raises_when_columns_are_missing():
    raw_df = pd.DataFrame({"mall_name": ["ICONSIAM"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        ensure_columns(raw_df)
