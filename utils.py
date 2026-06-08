

import csv
from pathlib import Path
import random
from typing import Optional


def _load_csv_rows(
    csv_path: str,
    limit: Optional[int] = None,
    random_sample: bool = False,
    sample_seed: Optional[int] = None,
) -> list:
    """
    Loads rows from a CSV file.
    Args:
        csv_path: The path to the CSV file.
        limit: The maximum number of rows to load. If None, loads all rows.
        random_sample: If True, randomly samples rows if limit is set and less than total rows.
        sample_seed: The seed for random sampling. Ignored if random_sample is False.
    Returns:
        A list of dictionaries representing the rows in the CSV file.
    """
    base_dir = Path(__file__).resolve().parent
    path = Path(csv_path)
    if not path.is_absolute():
        path = base_dir / path

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if limit is not None:
        limit = max(0, limit)
        if limit < len(rows):
            if random_sample:
                rng = random.Random(sample_seed)
                rows = rng.sample(rows, limit)
            else:
                rows = rows[:limit]

    return rows

def _parse_csv_number(raw_value: Optional[str]) -> Optional[float]:
    """
    Parses a CSV number from a string value.
    """
    if raw_value is None:
        return None

    value = raw_value.strip().replace('"', "")
    if not value:
        return None

    upper = value.upper()
    if upper in {"N/A", "NA", "NONE", "NULL"}:
        return None

    value = value.replace(" ", "")

    # The dataset uses locale formatting (e.g. 12.345.678 and "2,17E+10").
    if "," in value:
        if "E" in upper:
            value = value.replace(".", "").replace(",", ".")
        elif "." in value and value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        elif "." not in value:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(".") > 1:
        value = value.replace(".", "")

    try:
        return float(value)
    except ValueError:
        return None

def _rescale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    ratio = max(0.0, min(1.0, ratio))
    return dst_min + ratio * (dst_max - dst_min)