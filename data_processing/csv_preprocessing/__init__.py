"""CSV pre-processing utilities.

This package turns a single raw simulation CSV into a set of pre-processed
"world" CSV files, one per simulation time window, with task arrival times
normalized *relatively* (the earliest task of each world is shifted to t=0
while the gaps between arrivals are preserved).

See :mod:`data_processing.csv_preprocessing.world_splitter`.
"""

from .world_splitter import (
    NORMALIZED_ARRIVAL_COLUMN,
    SYSTEM_ARRIVAL_COLUMN,
    WorldManifest,
    load_world_paths,
    split_into_worlds,
)

__all__ = [
    "NORMALIZED_ARRIVAL_COLUMN",
    "SYSTEM_ARRIVAL_COLUMN",
    "WorldManifest",
    "load_world_paths",
    "split_into_worlds",
]
