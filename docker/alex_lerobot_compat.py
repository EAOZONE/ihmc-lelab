"""Small runtime compatibility fixes for the pinned Alex LeRobot image."""

from __future__ import annotations

import logging

import numpy as np

from lerobot.datasets import io_utils

ALEX_STATS_SANITIZER = True

_original_cast_stats_to_numpy = io_utils.cast_stats_to_numpy


def _cast_stats_to_numpy_with_nonfinite_json_support(stats: dict) -> dict:
    converted = _original_cast_stats_to_numpy(stats)
    repaired: list[str] = []

    for feature_name, feature_stats in converted.items():
        if not isinstance(feature_stats, dict):
            continue

        # JSON has no portable NaN value. LeRobot's serializer writes "NaN",
        # which otherwise creates a numpy string array and later crashes
        # torch.as_tensor before a policy can decide which stats it needs.
        for stat_name, value in tuple(feature_stats.items()):
            array = np.asarray(value)
            if array.dtype.kind in {"O", "S", "U"}:
                try:
                    feature_stats[stat_name] = array.astype(np.float32)
                except (TypeError, ValueError):
                    continue

        std = feature_stats.get("std")
        minimum = feature_stats.get("min")
        maximum = feature_stats.get("max")
        if std is None or minimum is None or maximum is None:
            continue

        std_array = np.asarray(std, dtype=np.float32)
        invalid = ~np.isfinite(std_array)
        if not invalid.any():
            continue

        fallback = np.abs(
            np.asarray(maximum, dtype=np.float32) - np.asarray(minimum, dtype=np.float32)
        ) * 0.5
        if fallback.shape != std_array.shape:
            continue
        feature_stats["std"] = np.where(invalid, fallback, std_array)
        repaired.append(f"{feature_name}.std ({int(invalid.sum())} values)")

    if repaired:
        logging.warning(
            "Alex training image repaired non-finite dataset statistics in memory: %s. "
            "Recompute the dataset stats on Hugging Face for a permanent fix.",
            ", ".join(repaired),
        )
    return converted


io_utils.cast_stats_to_numpy = _cast_stats_to_numpy_with_nonfinite_json_support
