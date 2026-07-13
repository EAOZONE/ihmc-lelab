import importlib.util
from pathlib import Path

import numpy as np


def test_stats_compat_repairs_json_nan_std_from_range() -> None:
    module_path = Path(__file__).parents[1] / "docker" / "alex_lerobot_compat.py"
    spec = importlib.util.spec_from_file_location("alex_lerobot_compat_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from lerobot.datasets.io_utils import cast_stats_to_numpy

    stats = cast_stats_to_numpy(
        {
            "observation.state": {
                "min": [1.0, 2.0],
                "max": [1.0, 4.0],
                "mean": [1.0, 3.0],
                "std": ["NaN", 0.25],
                "count": [10],
            }
        }
    )
    std = stats["observation.state"]["std"]
    assert std.dtype.kind == "f"
    assert np.allclose(std, [0.0, 0.25])
