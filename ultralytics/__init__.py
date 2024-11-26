# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.32"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.engine.model import Model as YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "YOLO",
    "checks",
    "download",
    "settings",
)
