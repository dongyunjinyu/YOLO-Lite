import os

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from yololite.engine.model import Model as YOLO

__all__ = "YOLO"

