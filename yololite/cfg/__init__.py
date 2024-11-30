# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from yololite.utils import (
    DEFAULT_CFG_DICT,
    LOGGER,
    ROOT,
    RUNS_DIR,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    colorstr,
    deprecation_warn,
    yaml_load,
)

# Define valid tasks and modes
MODES = {"train", "val", "predict"}
TASKS = {"detect"}
TASK2MODEL = {"detect": "yolo11n.pt"}
TASK2METRIC = {"detect": "metrics/mAP50-95(B)"}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []

# Define keys for arg type checks
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # integer-only arguments
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                )
                cfg[k] = bool(v)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def get_save_dir(args, name=None):
    """
    Returns the directory path for saving outputs, derived from arguments or default settings.
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from yololite.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok)

    return Path(save_dir)


def _handle_deprecation(custom):
    """
    Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings.
    """
    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    Checks alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (Dict): The base configuration dictionary containing valid keys.
        custom (Dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e
