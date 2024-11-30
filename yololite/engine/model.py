# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import torch
from PIL import Image

from yololite.engine.predictor import DetectionPredictor
from yololite.engine.trainer import DetectionTrainer
from yololite.engine.validator import DetectionValidator
from yololite.nn.tasks import DetectionModel

from yololite.cfg import get_cfg, get_save_dir
from yololite.engine.results import Results
from yololite.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from yololite.utils import (
    DEFAULT_CFG_DICT,
    checks,
    yaml_load,
)


class Model(nn.Module):
    """
    Attributes:
        callbacks (Dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (Dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (Dict): A dictionary of overrides for model configuration.
        metrics (Dict): The latest training/validation metrics.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        # self.callbacks = callbacks.get_default_callbacks()
        self.callbacks = None
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.task = task  # task type
        model = str(model).strip()

        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model)

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, verbose=False) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = DetectionModel(cfg_dict, verbose=verbose)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str) -> None:
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = self.model.args["task"]
        self.overrides = self.model.args = {k: v for k, v in self.model.args.items() if k in {"imgsz", "data", "task", "single_cls"}}
        self.ckpt_path = self.model.pt_path
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> List[Results]:

        custom = {"conf": 0.25, "batch": 1, "save": True, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = DetectionPredictor(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor(source=source, stream=stream)

    def val(self, **kwargs):
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right
        validator = DetectionValidator(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def train(self, **kwargs):
        """
        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.
        """
        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or {"detect": "coco8.yaml"}[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = DetectionTrainer(overrides=args)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.train()
        # Update model and cfg after training
        ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
        self.model, _ = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

