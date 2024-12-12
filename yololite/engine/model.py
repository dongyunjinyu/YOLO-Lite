# YOLO-Lite 🚀

from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from yololite.engine.predictor import DetectionPredictor
from yololite.engine.trainer import DetectionTrainer
from yololite.engine.validator import DetectionValidator
from yololite.nn.tasks import DetectionModel
from yololite.engine.results import Results
from yololite.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from yololite.utils import DEFAULT_CFG_DICT, checks, yaml_load


class YOLOLite(nn.Module):
    """
    属性：
        callbacks (Dict): 用于各种事件的回调函数字典。
        predictor (BasePredictor): 用于进行预测的预测器对象。
        model (nn.Module): 底层 PyTorch 模型。
        trainer (BaseTrainer): 用于训练模型的训练器对象。
        ckpt (Dict): 如果从 *.pt 文件加载模型，则为检查点数据。
        cfg (str): 如果从 *.yaml 文件加载模型，则为模型的配置。
        ckpt_path (str): 检查点文件的路径。
        overrides (Dict): 模型配置的覆盖字典。
        metrics (Dict): 最新的训练/验证指标。
        task (str): 模型的任务类型。
        model_name (str): 模型的名称。
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.callbacks = None  # 回调函数初始化
        self.predictor = None  # 预测器初始化
        self.model = None  # 模型对象初始化
        self.trainer = None  # 训练器对象初始化
        self.ckpt = None  # 检查点初始化
        self.cfg = None  # 配置初始化
        self.ckpt_path = None  # 检查点路径初始化
        self.overrides = {}  # 覆盖的训练器对象参数初始化
        self.metrics = None  # 验证/训练指标初始化
        self.task = task  # 任务类型
        model = str(model).strip()  # 转换模型路径为字符串并去除空格

        # 加载或创建新的 YOLO 模型
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)  # 加载新的模型配置
        else:
            self._load(model)  # 从文件加载模型权重

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        return self.predict(source, stream, **kwargs)  # 调用预测方法

    def _new(self, cfg: str, task=None, verbose=False) -> None:
        cfg_dict = yaml_model_load(cfg)  # 加载 YAML 配置
        self.cfg = cfg  # 设置配置
        self.task = task or guess_model_task(cfg_dict)  # 任务类型
        self.model = DetectionModel(cfg_dict, verbose=verbose)  # 创建检测模型
        self.overrides["model"] = self.cfg  # 覆盖模型配置
        self.overrides["task"] = self.task  # 覆盖任务类型

        # 将默认参数与模型参数合并以允许从 YAML 导出
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # 合并默认参数和模型参数
        self.model.task = self.task  # 设置模型任务
        self.model_name = cfg  # 设置模型名称

    def _load(self, weights: str) -> None:
        self.model, self.ckpt = attempt_load_one_weight(weights)  # 尝试加载权重
        self.task = self.model.args["task"]  # 获取任务类型
        # 过滤出需要的参数
        self.overrides = self.model.args = {k: v for k, v in self.model.args.items() if k in {"imgsz", "data", "task", "single_cls"}}
        self.ckpt_path = self.model.pt_path  # 设置检查点路径
        self.overrides["model"] = weights  # 设置模型权重
        self.overrides["task"] = self.task  # 设置任务类型
        self.model_name = weights  # 设置模型名称

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> List[Results]:
        custom = {"conf": 0.25, "batch": 1, "save": True, "mode": "predict"}  # 自定义参数
        args = {**self.overrides, **custom, **kwargs}  # 合并参数
        self.predictor = DetectionPredictor(overrides=args)  # 创建预测器
        self.predictor.setup_model(model=self.model)  # 设置模型
        return self.predictor(source=source, stream=stream)  # 调用预测器进行预测

    def val(self, **kwargs):
        custom = {"rect": True}  # 自定义参数
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 合并参数
        validator = DetectionValidator(args=args)  # 创建验证器
        validator(model=self.model)  # 验证模型
        self.metrics = validator.metrics  # 获取验证指标
        return validator.metrics  # 返回指标

    def train(self, **kwargs):
        """
        参数：
            trainer (BaseTrainer | None): 自定义训练器实例。如果为 None，则使用默认训练器。
            **kwargs (Any): 训练配置的任意关键字参数。常见选项包括：
                data (str): 数据集配置文件的路径。
                epochs (int): 训练的总周期数。
                batch_size (int): 训练的批量大小。
                imgsz (int): 输入图像大小。
                device (str): 训练运行的设备（例如，'cuda', 'cpu'）。
                workers (int): 数据加载的工作线程数。
                optimizer (str): 用于训练的优化器。
                lr0 (float): 初始学习率。
                patience (int): 在没有可观察到的改进时，提前停止训练的等待周期数。
        """
        # 检查 YAML 配置并解析
        overrides = yaml_load(kwargs["cfg"]) if kwargs.get("cfg") else self.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or {"detect": "coco8.yaml"}[self.task], # todo
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # 合并参数
        if args.get("resume"):
            args["resume"] = self.ckpt_path  # 设置恢复路径

        self.trainer = DetectionTrainer(overrides=args)  # 创建训练器
        if not args.get("resume"):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)  # 获取模型
            self.model = self.trainer.model  # 更新模型

        self.trainer.train()  # 开始训练
        # 训练后更新模型和配置
        ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last  # 获取最佳检查点
        self.model, _ = attempt_load_one_weight(ckpt)  # 加载最佳权重
        self.overrides = self.model.args  # 更新覆盖参数
        self.metrics = getattr(self.trainer.validator, "metrics", None)  # 获取验证指标
        return self.metrics  # 返回指标
