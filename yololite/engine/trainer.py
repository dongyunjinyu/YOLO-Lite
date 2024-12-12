# YOLO-Lite 🚀

import gc
import math
import random
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from yololite.data import build_dataloader, build_yolo_dataset
from yololite.engine.validator import DetectionValidator
from yololite.nn.tasks import DetectionModel
from yololite.utils.plotting import plot_images, plot_labels, plot_results
from yololite.cfg import get_cfg, get_save_dir
from yololite.data.utils import check_det_dataset
from yololite.nn.tasks import attempt_load_one_weight, attempt_load_weights
from yololite.utils import (
    DEFAULT_CFG,
    LOGGER,
    TQDM,
    colorstr,
    yaml_save,
)
from yololite.utils.checks import check_imgsz, print_args
from yololite.utils.files import get_latest_run
from yololite.utils.torch_utils import (
    TORCH_2_4,
    de_parallel,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer
)

class DetectionTrainer:
    """
    属性：
        args (SimpleNamespace): 训练器的配置参数。
        validator (BaseValidator): 验证器实例。
        model (nn.Module): 模型实例。
        save_dir (Path): 保存结果的目录。
        wdir (Path): 保存权重的目录。
        last (Path): 最近检查点的路径。
        best (Path): 最佳检查点的路径。
        save_period (int): 每 x 个 epoch 保存一次检查点（小于 1 时禁用）。
        batch_size (int): 训练的批量大小。
        epochs (int): 训练的总 epoch 数。
        start_epoch (int): 训练开始的 epoch 数。
        device (torch.device): 训练使用的设备。
        amp (bool): 是否启用自动混合精度（AMP）的标志。
        scaler (amp.GradScaler): 用于 AMP 的梯度缩放器。
        data (str): 数据的路径。
        trainset (torch.utils.data.Dataset): 训练数据集。
        testset (torch.utils.data.Dataset): 测试数据集。
        ema (nn.Module): 模型的 EMA（指数移动平均）。
        resume (bool): 是否从检查点恢复训练。
        lf (nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        best_fitness (float): 达到的最佳适应度值。
        fitness (float): 当前适应度值。
        loss (float): 当前损失值。
        tloss (float): 总损失值。
        loss_names (list): 损失名称列表。
        csv (Path): 结果 CSV 文件的路径。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        参数：
            cfg (str, optional): 配置文件的路径。默认为 DEFAULT_CFG。
            overrides (dict, optional): 配置覆盖项。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)  # 加载配置
        self.check_resume(overrides)  # 检查是否需要恢复训练
        self.device = select_device(self.args.device, self.args.batch)  # 选择设备
        self.validator = None  # 验证器初始化
        self.metrics = None  # 评估指标初始化
        self.plots = {}  # 绘图字典初始化
        init_seeds(self.args.seed + 1, deterministic=self.args.deterministic)  # 初始化随机种子

        # 目录初始化
        self.save_dir = get_save_dir(self.args)  # 获取保存目录
        self.args.name = self.save_dir.name  # 更新日志用的名称
        self.wdir = self.save_dir / "weights"  # 权重目录
        self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
        self.args.save_dir = str(self.save_dir)  # 保存目录字符串
        yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period  # 保存周期

        # 训练参数
        self.batch_size = self.args.batch  # 批量大小
        self.epochs = self.args.epochs or 100  # 训练的 epoch 数
        self.start_epoch = 0  # 开始的 epoch
        print_args(vars(self.args))  # 打印参数

        # 设备设置
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # 使用 CPU 时设置工作线程为 0

        # 模型和数据集初始化
        self.model = self.args.model  # 模型初始化
        self.trainset, self.testset = self.get_dataset()  # 获取数据集
        self.ema = None  # EMA 初始化

        # 优化工具初始化
        self.lf = None  # 损失函数初始化
        self.scheduler = None  # 学习率调度器初始化

        # epoch 级别的指标
        self.best_fitness = None  # 最佳适应度
        self.fitness = None  # 当前适应度
        self.loss = None  # 当前损失
        self.tloss = None  # 总损失
        self.loss_names = ["Loss"]  # 损失名称
        self.csv = self.save_dir / "results.csv"  # 结果 CSV 文件路径
        self.plot_idx = [0, 1, 2]  # 绘图索引

    def _setup_scheduler(self):
        """初始化训练学习率调度器。"""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 余弦调度
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线性调度
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # 学习率调度器

    def _setup_train(self):
        """构建数据加载器和优化器。"""
        # 模型设置
        ckpt = self.setup_model()  # 设置模型
        self.model = self.model.to(self.device)  # 将模型移至设备
        self.set_model_attributes()  # 设置模型属性

        # 冻结层设置
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # 始终冻结的层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False  # 冻结参数
            elif not v.requires_grad and v.dtype.is_floating_point:  # 仅浮点类型可以要求梯度
                LOGGER.info(
                    f"警告 ⚠️ 设置 'requires_grad=True' 为冻结层 '{k}'。"
                )
                v.requires_grad = True

        # 检查 AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # 是否启用 AMP
        if self.amp:  # 单 GPU
            self.model = self.model.to(self.device)  # 移动模型到指定设备
        self.amp = bool(self.amp)  # 转为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # 检查图像大小
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格大小
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 检查图像大小
        self.stride = gs  # 用于多尺度训练

        # 数据加载器
        batch_size = self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, mode="train")

        # 测试数据加载器
        self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, mode="val")
        self.validator = self.get_validator()  # 获取验证器
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # 指标键
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # 初始化指标
        self.ema = ModelEMA(self.model)  # 初始化 EMA
        if self.args.plots:
            self.plot_training_labels()  # 绘制训练标签

        # 优化器设置
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 迭代次数
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # 设置学习率调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 提前停止
        self.resume_training(ckpt)  # 恢复训练
        self.scheduler.last_epoch = self.start_epoch - 1  # 不移动

    def train(self):
        self._setup_train()  # 设置训练
        nb = len(self.train_loader)  # 批次数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 热身迭代
        last_opt_step = -1  # 最后优化步数
        self.epoch_time = 0  # epoch 时间
        self.epoch_time_start = time.time()  # epoch 开始时间
        self.train_time_start = time.time()  # 训练开始时间
        LOGGER.info(
            f'图像大小 {self.args.imgsz} 训练, {self.args.imgsz} 验证\n'
            f"将结果记录到 {colorstr('bold', self.save_dir)}\n"
            f'开始训练 ' + (f"{self.args.time} 小时..." if self.args.time else f"{self.epochs} 个周期...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # 更新绘图索引
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清零梯度
        while True:
            self.epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略警告
                self.scheduler.step()  # 更新学习率

            self.model.train()  # 设置模型为训练模式
            pbar = enumerate(self.train_loader)  # 训练进度条
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()  # 关闭马赛克增强
                self.train_loader.reset()  # 重置训练加载器

            LOGGER.info(self.progress_string())  # 打印进度信息
            pbar = TQDM(enumerate(self.train_loader), total=nb)  # 进度条显示
            self.tloss = None  # 总损失初始化
            for i, batch in pbar:
                # 热身
                ni = i + nb * epoch  # 当前迭代次数
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))  # 更新累积参数
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 学习率调整
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向传播
                with autocast(self.amp):  # 自动混合精度
                    batch = self.preprocess_batch(batch)  # 预处理批次
                    self.loss, self.loss_items = self.model(batch)  # 计算损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # 反向传播
                self.scaler.scale(self.loss).backward()  # 缩放损失并反向传播

                # 优化
                if ni - last_opt_step >= self.accumulate:  # 检查是否达到优化条件
                    self.optimizer_step()  # 执行优化步
                    last_opt_step = ni  # 更新最后优化步数

                    # 时间停止检查
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:  # 超过训练时间
                            break

                # 日志记录
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",  # GPU 内存使用情况
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失值
                        batch["cls"].shape[0],  # 当前批量大小
                        batch["img"].shape[-1],  # 图像大小
                    )
                )
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)  # 绘制训练样本

            # 学习率记录
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 记录学习率
            final_epoch = epoch + 1 >= self.epochs  # 检查是否为最后一个 epoch
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])  # 更新 EMA

            # 验证
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()  # 验证模型
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # 保存指标
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch  # 检查是否停止训练
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 时间停止检查

            # 保存模型
            if self.args.save or final_epoch:
                self.save_model()  # 保存模型

            # 学习率调度
            t = time.time()
            self.epoch_time = t - self.epoch_time  # 计算 epoch 时间
            self.epoch_time_start = t  # 更新开始时间
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 平均 epoch 时间
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # 更新总 epoch
                self._setup_scheduler()  # 设置学习率调度器
                self.scheduler.last_epoch = self.epoch  # 不移动
                self.stop |= epoch >= self.epochs  # 检查是否超过总 epoch
            self._clear_memory()  # 清理内存

            # 提前停止检查
            if self.stop:
                break  # 结束训练
            epoch += 1  # 更新 epoch 计数

        # 最终验证
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} 个周期完成，耗时 {seconds / 3600:.3f} 小时。")
        self.final_eval()  # 最终评估
        if self.args.plots:
            self.plot_metrics()  # 绘制指标
        self._clear_memory()  # 清理内存

    def _get_memory(self):
        """获取加速器的内存利用率（单位：GB）。"""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()  # MPS 设备内存
        elif self.device.type == "cpu":
            memory = 0  # CPU 内存
        else:
            memory = torch.cuda.memory_reserved()  # CUDA 设备内存
        return memory / 1e9  # 返回内存以 GB 为单位

    def _clear_memory(self):
        """在不同平台上清理加速器内存。"""
        gc.collect()  # 垃圾回收
        if self.device.type == "mps":
            torch.mps.empty_cache()  # 清空 MPS 缓存
        elif self.device.type == "cpu":
            return  # CPU 不需要操作
        else:
            torch.cuda.empty_cache()  # 清空 CUDA 缓存

    def read_results_csv(self):
        """读取 results.csv 并返回字典格式的数据。"""
        import pandas as pd  # 延迟导入以加快速度

        return pd.read_csv(self.csv).to_dict(orient="list")  # 将 CSV 转为字典

    def save_model(self):
        """保存模型训练检查点及附加元数据。"""
        import io

        # 将检查点序列化到字节缓存中（比重复调用 torch.save() 更快）
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,  # 当前 epoch
                "best_fitness": self.best_fitness,  # 最佳适应度
                "model": None,  # 由 EMA 派生的检查点
                "ema": deepcopy(self.ema.ema).half(),  # EMA 模型
                "updates": self.ema.updates,  # EMA 更新次数
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # 优化器状态
                "train_args": vars(self.args),  # 保存参数字典
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # 训练指标
                "train_results": self.read_results_csv(),  # 读取训练结果
                "date": datetime.now().isoformat(),  # 当前日期时间
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化内容

        # 保存检查点
        self.last.write_bytes(serialized_ckpt)  # 保存 last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存 best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存当前 epoch

    def get_dataset(self):
        """检查并获取数据集。"""
        data = check_det_dataset(self.args.data)  # 检查数据集
        if "yaml_file" in data:
            self.args.data = data["yaml_file"]  # 验证 'yolo train data=url.zip' 使用
        self.data = data
        return data["train"], data.get("val") or data.get("test")  # 返回训练集和验证集

    def setup_model(self):
        """加载/创建/下载模型以用于任何任务。"""
        if isinstance(self.model, torch.nn.Module):  # 如果模型已加载，则无需设置
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)  # 尝试加载权重
            cfg = weights.yaml  # 获取配置
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)  # 尝试加载预训练权重
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=True)  # 创建模型
        return ckpt

    def optimizer_step(self):
        """执行一次训练优化器步骤，包含梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)  # 取消缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 裁剪梯度
        self.scaler.step(self.optimizer)  # 执行优化器步骤
        self.scaler.update()  # 更新缩放器
        self.optimizer.zero_grad()  # 清零梯度
        if self.ema:
            self.ema.update(self.model)  # 更新 EMA

    def preprocess_batch(self, batch):
        """预处理一批图像，进行缩放并转换为浮点数。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # 转换图像
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
            )  # 随机大小
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（拉伸到网格倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小
            batch["img"] = imgs  # 更新批次图像
        return batch

    def validate(self):
        """验证模型并返回指标。"""
        metrics = self.validator(self)  # 使用验证器进行验证
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 使用损失作为适应度
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness  # 更新最佳适应度
        return metrics, fitness  # 返回指标和适应度

    def get_model(self, cfg=None, weights=None, verbose=True):
        """创建并返回检测模型。"""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)  # 创建检测模型
        if weights:
            model.load(weights)  # 加载权重
        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"  # 损失名称
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))  # 创建验证器

    def get_dataloader(self, dataset_path, batch_size=16, mode="train"):
        """构建并返回数据加载器。"""
        assert mode in {"train", "val"}, f"模式必须为 'train' 或 'val'，而不是 {mode}。"
        dataset = self.build_dataset(dataset_path, mode, batch_size)  # 构建数据集
        shuffle = mode == "train"  # 训练时打乱数据
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("警告 ⚠️ 'rect=True' 与 DataLoader 打乱不兼容，设置 shuffle=False")
            shuffle = False  # 如果矩形模式，关闭打乱
        workers = self.args.workers if mode == "train" else self.args.workers * 2  # 设置工作线程数
        return build_dataloader(dataset, batch_size, workers, shuffle)  # 返回数据加载器

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建并返回 YOLO 数据集。"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # 网格大小
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)  # 返回数据集

    def set_model_attributes(self):
        """设置模型属性，如类别数量和名称。"""
        self.model.nc = self.data["nc"]  # 附加类别数量到模型
        self.model.names = self.data["names"]  # 附加类别名称到模型
        self.model.args = self.args  # 附加超参数到模型

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回带标签的损失字典。

        对于分类不需要，但对于分割和检测是必需的。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]  # 创建带前缀的损失名称
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为 5 位小数
            return dict(zip(keys, loss_items))  # 返回字典
        else:
            return keys  # 返回损失名称

    def progress_string(self):
        """返回格式化的训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",  # 当前周期
            "GPU_mem",  # GPU 内存
            *self.loss_names,  # 损失名称
            "Instances",  # 实例数量
            "Size",  # 图像大小
        )

    def plot_training_samples(self, batch, ni):
        """绘制带有注释的训练样本。"""
        plot_images(
            images=batch["img"],  # 图像
            batch_idx=batch["batch_idx"],  # 批次索引
            cls=batch["cls"].squeeze(-1),  # 类别
            bboxes=batch["bboxes"],  # 边界框
            paths=batch["im_file"],  # 文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,  # 绘图回调
        )

    def plot_metrics(self):
        """绘制来自 CSV 文件的指标。"""
        plot_results(file=self.csv, on_plot=self.on_plot)  # 保存结果图

    def plot_training_labels(self):
        """创建 YOLO 模型的标记训练图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)  # 合并所有边界框
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)  # 合并所有类别
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)  # 绘制标签

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 列数
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # 头部
        t = time.time() - self.train_time_start  # 计算经过时间
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")  # 写入数据

    def on_plot(self, name, data=None):
        """注册绘图（例如，供回调使用）。"""
        path = Path(name)  # 将名称转换为路径
        self.plots[path] = {"data": data, "timestamp": time.time()}  # 存储绘图数据和时间戳

    def final_eval(self):
        """执行最终评估和验证 YOLO 模型。"""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)  # 从 last.pt 中移除优化器
                elif f is self.best:
                    k = "train_results"  # 从 last.pt 更新 best.pt 的训练指标
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\n验证 {f}...")
                    self.validator.args.plots = self.args.plots  # 更新绘图参数
                    self.metrics = self.validator(model=f)  # 验证模型
                    self.metrics.pop("fitness", None)  # 移除适应度

    def check_resume(self, overrides):
        """检查是否存在恢复检查点，并相应更新参数。"""
        resume = self.args.resume  # 恢复标志
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()  # 检查恢复路径是否存在
                last = Path(resume if exists else get_latest_run())  # 获取最新检查点

                # 检查恢复数据 YAML 是否存在，否则强制重新下载数据集
                ckpt_args = attempt_load_weights(last).args  # 获取检查点参数
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data  # 更新数据路径

                resume = True
                self.args = get_cfg(ckpt_args)  # 重新加载配置
                self.args.model = self.args.resume = str(last)  # 重新设置模型
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # 允许参数更新
                    if k in overrides:
                        setattr(self.args, k, overrides[k])  # 更新参数

            except Exception as e:
                raise FileNotFoundError("恢复检查点未找到。") from e
        self.resume = resume  # 设置恢复标志

    def resume_training(self, ckpt):
        """从检查点恢复训练。"""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0  # 最佳适应度初始化
        start_epoch = ckpt.get("epoch", -1) + 1  # 恢复的起始 epoch
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 恢复优化器状态
            best_fitness = ckpt["best_fitness"]  # 恢复最佳适应度
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 恢复 EMA
            self.ema.updates = ckpt["updates"]  # 恢复更新次数
        assert start_epoch > 0, (
            f"{self.args.model} 训练到 {self.epochs} 个周期已完成，无需恢复。\n"
            f"开始新训练，而不是恢复，i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"恢复训练 {self.args.model} 从第 {start_epoch + 1} 个周期到总共 {self.epochs} 个周期")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} 已训练 {ckpt['epoch']} 个周期。微调 {self.epochs} 个周期。"
            )
            self.epochs += ckpt["epoch"]  # 微调额外的周期
        self.best_fitness = best_fitness  # 更新最佳适应度
        self.start_epoch = start_epoch  # 更新起始 epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()  # 关闭马赛克增强

    def _close_dataloader_mosaic(self):
        """更新数据加载器以停止使用马赛克增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False  # 关闭马赛克
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("关闭数据加载器马赛克")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))  # 关闭马赛克增强

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        构建优化器。

        参数：
            model (torch.nn.Module): 要为其构建优化器的模型。
            name (str, optional): 使用的优化器名称。如果为 'auto'，则根据迭代次数选择优化器。默认值：'auto'。
            lr (float, optional): 优化器的学习率。默认值：0.001。
            momentum (float, optional): 优化器的动量因子。默认值：0.9。
            decay (float, optional): 优化器的权重衰减。默认值：1e-5。
            iterations (float, optional): 迭代次数，决定优化器类型（如果名称为 'auto'）。默认值：1e5。
        """
        g = [], [], []  # 优化器参数组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' 被找到，"
                f"忽略 'lr0={self.args.lr0}' 和 'momentum={self.args.momentum}'，"
                f"自动确定最佳 'optimizer'、'lr0' 和 'momentum'... "
            )
            nc = getattr(model, "nc", 10)  # 类别数量
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 适配方程
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # Adam 的学习率不高于 0.01

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname =f"{module_name}.{param_name}" if module_name else param_name  # 完整参数名称
                if "bias" in fullname:  # 偏置参数（不衰减）
                    g[2].append(param)  # 添加到偏置参数组
                elif isinstance(module, bn):  # 归一化层（不衰减）
                    g[1].append(param)  # 添加到归一化参数组
                else:  # 权重参数（衰减）
                    g[0].append(param)  # 添加到权重参数组

        # 可用的优化器
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())  # 将名称转为小写
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # 创建 Adam 类优化器
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)  # 创建 RMSProp 优化器
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)  # 创建 SGD 优化器
        else:
            raise NotImplementedError(f"优化器 '{name}' 未找到。")  # 抛出未实现错误

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 添加 g0（带权重衰减的权重参数）
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 添加 g1（不带衰减的归一化层权重）
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) 带参数组 "
            f'{len(g[1])} 权重(衰减=0.0)，{len(g[0])} 权重(衰减={decay})，{len(g[2])} 偏置(衰减=0.0)'
        )
        return optimizer  # 返回优化器实例