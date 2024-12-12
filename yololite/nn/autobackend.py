# YOLO-Lite 🚀

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from yololite.utils import LOGGER, yaml_load


def default_class_names(data=None):
    """应用默认类名到输入的 YAML 文件，或返回数字类名。"""
    if data:
        try:
            return yaml_load(data)["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # 返回默认值如果发生错误


class AutoBackend(nn.Module):
    """
    处理使用 YOLO-Lite 模型进行推理的动态后端选择。

    AutoBackend 类旨在为各种推理引擎提供一个抽象层。它支持广泛的格式，每种格式都有特定的命名约定，如下所示：

        支持的格式和命名约定：
            | 格式                  | 文件后缀         |
            |-----------------------|-------------------|
            | PyTorch               | *.pt              |

    此类基于输入模型格式提供动态后端切换功能，使模型在各种平台上更易于部署。
    """

    @torch.no_grad()
    def __init__(
        self,
        weights="yolo11n.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        batch=1,
        fuse=True,
        verbose=True,
    ):
        """
        初始化 AutoBackend 以进行推理。

        参数：
            weights (str): 模型权重文件的路径。默认为 'yolov8n.pt'。
            device (torch.device): 运行模型的设备。默认为 CPU。
            dnn (bool): 使用 OpenCV DNN 模块进行 ONNX 推理。默认为 False。
            data (str | Path | optional): 包含类名的额外 data.yaml 文件的路径。可选。
            fp16 (bool): 启用半精度推理。仅在特定后端上支持。默认为 False。
            batch (int): 假设的推理批处理大小。
            fuse (bool): 融合 Conv2D + BatchNorm 层以优化。默认为 True。
            verbose (bool): 启用详细日志记录。默认为 True。
        """

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt = (w.split('.')[1] == 'pt')
        fp16 &= True
        model, metadata, task = None, None, None

        # 设置设备
        cuda = torch.cuda.is_available() and device.type != "cpu"  # 使用 CUDA

        # 内存中的 PyTorch 模型
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # 仅限姿势
            stride = max(int(model.stride.max()), 32)  # 模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类名
            model.half() if fp16 else model.float()
            self.model = model  # 明确分配给 to()、cpu()、cuda()、half()
            pt = True

        # PyTorch
        elif pt:
            from yololite.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # 仅限姿势
            stride = max(int(model.stride.max()), 32)  # 模型步幅
            names = model.module.names if hasattr(model, "module") else model.names  # 获取类名
            model.half() if fp16 else model.float()
            self.model = model  # 明确分配给 to()、cpu()、cuda()、half()

        # 加载外部元数据 YAML
        if isinstance(metadata, (str, Path)) and Path(metadata).exists():
            metadata = yaml_load(metadata)
        if metadata and isinstance(metadata, dict):
            for k, v in metadata.items():
                if k in {"stride", "batch"}:
                    metadata[k] = int(v)
                elif k in {"imgsz", "names", "kpt_shape"} and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata["stride"]
            task = metadata["task"]
            batch = metadata["batch"]
            imgsz = metadata["imgsz"]
            names = metadata["names"]
            kpt_shape = metadata.get("kpt_shape")
        elif not (pt or nn_module):
            LOGGER.warning(f"WARNING ⚠️ 找不到元数据 'model={weights}'")

        # 检查类名
        if "names" not in locals():  # 缺少类名
            names = default_class_names(data)

        # 禁用梯度
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # 将所有变量分配给 self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        在 YOLOv8 MultiBackend 模型上运行推理。

        参数：
            im (torch.Tensor): 要执行推理的图像张量。
            augment (bool): 在推理期间是否执行数据增强，默认为 False。
            visualize (bool): 是否可视化输出预测，默认为 False。
            embed (list, optional): 要返回的特征向量/嵌入的列表。

        返回：
            (tuple): 包含原始输出张量的元组，以及处理后的输出用于可视化（如果 visualize=True）。
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # 转为 FP16

        y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # 段和名称未定义
                ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # 原型，框的索引
                nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """将 numpy 数组转换为张量。"""
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """通过使用虚拟输入运行一次前向传播来预热模型。"""
        import torchvision  # noqa（导入此处以便 torchvision 导入时间不记录在后处理时间中）

        warmup_types = self.pt, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu"):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # 预热
