# YOLO-Lite 🚀

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from yololite.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    RepC3,
    RepNCSPELAN4,
    ResNetLayer,
    SCDown,
)
from yololite.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, yaml_load
from yololite.utils.checks import check_requirements
from yololite.utils.loss import (
    E2EDetectLoss,
    v8DetectionLoss,
)
from yololite.utils.ops import make_divisible
from yololite.utils.plotting import feature_visualization
from yololite.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """BaseModel 类作为 YOLO-Lite 中所有模型的基类。"""

    def forward(self, x, *args, **kwargs):
        """
        执行模型的前向传播，用于训练或推理。

        如果 x 是字典，则计算并返回训练的损失。否则，返回推理的预测结果。

        参数：
            x (torch.Tensor | dict): 用于推理的输入张量，或带有图像张量和标签的字典用于训练。
            *args (Any): 可变长度的参数列表。
            **kwargs (Any): 任意关键字参数。

        返回：
            (torch.Tensor): 如果 x 是字典（训练），则返回损失；否则返回网络预测（推理）。
        """
        if isinstance(x, dict):  # 训练和验证时的情况。
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        通过网络执行前向传播。

        参数：
            x (torch.Tensor): 模型的输入张量。
            profile (bool): 如果为 True，打印每层的计算时间，默认为 False。
            visualize (bool): 如果为 True，保存模型的特征图，默认为 False。
            augment (bool): 在预测时增强图像，默认为 False。
            embed (list, optional): 要返回的特征向量/嵌入的列表。

        返回：
            (torch.Tensor): 模型的最后输出。
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        通过网络执行前向传播。

        参数：
            x (torch.Tensor): 模型的输入张量。
            profile (bool): 如果为 True，打印每层的计算时间，默认为 False。
            visualize (bool): 如果为 True，保存模型的特征图，默认为 False。
            embed (list, optional): 要返回的特征向量/嵌入的列表。

        返回：
            (torch.Tensor): 模型的最后输出。
        """
        y, dt, embeddings = [], [], []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不是来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自早期层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # 拉平
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强的推理。"""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} 不支持 'augment=True' 预测。"
            f"回退到单尺度预测。"
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        对模型单层的计算时间和 FLOPs 进行分析。结果附加到提供的列表中。

        参数：
            m (nn.Module): 要分析的层。
            x (torch.Tensor): 输入数据到该层。
            dt (list): 用于存储该层计算时间的列表。

        返回：
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # 是否为最终层列表，复制输入以修复就地问题
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _apply(self, fn):
        """
        将函数应用于模型中所有不是参数或注册缓冲区的张量。

        参数：
            fn (function): 要应用于模型的函数

        返回：
            (BaseModel): 更新后的 BaseModel 对象。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        将权重加载到模型中。

        参数：
            weights (dict | torch.nn.Module): 要加载的预训练权重。
            verbose (bool, optional): 是否记录转移进度。默认为 True。
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision 模型不是字典
        csd = model.float().state_dict()  # 检查点状态字典为 FP32
        csd = intersect_dicts(csd, self.state_dict())  # 交集
        self.load_state_dict(csd, strict=False)  # 加载
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """
        计算损失。

        参数：
            batch (dict): 计算损失的批次
            preds (torch.Tensor | List[torch.Tensor]): 预测结果。
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        """初始化 BaseModel 的损失标准。"""
        raise NotImplementedError("compute_loss() 需要由任务头实现")


class DetectionModel(BaseModel):
    """YOLOv8 检测模型。"""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # 模型，输入通道，类别数量
        """使用给定的配置和参数初始化 YOLOv8 检测模型。"""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 配置字典
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` 模块已弃用，替换为 nn.Identity。"
                "请删除本地 *.pt 文件并重新下载最新模型检查点。"
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # 定义模型
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"覆盖 model.yaml nc={self.yaml['nc']} 为 nc={nc}")
            self.yaml["nc"] = nc  # 覆盖 YAML 值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # 模型，保存列表
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # 默认名称字典
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # 构建步幅
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # 包括所有 Detect 子类，如 Segment、Pose、OBB、WorldDetect
            s = 256  # 2x 最小步幅
            m.inplace = self.inplace

            def _forward(x):
                """执行通过模型的前向传播，处理不同 Detect 子类类型。"""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向传播
            self.stride = m.stride
            m.bias_init()  # 仅运行一次
        else:
            self.stride = torch.Tensor([32])  # 默认步幅，即 RTDETR

        # 初始化权重，偏置
        initialize_weights(self)
        if verbose:
            LOGGER.info("")

    def _predict_augment(self, x):
        """对输入图像 x 执行增强并返回增强的推理和训练输出。"""
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("WARNING ⚠️ 模型不支持 'augment=True'，回退到单尺度预测。")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # 高度，宽度
        s = [1, 0.83, 0.67]  # 缩放
        f = [None, 3, None]  # 翻转（2-上下翻转，3-左右翻转）
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # 前向传播
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强的尾部
        return torch.cat(y, -1), None  # 增强的推理，训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """对增强推理后的预测进行反缩放（逆操作）。"""
        p[:, :4] /= scale  # 反缩放
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反翻转上下
        elif flips == 3:
            x = img_size[1] - x  # 反翻转左右
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """裁剪 YOLO 增强推理的尾部。"""
        nl = self.model[-1].nl  # 检测层数量 (P3-P5)
        g = sum(4**x for x in range(nl))  # 网格点
        e = 1  # 排除层计数
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小
        return y

    def init_criterion(self):
        """初始化 DetectionModel 的损失标准。"""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


# 函数 ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    用于暂时添加或修改 Python 模块缓存（`sys.modules`）中的模块的上下文管理器。

    此函数可用于在运行时更改模块路径。当重构代码时，
    如果将模块从一个位置移动到另一个位置，但仍希望支持旧的导入路径以保持向后兼容性，这将非常有用。

    参数：
        modules (dict, optional): 一个字典，将旧模块路径映射到新模块路径。
        attributes (dict, optional): 一个字典，将旧模块属性映射到新模块属性。

    示例：
        ```python
        with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
            import old.module  # 这将导入 new.module
            from old.module import attribute  # 这将导入 new.module.attribute
        ```

    注意：
        更改仅在上下文管理器内部生效，退出上下文管理器后将被撤销。
        直接操作 `sys.modules` 可能会导致不可预测的结果，特别是在较大的应用程序或库中。请谨慎使用此函数。
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # 在旧名称下设置 sys.modules 中的属性
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # 在旧名称下设置 sys.modules 中的模块
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # 删除临时模块路径
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """在反序列化期间替换未知类的占位符类。"""

    def __init__(self, *args, **kwargs):
        """初始化 SafeClass 实例，忽略所有参数。"""
        pass

    def __call__(self, *args, **kwargs):
        """运行 SafeClass 实例，忽略所有参数。"""
        pass


class SafeUnpickler(pickle.Unpickler):
    """自定义 Unpickler，用于在未知类时将其替换为 SafeClass。"""

    def find_class(self, module, name):
        """尝试查找一个类，如果不在安全模块中，则返回 SafeClass。"""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # 添加其他被认为是安全的模块
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """
    尝试使用 torch.load() 函数加载 PyTorch 模型。如果引发 ModuleNotFoundError，将捕获该错误，
    记录警告信息，并尝试通过 check_requirements() 函数安装缺少的模块。
    安装后，函数再次尝试使用 torch.load() 加载模型。

    参数：
        weight (str): PyTorch 模型的文件路径。
        safe_only (bool): 如果为 True，在加载时用 SafeClass 替换未知类。

    返回：
        ckpt (dict): 加载的模型检查点。
        file (str): 加载的文件名。
    """
    file = weight
    try:
        with temporary_modules():
            if safe_only:
                # 通过自定义 pickle 模块加载
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name 是缺少的模块名称
        check_requirements(e.name)  # 安装缺少的模块
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # 文件可能是 YOLO 实例，使用 i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ 文件 '{weight}' 似乎保存或格式不当。"
            f"为了获得最佳效果，请使用 model.save('filename.pt') 正确保存 YOLO 模型。"
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


class Ensemble(nn.ModuleList):
    """模型的集合。"""

    def __init__(self):
        """初始化模型的集合。"""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)  # NMS 集合，y 形状为 (B, HW, C)
        return y, None  # 推理，训练输出


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """加载模型集合 weights=[a,b,c] 或单个模型 weights=[a] 或 weights=a。"""
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # 加载检查点
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # 组合参数
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

        # 模型兼容性更新
        model.args = args  # 将参数附加到模型
        model.pt_path = w  # 将 *.pt 文件路径附加到模型
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # 添加到集合
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # 模型处于评估模式

    # 模块更新
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # 返回模型
    if len(ensemble) == 1:
        return ensemble[-1]

    # 返回集合
    LOGGER.info(f"创建的集合包含 {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"模型类别数量不同 {[m.nc for m in ensemble]}"
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """加载单个模型权重。"""
    ckpt, weight = torch_safe_load(weight)  # 加载检查点
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # 组合模型和默认参数，优先使用模型参数
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 模型

    # 模型兼容性更新
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 将参数附加到模型
    model.pt_path = weight  # 将 *.pt 文件路径附加到模型
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # 模型处于评估模式

    # 模块更新
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # 兼容 torch 1.11.0

    # 返回模型和检查点
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """将 YOLO 模型.yaml 字典解析为 PyTorch 模型。"""
    import ast

    # 参数
    legacy = True  # 向后兼容 v3/v5/v8/v9 模型
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ 未传递模型缩放，假设 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活，即 Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # 获取模块
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except ValueError:
                    pass
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # 深度增益
        if m in {
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            C3x,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # 如果 c2 不等于类别数量（即 Classify() 输出）
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # 嵌入通道
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
            }:
                args.insert(2, n)  # 重复次数
                n = 1
            if m is C3k2:  # 对于 M/L/X 尺寸
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # 重复次数
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, ImagePoolingAttn}:
            args.append([ch[x] for x in f])
            if m in {Detect}:
                m.legacy = legacy
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型
        m_.np = sum(x.numel() for x in m_.parameters())  # 参数数量
        m_.i, m_.f, m_.type = i, f, t  # 附加索引、'from' 索引、类型
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # 打印
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 添加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """从 YAML 文件加载 YOLOv8 模型。"""
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ YOLO P6 模型现在使用 -p6 后缀。重命名 {path.stem} 为 {new_stem}。")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # 即 yolov8x.yaml -> yolov8.yaml
    yaml_file = unified_path or path
    d = yaml_load(yaml_file)  # 模型字典
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    获取 YOLO 模型的 YAML 文件路径作为输入，并提取模型缩放的大小字符。该函数
    使用正则表达式匹配在 YAML 文件名中查找模型缩放的模式，缩放由
    n、s、m、l 或 x 表示。该函数返回模型缩放的大小字符作为字符串。

    参数：
        model_path (str | Path): YOLO 模型的 YAML 文件路径。

    返回：
        (str): 模型缩放的大小字符，可以是 n、s、m、l 或 x。
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # noqa，返回 n、s、m、l 或 x
    except AttributeError:
        return ""


def guess_model_task(model):
    return "detect"  # 假设是检测