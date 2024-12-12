# YOLO-Lite 🚀
"""猴子补丁，用于更新/扩展现有函数的功能。"""

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from yololite.utils import LOGGER
from yololite.utils.metrics import batch_probiou


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile 类。可以作为装饰器 @Profile() 使用，或作为上下文管理器使用 'with Profile():'。

    示例：
        ```python
        from yololite.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # 这里是慢操作

        print(dt)  # 输出 "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        初始化 Profile 类。

        参数：
            t (float): 初始时间。默认为 0.0。
            device (torch.device): 用于模型推理的设备。默认为 None（cpu）。
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """开始计时。"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """停止计时。"""
        self.dt = self.time() - self.start  # 增量时间
        self.t += self.dt  # 累加增量时间

    def __str__(self):
        """返回表示累积经过时间的可读字符串。"""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """获取当前时间。"""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    将边界框（默认格式为 xyxy）从它们最初指定的图像形状（img1_shape）缩放到另一幅图像的形状（img0_shape）。

    参数：
        img1_shape (tuple): 边界框对应的图像的形状，格式为 (高度, 宽度)。
        boxes (torch.Tensor): 图像中对象的边界框，格式为 (x1, y1, x2, y2)
        img0_shape (tuple): 目标图像的形状，格式为 (高度, 宽度)。
        ratio_pad (tuple): 一个 (ratio, pad) 的元组，用于缩放边界框。如果未提供，将根据两个图像之间的大小差异计算比率和填充。
        padding (bool): 如果为 True，假设边界框基于 yolo 风格增强的图像。如果为 False，则进行常规缩放。
        xywh (bool): 边界框格式是否为 xywh，默认为 False。

    返回：
        boxes (torch.Tensor): 缩放后的边界框，格式为 (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # 从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh 填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x 填充
        boxes[..., 1] -= pad[1]  # y 填充
        if not xywh:
            boxes[..., 2] -= pad[0]  # x 填充
            boxes[..., 3] -= pad[1]  # y 填充
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    返回最接近的可被给定除数整除的数字。

    参数：
        x (int): 要整除的数字。
        divisor (int | torch.Tensor): 除数。

    返回：
        (int): 最近的可被除数整除的数字。
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # 转为 int
    return math.ceil(x / divisor) * divisor


def nms_rotated(boxes, scores, threshold=0.45):
    """
    使用 probiou 和 fast-nms 对旋转边界框进行 NMS。

    参数：
        boxes (torch.Tensor): 旋转边界框，形状为 (N, 5)，格式为 xywhr。
        scores (torch.Tensor): 置信分数，形状为 (N,).
        threshold (float, optional): IoU 阈值。默认为 0.45。

    返回：
        (torch.Tensor): 保留的框的索引。
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # 类别数量（可选）
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    对一组框执行非最大抑制（NMS），支持掩码和每个框多个标签。

    参数：
        prediction (torch.Tensor): 形状为 (batch_size, num_classes + 4 + num_masks, num_boxes) 的张量，
            包含预测的框、类别和掩码。张量应为模型输出格式，例如 YOLO。
        conf_thres (float): 置信度阈值，低于该值的框将被过滤掉。
            有效值在 0.0 和 1.0 之间。
        iou_thres (float): 在 NMS 过程中将被过滤掉的 IoU 阈值。
            有效值在 0.0 和 1.0 之间。
        classes (List[int]): 要考虑的类别索引列表。如果为 None，将考虑所有类别。
        agnostic (bool): 如果为 True，模型对类别数量无关，并且所有类别将被视为一个。
        multi_label (bool): 如果为 True，每个框可能有多个标签。
        labels (List[List[Union[int, float, torch.Tensor]]]): 一个列表的列表，每个内层
            列表包含给定图像的先验标签。列表应为数据加载器输出格式，
            每个标签为 (class_index, x1, y1, x2, y2) 的元组。
        max_det (int): NMS 后保留的最大框数量。
        nc (int, optional): 模型输出的类别数量。此之后的任何索引将视为掩码。
        max_time_img (float): 处理一张图像的最大时间（秒）。
        max_nms (int): 传递给 torchvision.ops.nms() 的最大框数量。
        max_wh (int): 像素的最大框宽度和高度。
        in_place (bool): 如果为 True，将原输入预测张量就地修改。
        rotated (bool): 如果传递的是定向边界框（OBB）用于 NMS。

    返回：
        (List[torch.Tensor]): 一个长度为 batch_size 的列表，每个元素是一个形状为
            (num_boxes, 6 + num_masks) 的张量，包含保留的框，列
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # 为了更快的 'import yololite'

    # 检查
    assert 0 <= conf_thres <= 1, f"无效的置信度阈值 {conf_thres}，有效值在 0.0 和 1.0 之间"
    assert 0 <= iou_thres <= 1, f"无效的 IoU {iou_thres}，有效值在 0.0 和 1.0 之间"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 模型在验证模型中，输出 = (inference_out, loss_out)
        prediction = prediction[0]  # 仅选择推理输出
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # 端到端模型 (BNC，即 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # 批处理大小 (BCN，即 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # 类别数量
    nm = prediction.shape[1] - nc - 4  # 掩码数量
    mi = 4 + nc  # 掩码开始索引
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # 候选框

    # 设置
    # min_wh = 2  # (像素) 最小框宽度和高度
    time_limit = 2.0 + max_time_img * bs  # 超过后退出的秒数
    multi_label &= nc > 1  # 每个框多个标签（增加 0.5ms/img）

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) 转换为 shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh 转 xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh 转 xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # 图像索引，图像推理
        # 应用约束
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # 宽高
        x = x[xc[xi]]  # 置信度

        # 如果自动标注，则连接先前标签
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # 边界框
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # 类
            x = torch.cat((x, v), 0)

        # 如果没有剩余，处理下一个图像
        if not x.shape[0]:
            continue

        # 检测矩阵 nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # 仅最佳类别
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # 按类别筛选
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # 检查形状
        n = x.shape[0]  # 框数量
        if not n:  # 没有框
            continue
        if n > max_nms:  # 超过框数量
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序并移除多余框

        # 批量 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        scores = x[:, 4]  # 分数
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # 框（按类别偏移）
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # 限制检测数量

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS 超过时间限制 {time_limit:.3f}s")
            break  # 超过时间限制

    return output


def clip_boxes(boxes, shape):
    """
    获取边界框列表和形状（高度，宽度），并将边界框剪裁到该形状。

    参数：
        boxes (torch.Tensor): 要剪裁的边界框
        shape (tuple): 图像的形状

    返回：
        (torch.Tensor | numpy.ndarray): 剪裁后的框
    """
    if isinstance(boxes, torch.Tensor):  # 更快的单独处理（警告：就地 .clamp_() 的 Apple MPS bug）
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array（更快的批量处理）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    将线坐标剪裁到图像边界。

    参数：
        coords (torch.Tensor | numpy.ndarray): 一组线坐标。
        shape (tuple): 代表图像大小的整数元组，格式为 (高度, 宽度)。

    返回：
        (torch.Tensor | numpy.ndarray): 剪裁后的坐标
    """
    if isinstance(coords, torch.Tensor):  # 更快的单独处理（警告：就地 .clamp_() 的 Apple MPS bug）
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array（更快的批量处理）
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    获取一个掩码，并将其调整为原始图像大小。

    参数：
        masks (np.ndarray): 调整大小和填充的掩码/图像，形状为 [h, w, num]/[h, w, 3]。
        im0_shape (tuple): 原始图像的形状
        ratio_pad (tuple): 用于原始图像的填充比率。

    返回：
        masks (np.ndarray): 形状为 [h, w, num] 的掩码。
    """
    # 从 im1_shape 缩放坐标 (xyxy) 到 im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # 从 im0_shape 计算
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh 填充
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" 应为 2 或 3，但得到 {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式，其中 (x1, y1) 是
    左上角，(x2, y2) 是右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x1, y1, x2, y2)。

    返回：
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 (x, y, width, height)。
    """
    assert x.shape[-1] == 4, f"输入形状最后维度期望为 4，但输入形状为 {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # 比克隆/复制更快
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x 中心
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y 中心
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def xywh2xyxy(x):
    """
    将边界框坐标从 (x, y, width, height) 格式转换为 (x1, y1, x2, y2) 格式，其中 (x1, y1) 是
    左上角，(x2, y2) 是右下角。注意：每 2 个通道的操作比每个通道的操作更快。

    参数：
        x (np.ndarray | torch.Tensor): 输入的边界框坐标，格式为 (x, y, width, height)。

    返回：
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 (x1, y1, x2, y2)。
    """
    assert x.shape[-1] == 4, f"输入形状最后维度期望为 4，但输入形状为 {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # 比克隆/复制更快
    xy = x[..., :2]  # 中心
    wh = x[..., 2:] / 2  # 半宽高
    y[..., :2] = xy - wh  # 左上角 xy
    y[..., 2:] = xy + wh  # 右下角 xy
    return y


def xywh2ltwh(x):
    """
    将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h]，其中 x1, y1 是左上角坐标。

    参数：
        x (np.ndarray | torch.Tensor): 输入张量，包含边界框坐标。

    返回：
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上角 x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上角 y
    return y


def xyxy2ltwh(x):
    """
    将 nx4 边界框从 [x1, y1, x2, y2] 转换为 [x1, y1, w, h]，其中 xy1=左上角，xy2=右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入张量，包含边界框坐标，格式为 xyxy。

    返回：
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 xyltwh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def ltwh2xywh(x):
    """
    将 nx4 框从 [x1, y1, w, h] 转换为 [x, y, w, h]，其中 xy1=左上角，xy=center。

    参数：
        x (torch.Tensor): 输入张量。

    返回：
        y (np.ndarray | torch.Tensor): 边界框坐标，格式为 xywh。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # 中心 x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # 中心 y
    return y


def xywhr2xyxyxyxy(x):
    """
    将批量定向边界框（OBB）从 [xywh, 旋转] 转换为 [xy1, xy2, xy3, xy4]。旋转值应为
    从 0 到 pi/2 的弧度。

    参数：
        x (numpy.ndarray | torch.Tensor): 形状为 (n, 5) 或 (b, n, 5) 的框，格式为 [cx, cy, w, h, 旋转]。

    返回：
        (numpy.ndarray | torch.Tensor): 转换后的角点，形状为 (n, 4, 2) 或 (b, n, 4, 2)。
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    将边界框从 [x1, y1, w, h] 转换为 [x1, y1, x2, y2]，其中 xy1=左上角，xy2=右下角。

    参数：
        x (np.ndarray | torch.Tensor): 输入图像

    返回：
        y (np.ndarray | torch.Tensor): 边界框的 xyxy 坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] + x[..., 1]  # 高度
    return y


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)