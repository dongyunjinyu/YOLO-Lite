# YOLO-Lite 🚀

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List
import numpy as np
from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh

def _ntuple(n):
    """From PyTorch internals."""
    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` 表示左上角和右下角
# `xywh` 表示中心 x，中心 y 和宽度，高度（YOLO 格式）
# `ltwh` 表示左上角和宽度，高度（COCO 格式）
_formats = ["xyxy", "xywh", "ltwh"]
__all__ = ("Bboxes",)  # tuple or list


class Bboxes:
    """
    处理边界框的类。

    该类支持多种边界框格式，如 'xyxy'、'xywh' 和 'ltwh'。
    边界框数据应以 numpy 数组的形式提供。

    属性：
        bboxes (numpy.ndarray)：以 2D numpy 数组存储的边界框。
        format (str)：边界框的格式（'xyxy'、'xywh' 或 'ltwh'）。

    注意：
        此类不处理边界框的归一化或反归一化。
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """使用指定格式的边界框数据初始化 Bboxes 类。"""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format):
        """将边界框格式从一种类型转换为另一种类型。"""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """返回边界框的面积。"""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    def mul(self, scale):
        """
        将边界框坐标乘以缩放因子。

        参数：
            scale (int | tuple | list)：四个坐标的缩放因子。
                如果是 int，则对所有坐标应用相同的缩放。
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        将偏移量添加到边界框坐标。

        参数：
            offset (int | tuple | list)：四个坐标的偏移量。
                如果是 int，则对所有坐标应用相同的偏移。
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """返回边界框的数量。"""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> "Bboxes":
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


class Instances:
    """
    图像中检测到的对象边界框的容器。

    属性：
        _bboxes (Bboxes)：用于处理边界框操作的内部对象。
        normalized (bool)：指示边界框坐标是否已归一化的标志。

    参数：
        bboxes (ndarray)：形状为 [N, 4] 的边界框数组。
        bbox_format (str, optional)：边界框的格式（'xywh' 或 'xyxy'）。默认是 'xywh'。
        normalized (bool, optional)：边界框坐标是否已归一化。默认是 True。
    """

    def __init__(self, bboxes, bbox_format="xywh", normalized=True) -> None:
        """
        Initialize the object with bounding boxes
        Args:
            bboxes (np.ndarray): Bounding boxes, shape [N, 4].
            bbox_format (str, optional): Format of bboxes. Defaults to "xywh".
            normalized (bool, optional): Whether the coordinates are normalized. Defaults to True.
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.normalized = normalized

    def convert_bbox(self, format):
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """Similar to denormalize func but without normalized sign."""
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only:
            return

    def denormalize(self, w, h):
        """Denormalizes boxes from normalized coordinates."""
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        self.normalized = False

    def normalize(self, w, h):
        """Normalize bounding boxes to image dimensions."""
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        self.normalized = True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation."""
        assert not self.normalized, "you should add padding with absolute coordinates."
        self._bboxes.add(offset=(padw, padh, padw, padh))

    def __getitem__(self, index) -> "Instances":
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes if present.
        """
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        """Flips the coordinates of bounding boxes vertically."""
        if self._bboxes.format == "xyxy":
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]

    def fliplr(self, w):
        """Reverses the order of the bounding boxes horizontally."""
        if self._bboxes.format == "xyxy":
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]

    def clip(self, w, h):
        """Clips bounding boxes values to stay within image boundaries."""
        ori_format = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori_format != "xyxy":
            self.convert_bbox(format=ori_format)

    def remove_zero_area_boxes(self):
        """Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height."""
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
        return good

    def update(self, bboxes):
        """Updates instance variables."""
        self._bboxes = Bboxes(bboxes, format=self._bboxes.format)

    def __len__(self):
        """Return the length of the instance list."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        return cls(cat_boxes, bbox_format, normalized)

    @property
    def bboxes(self):
        """Return bounding boxes."""
        return self._bboxes.bboxes
