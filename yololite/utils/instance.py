# Ultralytics YOLO 🚀, AGPL-3.0 license

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

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes",)  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """Converts bounding box format from one type to another."""
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
        """Return box areas."""
        return (
            (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])  # format xyxy
            if self.format == "xyxy"
            else self.bboxes[:, 3] * self.bboxes[:, 2]  # format xywh or ltwh
        )

    def mul(self, scale):
        """
        Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates.
                If int, the same scale is applied to all coordinates.
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
        Add offset to bounding box coordinates.

        Args:
            offset (int | tuple | list): Offset(s) for four coordinates.
                If int, the same offset is applied to all coordinates.
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
        """Return the number of boxes."""
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
    Container for bounding boxes of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4].
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        normalized (bool, optional): Whether the bounding box coordinates are normalized. Default is True.
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
