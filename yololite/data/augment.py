# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy
from typing import Tuple, Union
import cv2
import numpy as np
import torch

from yololite.utils import LOGGER, colorstr
from yololite.utils.checks import check_version
from yololite.utils.instance import Instances
from yololite.utils.metrics import bbox_ioa

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

class Compose:
    """
    A class for composing multiple image transformations.

    Attributes:
        transforms (List[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Applies a series of transformations to input data.
        append: Appends a new transform to the existing list of transforms.
        insert: Inserts a new transform at a specified index in the list of transforms.
        __getitem__: Retrieves a specific transform or a set of transforms using indexing.
        __setitem__: Sets a specific transform or a set of transforms using indexing.
        tolist: Converts the list of transforms to a standard Python list.
    """

    def __init__(self, transforms):
        """
        Initializes the Compose object with a list of transforms.

        Args:
            transforms (List[Callable]): A list of callable transform objects to be applied sequentially.
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """
        Applies a series of transformations to input data. This method sequentially applies each transformation in the
        Compose object's list of transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the
                transformations in the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """
        Appends a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        Inserts a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """
        Retrieves a specific transform or a set of transforms using indexing.

        Args:
            index (int | List[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose): A new Compose object containing the selected transform(s).

        Raises:
            AssertionError: If the index is not of type int or list.
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """
        Sets one or more transforms in the composition using indexing.

        Args:
            index (int | List[int]): Index or list of indices to set transforms at.
            value (Any | List[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """
        Converts the list of transforms to a standard Python list.

        Returns:
            (List): A list containing all the transform objects in the Compose instance.
            3
        """
        return self.transforms

    def __repr__(self):
        """
        Returns a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"

class BaseMixTransform:
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)
        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        raise NotImplementedError

    def get_indexes(self):
        raise NotImplementedError

    def _update_label_text(self, labels):
        if "texts" not in labels:
            return labels

        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (Tuple[int, int]): Border size for width and height.

    Methods:
        get_indexes: Returns a list of random indexes from the dataset.
        _mix_transform: Applies mixup transformation to the input image and labels.
        _mosaic3: Creates a 1x3 image mosaic.
        _mosaic4: Creates a 2x2 image mosaic.
        _mosaic9: Creates a 3x3 image mosaic.
        _update_labels: Updates labels with padding.
        _cat_labels: Concatenates labels and clips mosaic border instances.
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def get_indexes(self, buffer=True):
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    def _mosaic3(self, labels):
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            if i == 0:  # center
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels

class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Attributes:
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    Methods:
        get_indexes: Returns a random index from the dataset.
        _mix_transform: Applies MixUp augmentation to the input labels.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels

class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (Tuple[int, int]): Mosaic border size as (x, y).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.

    Methods:
        affine_transform: Applies affine transformations to the input image.
        apply_bboxes: Transforms bounding boxes using the affine matrix.
        apply_segments: Transforms segments and generates new bounding boxes.
        apply_keypoints: Transforms keypoints using the affine matrix.
        __call__: Applies the random perspective transformation to images and annotations.
        box_candidates: Filters transformed bounding boxes based on size and aspect ratio.
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def __call__(self, labels):
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        new_instances = Instances(bboxes, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

class RandomHSV:
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels

class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.
    """
    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels

class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """
        Updates labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (Dict): A dictionary containing image labels and instances.
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

class CopyPaste(BaseMixTransform):
    """
    CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    Attributes:
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.
    """

    def __init__(self, dataset=None, pre_transform=None, p=0.5, mode="flip") -> None:
        """Initializes CopyPaste object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def get_indexes(self):
        """Returns a list of random indexes from the dataset for CopyPaste augmentation."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies Copy-Paste augmentation to combine objects from another image into the current image."""
        labels2 = labels["mix_labels"][0]
        return self._transform(labels, labels2)

    def __call__(self, labels):
        """Applies Copy-Paste augmentation to an image and its labels."""
        if self.p == 0:
            return labels
        if self.mode == "flip":
            return self._transform(labels)

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)
        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _transform(self, labels1, labels2={}):
        """Applies Copy-Paste augmentation to combine objects from another image into the current image."""
        im = labels1["img"]
        cls = labels1["cls"]
        h, w = im.shape[:2]
        instances = labels1.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        im_new = np.zeros(im.shape, np.uint8)
        instances2 = labels2.pop("instances", None)
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)
        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)  # intersection over area, (N, M)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        for j in indexes[: round(self.p * n)]:
            cls = np.concatenate((cls, labels2.get("cls", cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

        result = labels2.get("img", cv2.flip(im, 1))  # augment segments
        i = im_new.astype(bool)
        im[i] = result[i]

        labels1["img"] = im
        labels1["cls"] = cls
        labels1["instances"] = instances
        return labels1

class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.
    """

    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        if self.transform is None or random.random() > self.p:
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                im = labels["img"]
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels

class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        batch_idx=True,
        bgr=0.0,
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        return img

def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Dict): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms
