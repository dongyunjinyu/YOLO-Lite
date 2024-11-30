# Ultralytics YOLOLite 🚀, AGPL-3.0 license

from itertools import repeat
import torch
from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    v8_transforms,
)
from .utils import (
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image_label,
)

DATASET_CACHE_VERSION = "1.0.3"
import glob
import math
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from torch.utils.data import Dataset
from yololite.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from yololite.utils import DEFAULT_CFG, LOGGER, NUM_THREADS, TQDM


class YOLODataset(Dataset):
    """
    用于加载和处理图像数据的基础数据集类。

    参数：
        img_path (str): 包含图像的文件夹路径。
        imgsz (int, optional): 图像大小。默认为 640。
        cache (bool, optional): 在训练期间将图像缓存到 RAM 或磁盘。默认为 False。
        augment (bool, optional): 如果为 True，则应用数据增强。默认为 True。
        hyp (dict, optional): 应用数据增强的超参数。默认为 None。
        prefix (str, optional): 日志消息中打印的前缀。默认为 ''。
        rect (bool, optional): 如果为 True，则使用矩形训练。默认为 False。
        batch_size (int, optional): 批量大小。默认为 None。
        stride (int, optional): 步幅。默认为 32。
        pad (float, optional): 填充。默认为 0.0。
        single_cls (bool, optional): 如果为 True，则使用单类训练。默认为 False。
        classes (list): 包含的类别列表。默认为 None。

    属性：
        im_files (list): 图像文件路径列表。
        labels (list): 标签数据字典列表。
        ni (int): 数据集中图像的数量。
        ims (list): 加载的图像列表。
        npy_files (list): Numpy 文件路径列表。
        transforms (callable): 图像转换函数。
    """

    def __init__(
            self,
            img_path,
            imgsz=640,
            augment=True,
            hyp=DEFAULT_CFG,
            prefix="",
            rect=False,
            batch_size=16,
            stride=32,
            pad=0.5,
            single_cls=False,
            classes=None,
            data=None
    ):
        super().__init__()
        self.img_path = img_path  # 图像路径
        self.imgsz = imgsz  # 图像大小
        self.augment = augment  # 是否使用数据增强
        self.single_cls = single_cls  # 是否使用单类训练
        self.prefix = prefix  # 日志前缀
        self.im_files = self.get_img_files(self.img_path)  # 获取图像文件路径
        self.labels = self.get_labels()  # 获取标签信息
        self.update_labels(include_class=classes)  # 更新标签以包含指定类别
        self.ni = len(self.labels)  # 数据集中图像的数量
        self.rect = rect  # 是否使用矩形训练
        self.batch_size = batch_size  # 批量大小
        self.stride = stride  # 步幅
        self.pad = pad  # 填充
        self.data = data  # 其他数据（可选）

        if self.rect:
            assert self.batch_size is not None  # 确保批量大小已定义
            self.set_rectangle()  # 设置矩形训练参数

        # 缓存图像配置
        self.buffer = []  # 用于存储图像的缓冲区
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0  # 最大缓冲区长度

        # 缓存图像（选项为：cache = True, False, None, "ram", "disk"）
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni  # 图像及其尺寸缓存
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]  # Numpy 文件路径列表

        # 设置图像转换
        self.transforms = self.build_transforms(hyp=hyp)  # 构建图像转换函数

    def get_img_files(self, img_path):
        f = []  # 图像文件列表
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # 处理路径
            if p.is_dir():  # 如果是文件夹
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)  # 获取所有图像文件
            elif p.is_file():  # 如果是文件
                with open(p) as t:
                    t = t.read().strip().splitlines()  # 读取文件内容
                    parent = str(p.parent) + os.sep  # 获取父目录
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # 转换为全局路径
            else:
                raise FileNotFoundError(f"{self.prefix}{p} 不存在")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)  # 过滤有效的图像格式
        assert im_files, f"在 {img_path} 中未找到任何图像。 {FORMATS_HELP_MSG}"  # 确保至少找到一个图像文件
        return im_files

    def update_labels(self, include_class: Optional[list]):
        include_class_array = np.array(include_class).reshape(1, -1)  # 转换为数组格式
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]  # 类别
                bboxes = self.labels[i]["bboxes"]  # 边界框
                j = (cls == include_class_array).any(1)  # 检查类别是否在指定类别中
                self.labels[i]["cls"] = cls[j]  # 更新类别
                self.labels[i]["bboxes"] = bboxes[j]  # 更新边界框
            if self.single_cls:  # 如果使用单类训练
                self.labels[i]["cls"][:, 0] = 0  # 将所有类别设置为 0

    def load_image(self, i, rect_mode=True):
        """从数据集中加载第 i 张图像，返回 (im, resized hw)。"""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]  # 获取当前图像及其文件名
        if im is None:  # 如果图像未缓存
            if fn.exists():  # 如果存在 npy 文件
                try:
                    im = np.load(fn)  # 加载 npy 文件
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}警告 ⚠️ 删除损坏的 *.npy 图像文件 {fn}，原因：{e}")
                    Path(fn).unlink(missing_ok=True)  # 删除损坏的文件
                    im = cv2.imread(f)  # 读取图像
            else:  # 如果是普通图像文件
                im = cv2.imread(f)  # 读取图像
            if im is None:
                raise FileNotFoundError(f"未找到图像 {f}")

            h0, w0 = im.shape[:2]  # 原始图像高宽
            if rect_mode:  # 如果是矩形模式
                r = self.imgsz / max(h0, w0)  # 计算缩放比例
                if r != 1:  # 如果高宽不相等
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))  # 计算新的高宽
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)  # 调整图像大小
            elif not (h0 == w0 == self.imgsz):  # 如果不是正方形且不是目标大小
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)  # 强制调整为正方形大小

            # 如果在训练时使用数据增强，则添加到缓冲区
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, 原始尺寸, 调整后尺寸
                self.buffer.append(i)  # 将索引添加到缓冲区
                if 1 < len(self.buffer) >= self.max_buffer_length:  # 防止缓冲区为空
                    j = self.buffer.pop(0)  # 移除最旧的索引
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None  # 清理 RAM 中的缓存

            return im, (h0, w0), im.shape[:2]  # 返回图像和尺寸信息

        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # 如果已缓存，返回缓存的图像和尺寸信息

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}  # 初始化标签字典
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # 初始化计数器
        desc = f"{self.prefix}扫描 {path.parent / path.stem}..."
        total = len(self.im_files)  # 数据集中图像的总数
        with ThreadPool(NUM_THREADS) as pool:  # 使用线程池
            results = pool.imap(
                func=verify_image_label,  # 验证每个图像的标签
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)  # 创建进度条
            for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f  # 更新缺失计数
                nf += nf_f  # 更新找到计数
                ne += ne_f  # 更新空计数
                nc += nc_f  # 更新损坏计数
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,  # 图像文件路径
                            "shape": shape,  # 图像形状
                            "cls": lb[:, 0:1],  # 类别
                            "bboxes": lb[:, 1:],  # 边界框
                            "normalized": True,  # 是否归一化
                            "bbox_format": "xywh",  # 边界框格式
                        }
                    )
                if msg:
                    msgs.append(msg)  # 收集消息
                pbar.desc = f"{desc} {nf} 图像, {nm + ne} 背景, {nc} 损坏"  # 更新进度条描述
            pbar.close()  # 关闭进度条

        if nf == 0:
            LOGGER.warning(f"{self.prefix}警告 ⚠️ 在 {path} 中未找到任何标签。 {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)  # 计算哈希值
        x["results"] = nf, nm, ne, nc, len(self.im_files)  # 更新结果信息
        x["msgs"] = msgs  # 收集警告消息
        save_dataset_cache_file(self.prefix, path, x)  # 保存缓存文件
        return x  # 返回标签信息

    def set_rectangle(self):
        """设置 YOLO 检测的边界框形状为矩形。"""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # 批次索引
        nb = bi[-1] + 1  # 批次数

        s = np.array([x.pop("shape") for x in self.labels])  # 获取图像原始形状
        ar = s[:, 0] / s[:, 1]  # 计算长宽比
        irect = ar.argsort()  # 按长宽比排序
        self.im_files = [self.im_files[i] for i in irect]  # 根据排序更新图像文件列表
        self.labels = [self.labels[i] for i in irect]  # 根据排序更新标签列表
        ar = ar[irect]  # 更新长宽比

        # 设置训练图像的形状
        shapes = [[1, 1]] * nb  # 初始化形状
        for i in range(nb):
            ari = ar[bi == i]  # 获取当前批次的长宽比
            mini, maxi = ari.min(), ari.max()  # 获取当前批次的最小和最大长宽比
            if maxi < 1:
                shapes[i] = [maxi, 1]  # 如果最大比值小于 1
            elif mini > 1:
                shapes[i] = [1, 1 / mini]  # 如果最小比值大于 1

        # 计算批次形状
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # 更新图像的批次索引

    def __getitem__(self, index):
        """返回指定索引的标签信息的转换结果。"""
        return self.transforms(self.get_image_and_label(index))  # 应用转换并返回结果

    def get_image_and_label(self, index):
        """获取并返回数据集中的图像和标签信息。"""
        label = deepcopy(self.labels[index])  # 深拷贝标签，避免修改原始数据
        label.pop("shape", None)  # 移除形状信息
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)  # 加载图像及其形状
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # 计算缩放比例
        if self.rect:  # 如果使用矩形训练
            label["rect_shape"] = self.batch_shapes[self.batch[index]]  # 获取当前图像的矩形形状
        return self.update_labels_info(label)  # 更新标签信息并返回

    def __len__(self):
        """返回数据集中标签的数量。"""
        return len(self.labels)  # 返回标签数量

    def update_labels_info(self, label):
        """
        自定义标签格式。

        注意：
            cls 现在不与边界框一起存在，分类和语义分割需要独立的 cls 标签。
            通过添加或删除字典键，也可以支持分类和语义分割。
        """
        bboxes = label.pop("bboxes")  # 获取并移除边界框信息
        bbox_format = label.pop("bbox_format")  # 获取并移除边界框格式
        normalized = label.pop("normalized")  # 获取并移除归一化标志

        # 注意：不对有向边界框进行重采样
        label["instances"] = Instances(bboxes,  bbox_format=bbox_format, normalized=normalized)  # 更新实例信息
        return label  # 返回更新后的标签

    def build_transforms(self, hyp=None):
        """构建图像转换函数。"""
        if self.augment:  # 如果使用数据增强
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0  # 设置马赛克比例
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0  # 设置混合比例
            transforms = v8_transforms(self, self.imgsz, hyp)  # 构建数据增强转换
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])  # 如果不使用增强，设置为固定大小
        transforms.append(
            Format(
                bbox_format="xywh",  # 设置边界框格式
                normalize=True,  # 是否归一化
                batch_idx=True,  # 是否返回批次索引
                bgr=hyp.bgr if self.augment else 0.0,  # 仅影响训练
            )
        )
        return transforms  # 返回构建好的转换

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)  # 获取标签文件路径
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")  # 缓存文件路径
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # 尝试加载缓存文件
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # 确保哈希值匹配
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # 如果加载失败，运行缓存操作

        # 显示缓存信息
        nf, nm, ne, nc, n = cache.pop("results")  # 找到的、缺失的、空的、损坏的图像数量
        if exists:
            d = f"扫描 {cache_path}... {nf} 图像, {nm + ne} 背景, {nc} 损坏"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # 显示结果
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # 打印警告信息

        # 读取缓存
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # 移除无用项
        labels = cache["labels"]  # 获取标签信息
        if not labels:
            LOGGER.warning(f"警告 ⚠️ 在 {cache_path} 中未找到任何标签，训练可能无法正常工作。")
        self.im_files = [lb["im_file"] for lb in labels]  # 更新图像文件路径
        return labels  # 返回标签信息

    def close_mosaic(self, hyp):
        """将马赛克、复制粘贴和混合选项设置为 0.0 并构建转换。"""
        hyp.mosaic = 0.0  # 设置马赛克比例为 0.0
        hyp.copy_paste = 0.0  # 保持与之前版本一致
        hyp.mixup = 0.0  # 保持与之前版本一致
        self.transforms = self.build_transforms(hyp)  # 更新转换

    @staticmethod
    def collate_fn(batch):
        """将数据样本合并成批次。"""
        new_batch = {}
        keys = batch[0].keys()  # 获取批次中的键
        values = list(zip(*[list(b.values()) for b in batch]))  # 将每个样本的值按键合并
        for i, k in enumerate(keys):
            value = values[i]  # 获取当前键的值
            if k == "img":
                value = torch.stack(value, 0)  # 堆叠图像
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)  # 合并掩码、关键点、边界框、类别、分段等
            new_batch[k] = value  # 更新合并后的批次

        new_batch["batch_idx"] = list(new_batch["batch_idx"])  # 获取批次索引
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # 为 build_targets() 添加目标图像索引
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)  # 合并索引
        return new_batch  # 返回合并后的批次
