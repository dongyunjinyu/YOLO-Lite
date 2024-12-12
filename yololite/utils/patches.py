# YOLO-Lite 🚀
"""猴子补丁，用于更新/扩展现有函数的功能。"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch

# OpenCV 多语言友好的函数 ------------------------------------------------------------------------------
_imshow = cv2.imshow  # 复制以避免递归错误


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    从文件读取图像。

    参数：
        filename (str): 要读取的文件路径。
        flags (int, optional): 可以取值为 cv2.IMREAD_* 的标志。默认为 cv2.IMREAD_COLOR。

    返回：
        (np.ndarray): 读取的图像。
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def imwrite(filename: str, img: np.ndarray, params=None):
    """
    将图像写入文件。

    参数：
        filename (str): 要写入的文件路径。
        img (np.ndarray): 要写入的图像。
        params (list of ints, optional): 额外参数。请参见 OpenCV 文档。

    返回：
        (bool): 如果文件写入成功则返回 True，否则返回 False。
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray):
    """
    在指定窗口中显示图像。

    参数：
        winname (str): 窗口名称。
        mat (np.ndarray): 要显示的图像。
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch 函数 ----------------------------------------------------------------------------------------------------
_torch_load = torch.load  # 复制以避免递归错误
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """
    使用更新的参数加载 PyTorch 模型，以避免警告。

    此函数包装 torch.load，并为 PyTorch 1.13.0+ 添加 'weights_only' 参数以防止警告。

    参数：
        *args (Any): 可变长度参数列表，传递给 torch.load。
        **kwargs (Any): 任意关键字参数，传递给 torch.load。

    返回：
        (Any): 加载的 PyTorch 对象。

    注意：
        对于 PyTorch 2.0 及以上版本，如果未提供该参数，此函数会自动将 'weights_only' 设置为 False，
        以避免弃用警告。
    """
    from yololite.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return _torch_load(*args, **kwargs)


def torch_save(*args, **kwargs):
    """
    可选择使用 dill 序列化 lambda 函数（在 pickle 无法时），
    通过 3 次重试和指数退避增加鲁棒性，以防保存失败。

    参数：
        *args (tuple): 传递给 torch.save 的位置参数。
        **kwargs (Any): 传递给 torch.save 的关键字参数。
    """
    for i in range(4):  # 3 次重试
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:  # 无法保存，可能在等待设备刷新或防病毒扫描
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # 指数退避：0.5s, 1.0s, 2.0s