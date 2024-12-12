# YOLO-Lite 🚀

import glob
import os
from pathlib import Path


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    递增文件或目录路径，即将 runs/exp 变为 runs/exp{sep}2，runs/exp{sep}3，等等。

    如果路径存在且 `exist_ok` 不为 True，则路径将通过在路径末尾附加一个数字和 `sep` 来递增。
    如果路径是文件，则文件扩展名将被保留。如果路径是目录，则数字将直接附加到路径的末尾。
    如果 `mkdir` 设置为 True，则如果路径不存在，将创建该路径作为目录。

    参数：
        path (str | pathlib.Path)：要递增的路径。
        exist_ok (bool)：如果为 True，则路径不会递增，并将原样返回。
        sep (str)：路径与递增数字之间使用的分隔符。
        mkdir (bool)：如果目录不存在，则创建一个目录。
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def get_latest_run(search_dir="."):
    """返回指定目录中最新的 'last.pt' 文件的路径，以便恢复训练。"""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""
