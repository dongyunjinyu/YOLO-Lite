# YOLO-Lite 🚀

import platform
import re
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
from yololite.engine.results import Results
from yololite.cfg import get_cfg, get_save_dir
from yololite.data import load_inference_source
from yololite.data.augment import LetterBox
from yololite.nn.autobackend import AutoBackend
from yololite.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, colorstr, ops
from yololite.utils.checks import check_imgsz, check_imshow
from yololite.utils.files import increment_path
from yololite.utils.torch_utils import select_device, smart_inference_mode


class DetectionPredictor:
    """
    属性：
        args (SimpleNamespace): 预测器的配置。
        save_dir (Path): 结果保存目录。
        done_warmup (bool): 预测器是否完成设置。
        model (nn.Module): 用于预测的模型。
        data (dict): 数据配置。
        device (torch.device): 用于预测的设备。
        dataset (Dataset): 用于预测的数据集。
        vid_writer (dict): 字典 {save_path: video_writer, ...} 用于保存视频输出的写入器。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        初始化 BasePredictor 类。

        参数：
            cfg (str, optional): 配置文件的路径。默认为 DEFAULT_CFG。
            overrides (dict, optional): 配置覆盖。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认 conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # 可用的设置完成后
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # 字典 {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.txt_path = None
        self._lock = threading.Lock()  # 用于自动线程安全推理

    def preprocess(self, im):
        """
        在推理之前准备输入图像。

        参数：
            im (torch.Tensor | List(np.ndarray)): BCHW 对于张量，[(HWC) x B] 对于列表。
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR 到 RGB，BHWC 到 BCHW，(n, 3, h, w)
            im = np.ascontiguousarray(im)  # 连续
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 转为 fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 转为 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """对给定图像运行推理，使用指定的模型和参数。"""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        在推理之前对输入图像进行预处理。

        参数：
            im (List(np.ndarray)): (N, 3, h, w) 对于张量，[(h, w, 3) x N] 对于列表。

        返回：
            (list): 经过转换的图像列表。
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """对图像或流执行推理。"""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # 将结果合并为一个列表

    def setup_source(self, source):
        """设置源和推理模式。"""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # 检查图像大小
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """对摄像头 feed 进行实时推理并将结果保存到文件。"""
        if self.args.verbose:
            LOGGER.info("")

        # 设置模型
        if not self.model:
            self.setup_model(model)

        with self._lock:  # 线程安全推理
            # 每次调用 predict 时设置源
            self.setup_source(source if source is not None else self.args.source)

            # 检查 save_dir/标签文件是否存在
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # 预热模型
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            for self.batch in self.dataset:
                paths, im0s, s = self.batch

                # 预处理
                with profilers[0]:
                    im = self.preprocess(im0s)

                # 推理
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # 生成嵌入张量
                        continue

                # 后处理
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)

                # 可视化，保存，写入结果
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # 打印批处理结果
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                yield from self.results

        # 释放资源
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # 打印最终结果
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # 每张图像的速度
            LOGGER.info(
                f"速度: %.1fms 预处理, %.1fms 推理, %.1fms 后处理每张图像，形状 "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # 标签数量
            s = f"\n{nl} 标签{'s' * (nl > 1)} 保存到 {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"结果保存到 {colorstr('bold', self.save_dir)}{s}")

    def setup_model(self, model, verbose=True):
        """使用给定参数初始化 YOLO 模型并将其置于评估模式。"""
        self.model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # 更新设备
        self.args.half = self.model.fp16  # 更新半精度
        self.model.eval()

    def write_results(self, i, p, im, s):
        """将推理结果写入文件或目录。"""
        string = ""  # 打印字符串
        if len(im.shape) == 3:
            im = im[None]  # 扩展批处理维度
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 如果帧未确定则为 0

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # 用于其他位置
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 将预测结果添加到图像
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
            )

        # 保存结果
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(str(self.save_dir / p.name), frame)

        return string

    def save_predicted_images(self, save_path="", frame=0):
        """将视频预测保存为 mp4 在指定路径。"""
        im = self.plotted_img

        # 保存视频和流
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if save_path not in self.vid_writer:  # 新视频
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # 整数要求，浮点数在 MP4 编解码时产生错误
                    frameSize=(im.shape[1], im.shape[0]),  # (宽度, 高度)
                )

            # 保存视频
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # 保存图像
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # 保存为 JPG，以获得最佳支持

    def show(self, p=""):
        """使用 OpenCV imshow 函数在窗口中显示图像。"""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (宽度, 高度)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 毫秒
