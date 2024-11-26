# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils import LOGGER, yaml_load
from ultralytics.utils.checks import check_yaml


def default_class_names(data=None):
    """Applies default class names to an input YAML file or returns numerical class names."""
    if data:
        try:
            return yaml_load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # return default if above errors


class AutoBackend(nn.Module):
    """
    Handles dynamic backend selection for running inference using Ultralytics YOLO models.

    The AutoBackend class is designed to provide an abstraction layer for various inference engines. It supports a wide
    range of formats, each with specific naming conventions as outlined below:

        Supported Formats and Naming Conventions:
            | Format                | File Suffix       |
            |-----------------------|-------------------|
            | PyTorch               | *.pt              |

    This class offers dynamic backend switching capabilities based on the input model format, making it easier to deploy
    models across various platforms.
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
        Initialize the AutoBackend for inference.

        Args:
            weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        nn_module = isinstance(weights, torch.nn.Module)
        pt = (w.split('.')[1]=='pt')
        fp16 &= True
        model, metadata, task = None, None, None

        # Set device
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA

        # In-memory PyTorch model
        if nn_module:
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
            pt = True

        # PyTorch
        elif pt:
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(
                weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
            )
            if hasattr(model, "kpt_shape"):
                kpt_shape = model.kpt_shape  # pose-only
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        # Load external metadata YAML
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
            LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")

        # Check names
        if "names" not in locals():  # names missing
            names = default_class_names(data)

        # Disable gradients
        if pt:
            for p in model.parameters():
                p.requires_grad = False

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, embed=None):
        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
            augment (bool): whether to perform data augmentation during inference, defaults to False
            visualize (bool): whether to visualize the output predictions, defaults to False
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (tuple): Tuple containing the raw output tensor, and processed output for visualization (if visualize=True)
        """
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        y = self.model(im, augment=augment, visualize=visualize, embed=embed)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # segments and names not defined
                ip, ib = (0, 1) if len(y[0].shape) == 4 else (1, 0)  # index of protos, boxes
                nc = y[ib].shape[1] - y[ip].shape[3] - 4  # y = (1, 160, 160, 32), (1, 116, 8400)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Convert a numpy array to a tensor."""
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Warm up the model by running one forward pass with a dummy input."""
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

        warmup_types = self.pt, self.nn_module
        if any(warmup_types) and (self.device.type != "cpu"):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):
                self.forward(im)  # warmup

