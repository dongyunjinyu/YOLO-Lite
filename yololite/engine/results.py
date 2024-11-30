# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from yololite.utils import LOGGER, SimpleClass, ops
from yololite.utils.checks import check_requirements
from yololite.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):
    def __init__(self, data, orig_shape) -> None:
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):
    """
    A class for storing and manipulating inference results.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Object containing detection bounding boxes.
        speed (Dict[str, float | None]): Dictionary of preprocess, inference, and postprocess speeds.
        names (Dict[int, str]): Dictionary mapping class IDs to class names.
        path (str): Path to the image file.
        _keys (Tuple[str, ...]): Tuple of attribute names for internal use.

    Methods:
        update: Updates object attributes with new detection results.
        cpu: Returns a copy of the Results object with all tensors on CPU memory.
        numpy: Returns a copy of the Results object with all tensors as numpy arrays.
        cuda: Returns a copy of the Results object with all tensors on GPU memory.
        to: Returns a copy of the Results object with tensors on a specified device and dtype.
        new: Returns a new Results object with the same image, path, and names.
        plot: Plots detection results on an input image, returning an annotated image.
        show: Shows annotated results on screen.
        save: Saves annotated results to file.
        verbose: Returns a log string for each task, detailing detections and classifications.
        save_txt: Saves detection results to a text file.
        save_crop: Saves cropped detection images.
        tojson: Converts detection results to JSON format.
    """

    def __init__(
        self, orig_img, path, names, boxes=None, speed=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (Dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            speed (Dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = ("boxes",)

    def __getitem__(self, idx):
        return self._apply("__getitem__", idx)

    def __len__(self):

        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None):
        if boxes is not None:
            self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply("cpu")

    def numpy(self):
        return self._apply("numpy")

    def cuda(self):
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        return self._apply("to", *args, **kwargs)

    def new(self):
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        labels=True,
        boxes=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
    ):
        """
        Plots detection results on an input RGB image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (bool): Specify the color mode, e.g., 'instance' or 'class'. Default to 'class'.

        Returns:
            (np.ndarray): Annotated image as a numpy array.
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil,  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
                box = d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                    rotated=False,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """
        Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            ...     result.show()  # Display all results
        """
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """
        Saves annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | Path | None): The filename to save the annotated image. If None, a default filename
                is generated based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # Or with custom plot arguments
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """
        Returns a log string for each task in the results, detailing detection and classification outcomes.

        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.

        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the
                number of detections per class. For classification tasks, it includes the top 5 class probabilities.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - If there are no detections, the method returns "(no detections), " for detection tasks.
            - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
            - The returned string is comma-separated and ends with a comma and a space.
        """
        log_string = ""
        boxes = self.boxes
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from yololite import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class confidence x_center y_center width height`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Saves cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images. Default is Path("im.jpg").

        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """
        Converts inference results to a summarized dictionary with optional normalization for box coordinates.

        This method creates a list of detection dictionaries, each containing information about a single
        detection or classification result. For classification tasks, it returns the top class and its
        confidence. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.

        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.

        Returns:
            (List[Dict]): A list of dictionaries, each containing summarized information for a single
                detection or classification result. The structure of each dictionary varies based on the
                task type (classification or detection) and available information (boxes, masks, keypoints).

        Examples:
            >>> results = model("image.jpg")
            >>> summary = results[0].summary()
            >>> print(summary)
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def to_df(self, normalize=False, decimals=5):
        """
        Converts detection results to a Pandas Dataframe.

        This method converts the detection results into Pandas Dataframe format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.

        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.

        Returns:
            (DataFrame): A Pandas Dataframe containing all the information in results in an organized way.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> df_result = results[0].to_df()
            >>> print(df_result)
        """
        import pandas as pd

        return pd.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5, *args, **kwargs):
        """
        Converts detection results to a CSV format.

        This method serializes the detection results into a CSV format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.

        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.
            *args (Any): Variable length argument list to be passed to pandas.DataFrame.to_csv().
            **kwargs (Any): Arbitrary keyword arguments to be passed to pandas.DataFrame.to_csv().


        Returns:
            (str): CSV containing all the information in results in an organized way.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> csv_result = results[0].to_csv()
            >>> print(csv_result)
        """
        return self.to_df(normalize=normalize, decimals=decimals).to_csv(*args, **kwargs)

    def to_xml(self, normalize=False, decimals=5, *args, **kwargs):
        """
        Converts detection results to XML format.

        This method serializes the detection results into an XML format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.

        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.
            *args (Any): Variable length argument list to be passed to pandas.DataFrame.to_xml().
            **kwargs (Any): Arbitrary keyword arguments to be passed to pandas.DataFrame.to_xml().

        Returns:
            (str): An XML string containing all the information in results in an organized way.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> xml_result = results[0].to_xml()
            >>> print(xml_result)
        """
        check_requirements("lxml")
        df = self.to_df(normalize=normalize, decimals=decimals)
        return '<?xml version="1.0" encoding="utf-8"?>\n<root></root>' if df.empty else df.to_xml(*args, **kwargs)

    def tojson(self, normalize=False, decimals=5):
        """Deprecated version of to_json()."""
        LOGGER.warning("WARNING ⚠️ 'result.tojson()' is deprecated, replace with 'result.to_json()'.")
        return self.to_json(normalize, decimals)

    def to_json(self, normalize=False, decimals=5):
        """
        Converts detection results to JSON format.

        This method serializes the detection results into a JSON-compatible format. It includes information
        about detected objects such as bounding boxes, class names, confidence scores, and optionally
        segmentation masks and keypoints.

        Args:
            normalize (bool): Whether to normalize the bounding box coordinates by the image dimensions.
                If True, coordinates will be returned as float values between 0 and 1. Defaults to False.
            decimals (int): Number of decimal places to round the output values to. Defaults to 5.

        Returns:
            (str): A JSON string containing the serialized detection results.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> json_result = results[0].to_json()
            >>> print(json_result)

        Notes:
            - For classification tasks, the JSON will contain class probabilities instead of bounding boxes.
            - For object detection tasks, the JSON will include bounding box coordinates, class names, and
              confidence scores.
            - If available, segmentation masks and keypoints will also be included in the JSON output.
            - The method uses the `summary` method internally to generate the data structure before
              converting it to JSON.
        """
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes.

    This class provides functionality for handling detection boxes, including their coordinates, confidence scores,
    class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipulation
    and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray): Tracking IDs for each box (if available).
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu(): Returns a copy of the object with all tensors on CPU memory.
        numpy(): Returns a copy of the object with all tensors as numpy arrays.
        cuda(): Returns a copy of the object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates,
        confidence scores, class identifiers, and optional tracking IDs. It supports multiple formats
        for box coordinates, including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape
                (num_boxes, 6) or (num_boxes, 7). Columns should contain
                [x1, y1, x2, y2, confidence, class, (optional) track_id].
            orig_shape (Tuple[int, int]): The original image shape as (height, width). Used for normalization.

        Attributes:
            data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
            orig_shape (Tuple[int, int]): The original image size, used for normalization.
            is_track (bool): Indicates whether tracking IDs are included in the box data.

        Examples:
            >>> import torch
            >>> boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0]])
            >>> orig_shape = (480, 640)
            >>> detection_boxes = Boxes(boxes, orig_shape)
            >>> print(detection_boxes.xyxy)
            tensor([[100.,  50., 150., 100.]])
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box
                coordinates in [x1, y1, x2, y2] format, where n is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> xyxy = boxes.xyxy
            >>> print(xyxy)
        """
        return self.data[:, :4]

    @property
    def conf(self):
        """
        Returns the confidence scores for each detection box.

        Returns:
            (torch.Tensor | numpy.ndarray): A 1D tensor or array containing confidence scores for each detection,
                with shape (N,) where N is the number of detections.

        Examples:
            >>> boxes = Boxes(torch.tensor([[10, 20, 30, 40, 0.9, 0]]), orig_shape=(100, 100))
            >>> conf_scores = boxes.conf
            >>> print(conf_scores)
            tensor([0.9000])
        """
        return self.data[:, -2]

    @property
    def cls(self):
        """
        Returns the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the class IDs for each detection box.
                The shape is (N,), where N is the number of boxes.

        Examples:
            >>> results = model("image.jpg")
            >>> boxes = results[0].boxes
            >>> class_ids = boxes.cls
            >>> print(class_ids)  # tensor([0., 2., 1.])
        """
        return self.data[:, -1]

    @property
    def id(self):
        """
        Returns the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | None): A tensor containing tracking IDs for each box if tracking is enabled,
                otherwise None. Shape is (N,) where N is the number of boxes.

        Examples:
            >>> results = model.track("path/to/video.mp4")
            >>> for result in results:
            ...     boxes = result.boxes
            ...     if boxes.is_track:
            ...         track_ids = boxes.id
            ...         print(f"Tracking IDs: {track_ids}")
            ...     else:
            ...         print("Tracking is not enabled for these boxes.")

        Notes:
            - This property is only available when tracking is enabled (i.e., when `is_track` is True).
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """
        Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | numpy.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center are the coordinates of
                the center point of the bounding box, width, height are the dimensions of the bounding box and the
                shape of the returned tensor is (N, 4), where N is the number of boxes.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100], [200, 150, 300, 250]]), orig_shape=(480, 640))
            >>> xywh = boxes.xywh
            >>> print(xywh)
            tensor([[100.0000,  50.0000,  50.0000,  50.0000],
                    [200.0000, 150.0000, 100.0000, 100.0000]])
        """
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """
        Returns normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format,
        normalized to the range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is
                the number of boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 300, 400, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xyxyn
            >>> print(normalized)
            tensor([[0.1562, 0.1042, 0.4688, 0.8333]])
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """
        Returns normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format
        [x_center, y_center, width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | numpy.ndarray): Normalized bounding boxes with shape (N, 4), where N is the
                number of boxes. Each row contains [x_center, y_center, width, height] values normalized
                to [0, 1] based on the original image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh

