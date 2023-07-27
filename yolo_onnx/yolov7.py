import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from utils import nms, xywh2xyxy


class YOLOv7:
    """Class for running YOLOv5-7 inference on ONNX Runtime."""

    def __init__(
        self,
        path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        official_nms: bool = False,
    ) -> None:
        """
        Initializes the YOLOv7 model (Works also with YOLOv5 and probably with YOLOv6).
        Args:
            path (str): The path to the ONNX model.
            conf_thres (float, optional): The confidence threshold for predictions. Defaults to 0.25.
            iou_thres (float, optional): The intersection over union (IoU) threshold for non-maximum suppression. Defaults to 0.5.
            official_nms (bool, optional): If true, uses official version of non-maximum suppression. Defaults to False.
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.official_nms = official_nms

        # Initialize model
        self._initialize_model(path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs object detection on an image.
        Args:
            image (np.ndarray): The input image.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs of detected objects.
        """
        return self.predict(image)

    def _initialize_model(self, path: str) -> None:
        """
        Initializes the ONNX model.
        Args:
            path (str): The path to the ONNX model.
        """
        self.session = onnxruntime.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Get model info
        self._get_input_details()
        self._get_output_details()

        self.has_postprocess = "score" in self.output_names or self.official_nms

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects objects in an image.
        Args:
            image (np.ndarray): The input image.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs of detected objects.
        """
        input_tensor = self._prepare_input(image)

        # Perform inference on the image
        outputs = self._inference(input_tensor)

        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self._parse_processed_output(
                outputs
            )

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self._process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """
        Prepares image for inference.
        Args:
            image (np.ndarray): The input image.
        Returns:
            np.ndarray: The prepared image.
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def _inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Performs inference on the input tensor.
        Args:
            input_tensor (np.ndarray): The input tensor.
        Returns:
            List[np.ndarray]: The output from the model.
        """
        start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def _process_output(
        self, output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes the output from the model.
        Args:
            output (np.ndarray): The raw output from the model.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs of detected objects.
        """
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self._extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def _parse_processed_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses the processed output from the model.
        Args:
            outputs (List[np.ndarray]): The processed output from the model.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs of detected objects.
        """
        # Pinto's postprocessing is different from the official nms version
        if self.official_nms:
            scores = outputs[0][:, -1]
            predictions = outputs[0][:, [0, 5, 1, 2, 3, 4]]
        else:
            scores = np.squeeze(outputs[0], axis=1)
            predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        if not self.official_nms:
            boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self._rescale_boxes(boxes)

        return boxes, scores, class_ids

    def _extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """
        Extracts bounding boxes from predictions.
        Args:
            predictions (np.ndarray): The predictions from the model.
        Returns:
            np.ndarray: The extracted bounding boxes.
        """
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self._rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def _rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Rescales the bounding boxes to the original image dimensions.
        Args:
            boxes (np.ndarray): The bounding boxes.
        Returns:
            np.ndarray: The rescaled bounding boxes.
        """
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def _get_input_details(self) -> None:
        """Fetches details about the model inputs."""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self) -> None:
        """Fetches details about the model outputs."""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
