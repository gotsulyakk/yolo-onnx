import time
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime
from utils import nms, xywh2xyxy


class YOLOv8:
    def __init__(
        self, path: str, conf_thres: float = 0.25, iou_thres: float = 0.45
    ) -> None:
        """Initializes the YOLOv8 model with provided parameters.

        Args:
        path (str): Path to the ONNX model file.
        conf_thres (float, optional): Confidence threshold. Defaults to 0.5.
        iou_thres (float, optional): Intersection over Union threshold. Defaults to 0.45.
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.session = None
        self.input_names = []
        self.output_names = []
        self.input_shape = []
        self.input_height = 0
        self.input_width = 0
        self.img_height = 0
        self.img_width = 0

        # Initialize model
        self._initialize_model(path)

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """Makes the object callable. It's a Python special method.

        Args:
        image (np.ndarray): Input image.

        Returns:
        Tuple[List[np.ndarray], List[float], List[int]]: Returns boxes, scores, and class IDs.
        """
        return self.predict(image)

    def _initialize_model(self, path: str) -> None:
        """Initializes the model.

        Args:
        path (str): Path to the model file.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            # Get model info
            self._get_input_details()
            self._get_output_details()
        except Exception as e:
            print(f"Failed to load the model. Error: {e}")
            raise e

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """Performs prediction on the given image.

        Args:
        image (np.ndarray): Input image.

        Returns:
        Tuple[List[np.ndarray], List[float], List[int]]: Returns boxes, scores, and class IDs.
        """
        try:
            # Prepare input image
            input_tensor = self._prepare_input(image)

            # Perform _inference on the image
            outputs = self._inference(input_tensor)

            return self._process_output(outputs)  # boxes, scores, class_ids
        except Exception as e:
            print(f"Failed to predict. Error: {e}")
            raise e

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepares the input for the model.

        Args:
        image (np.ndarray): The input image.

        Returns:
        np.ndarray: The processed image.
        """
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image to match the input size required by the model
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Normalize pixel values (0-255 -> 0-1), and transpose the dimensions to meet the model's input shape
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)

        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def _inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Runs the inference on the given input tensor.

        Args:
        input_tensor (np.ndarray): The input tensor.

        Returns:
        List[np.ndarray]: The output from the model.
        """
        start = time.perf_counter()
        outputs = self.session.run(
            [self.output_names[0]], {self.input_names[0]: input_tensor}
        )
        end = time.perf_counter()

        print(f"Inference time: {(end - start) * 1000}ms")

        return outputs

    def _process_output(
        self, output: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """Processes the output from the model.

        Args:
        output (List[np.ndarray]): The output from the model.

        Returns:
        Tuple[List[np.ndarray], List[float], List[int]]: Bounding boxes, scores, and class IDs.
        """
        predictions = np.squeeze(output[0]).T

        # Filter out predictions with confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Extract and format bounding boxes
        boxes = self._extract_boxes(predictions)

        # Apply non-maxima suppression to filter out weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def _extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """Extracts the bounding boxes from the predictions and scales them to the original image dimensions.

        Args:
        predictions (np.ndarray): The output predictions from the model.

        Returns:
        np.ndarray: The bounding boxes.
        """
        # Extract bounding boxes from predictions
        boxes = predictions[:, :4]

        # Scale and format bounding boxes
        boxes = self._rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)

        return boxes

    def _rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Rescales the bounding boxes to match the original image dimensions.

        Args:
        boxes (np.ndarray): The bounding boxes.

        Returns:
        np.ndarray: The rescaled bounding boxes.
        """
        # Rescale bounding boxes to original image dimensions
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes.astype(np.int32)

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
