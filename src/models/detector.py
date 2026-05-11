"""
YOLOv8 object detector wrapper.

Takes a frame (numpy array or PIL Image) and returns detected objects
with bounding boxes, class names, and confidence scores.

Why YOLOv8 runs on top of the LSTM, not instead of it:
    LSTM  -> answers "is this clip anomalous?" (temporal reasoning)
    YOLO  -> answers "what is in this frame?"  (spatial reasoning)

    Combined: "This clip is anomalous [LSTM] because it contains
    a person and a weapon in close proximity [YOLO]."
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO

from config import DEVICE


# Classes most relevant to surveillance anomaly detection
# YOLOv8 has 80 COCO classes — we only surface the meaningful ones
SURVEILLANCE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "knife", "scissors",           # weapons
    "fire hydrant", "stop sign",   # scene context
    "backpack", "handbag", "suitcase",  # theft context
    "bottle", "chair", "couch",    # environmental context
}


@dataclass
class Detection:
    class_name:  str
    confidence:  float
    bbox:        tuple   # (x1, y1, x2, y2) in pixel coords
    bbox_norm:   tuple   # (x1, y1, x2, y2) normalized to [0,1]


class YOLOv8Detector:
    """
    Wraps YOLOv8 for per-frame object detection.

    Input  : PIL Image or numpy array [H, W, 3]
    Output : List[Detection]
    """

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.3):
        """
        Args:
            model_size     : 'n' (nano, fastest) | 's' | 'm' | 'l' | 'x'
            conf_threshold : minimum confidence to report a detection
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"Loading YOLOv8{model_size}...")
        self.model = YOLO(model_name)
        self.conf  = conf_threshold
        # Move to GPU if available — YOLOv8 handles this internally
        self.device = "0" if DEVICE == "cuda" else "cpu"
        print(f"YOLOv8{model_size} ready on device: {self.device}")

    def detect(self, frame) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame : PIL Image or numpy array [H, W, 3] (RGB)
        Returns:
            List of Detection objects, sorted by confidence descending
        """
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
        else:
            frame_np = frame

        H, W = frame_np.shape[:2]

        results = self.model(
            frame_np,
            conf=self.conf,
            device=self.device,
            verbose=False,   # suppress per-frame stdout
        )[0]

        detections = []
        for box in results.boxes:
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(Detection(
                class_name = class_name,
                confidence = confidence,
                bbox       = (int(x1), int(y1), int(x2), int(y2)),
                bbox_norm  = (x1/W, y1/H, x2/W, y2/H),
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_batch(self, frames: list) -> List[List[Detection]]:
        """
        Run detection on a list of frames — more efficient than detect() in a loop.

        Args:
            frames : list of PIL Images or numpy arrays
        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []

        frame_arrays = []
        shapes = []
        for f in frames:
            arr = np.array(f) if isinstance(f, Image.Image) else f
            frame_arrays.append(arr)
            shapes.append(arr.shape[:2])

        results_list = self.model(
            frame_arrays,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for results, (H, W) in zip(results_list, shapes):
            detections = []
            for box in results.boxes:
                class_id   = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(Detection(
                    class_name = class_name,
                    confidence = confidence,
                    bbox       = (int(x1), int(y1), int(x2), int(y2)),
                    bbox_norm  = (x1/W, y1/H, x2/W, y2/H),
                ))

            detections.sort(key=lambda d: d.confidence, reverse=True)
            all_detections.append(detections)

        return all_detections

    def summarize(self, detections: List[Detection]) -> str:
        """
        Convert a list of detections into a human-readable explanation string.

        Example output:
            "Detected: person (0.94), person (0.87), knife (0.72)"
        """
        if not detections:
            return "No objects detected"

        parts = [f"{d.class_name} ({d.confidence:.2f})" for d in detections[:5]]
        return "Detected: " + ", ".join(parts)


def get_detector(model_size: str = "n", conf_threshold: float = 0.3) -> YOLOv8Detector:
    return YOLOv8Detector(model_size=model_size, conf_threshold=conf_threshold)


if __name__ == "__main__":
    import urllib.request
    import tempfile, os

    detector = get_detector(model_size="n")

    # Test with a blank image
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy[200:280, 280:360] = [200, 150, 100]   # rough person-coloured blob

    detections = detector.detect(dummy)
    print(f"\nDetections on dummy frame: {len(detections)}")
    print(detector.summarize(detections))

    # Test batch
    batch = [dummy, dummy]
    batch_results = detector.detect_batch(batch)
    print(f"\nBatch results: {len(batch_results)} frames processed")
    print("YOLOv8Detector OK")
