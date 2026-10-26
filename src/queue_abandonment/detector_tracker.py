"""Ultralytics person detector with configurable built-in multi-object tracker."""
from pathlib import Path
from importlib.util import find_spec
from typing import Any

class DetectorTracker:
    """Wrap Ultralytics tracking and keep only the person class."""
    def __init__(self, config: dict[str, Any]):
        """Load requested weights, attempting the documented model download."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is missing. Install pinned dependencies with `pip install -r requirements.txt`.") from exc
        if find_spec("lap") is None:
            raise RuntimeError("Ultralytics tracking dependency lapx is missing. Install project dependencies with `pip install -r requirements.txt`.")
        weights = Path(config["model_name"])
        try:
            if not weights.is_file():
                print(f"Model weights '{weights}' are missing; asking Ultralytics to download them now if internet is available.")
            self.model = YOLO(str(weights))
        except Exception as exc:
            raise RuntimeError(f"Could not load model weights '{weights}'. Install internet access and rerun to download them, or place the weights file at that path. Details: {exc}") from exc
        self.confidence = float(config["confidence_threshold"])
        self.image_size = int(config["inference_size"])
        self.tracker = "bytetrack.yaml" if config["tracker"] == "bytetrack" else "botsort.yaml"
        self.device = config["device"]

    def track(self, frame: Any) -> list[dict[str, Any]]:
        """Return person boxes, IDs, confidence, and bottom-center foot point."""
        results = self.model.track(frame, persist=True, tracker=self.tracker, classes=[0], conf=self.confidence, imgsz=self.image_size, device=self.device, verbose=False)
        output = []
        if not results or results[0].boxes is None:
            return output
        boxes = results[0].boxes
        for index, raw in enumerate(boxes.xyxy.cpu().tolist()):
            track_id = int(boxes.id[index].item()) if boxes.id is not None else index
            output.append({"track_id": track_id, "box": [int(value) for value in raw], "foot": ((raw[0] + raw[2]) / 2, raw[3]), "confidence": float(boxes.conf[index].item())})
        return output
