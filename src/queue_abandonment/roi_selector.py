"""Interactive queue and service polygon selector."""
import argparse
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml

REMINDER = "Draw ROIs on the floor where feet will be."

def _polygons(value: list[Any]) -> list[list[list[int]]]:
    """Accept a single polygon or a list of separate staff-area polygons."""
    if not value:
        return []
    if len(value) >= 3 and isinstance(value[0], (list, tuple)) and len(value[0]) == 2 and all(isinstance(v, (int, float)) for v in value[0]):
        return [value]
    return list(value)

def save_roi_preview(video: str, config_path: str, output_path: str = "output/roi_preview.png") -> Path:
    """Draw configured queue and service polygons over the video's first frame."""
    capture = cv2.VideoCapture(video)
    ok, frame = capture.read(); capture.release()
    if not ok:
        raise ValueError(f"Cannot read first frame from {video}.")
    config: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    preview = frame.copy()
    for roi in config["queue_rois"]:
        for key, color, label in (("queue", (230, 210, 40), "QUEUE"), ("service", (220, 70, 220), "SERVICE"), ("staff_zone", (140, 140, 140), "STAFF AREA")):
            if key == "staff_zone" and not roi.get(key):
                continue
            polygons = [np.asarray(polygon, dtype=np.int32) for polygon in _polygons(roi[key])]
            cv2.polylines(preview, polygons, True, color, 2)
            x, y = polygons[0][0]
            cv2.putText(preview, label, (int(x), max(22, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        anchor = roi.get("service_anchor")
        if anchor:
            cv2.drawMarker(preview, tuple(map(int, anchor)), (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
            cv2.putText(preview, "SERVICE ANCHOR", (int(anchor[0]) + 10, int(anchor[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), preview):
        raise ValueError(f"Could not save ROI preview to {target}.")
    return target

def select_rois(video: str, config_path: str) -> Path:
    """Let the user draw queue and service ROIs and save a preview image."""
    capture = cv2.VideoCapture(video)
    ok, frame = capture.read(); capture.release()
    if not ok:
        raise ValueError(f"Cannot read first frame from {video}.")
    config: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    height, width = frame.shape[:2]
    selected: dict[str, list[list[int]]] = {}
    selected_anchor: list[int] = []
    for key, label, color in (("queue", "Queue ROI", (230, 210, 40)), ("service", "Service ROI", (220, 70, 220)),
                              ("staff_zone", "Staff area", (140, 140, 140))):
        points: list[tuple[int, int]] = []
        window = f"{label}: click >=3 points, press Enter to save"
        def click(event: int, x: int, y: int, flags: int, params: Any) -> None:
            """Append clicked vertices to the current polygon."""
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, click)
        while True:
            display = frame.copy()
            for old_key, old_points in selected.items():
                old_color = {"queue": (230, 210, 40), "service": (220, 70, 220), "staff_zone": (140, 140, 140)}[old_key]
                cv2.polylines(display, [np.asarray(old_points, dtype=np.int32)], True, old_color, 2)
            if len(points) >= 2:
                cv2.polylines(display, [np.asarray(points, dtype=np.int32)], len(points) >= 3, color, 2)
            for point in points:
                cv2.circle(display, point, 5, color, -1)
            cv2.putText(display, REMINDER, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(display, REMINDER, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)
            keycode = cv2.waitKey(20) & 0xFF
            if keycode in (10, 13) and len(points) >= 3:
                break
            if keycode == 27:
                cv2.destroyAllWindows()
                raise ValueError("ROI selection cancelled.")
        cv2.destroyWindow(window)
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError(f"{label} contains points outside the {width}x{height} frame.")
        selected[key] = [[int(x), int(y)] for x, y in points]
    window = "Service anchor: click one customer standing point, press Enter"
    def anchor_click(event: int, x: int, y: int, flags: int, params: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_anchor[:] = [x, y]
    cv2.namedWindow(window); cv2.setMouseCallback(window, anchor_click)
    while True:
        display = frame.copy()
        for key, points in selected.items():
            color = {"queue": (230, 210, 40), "service": (220, 70, 220), "staff_zone": (140, 140, 140)}[key]
            cv2.polylines(display, [np.asarray(points, dtype=np.int32)], True, color, 2)
        if selected_anchor:
            cv2.drawMarker(display, tuple(selected_anchor), (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
        cv2.putText(display, "Click one service standing point, press Enter", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(display, "Click one service standing point, press Enter", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, display); keycode = cv2.waitKey(20) & 0xFF
        if keycode in (10, 13) and selected_anchor: break
        if keycode == 27:
            cv2.destroyAllWindows(); raise ValueError("Service-anchor selection cancelled.")
    cv2.destroyWindow(window)
    for roi in config["queue_rois"]:
        roi.update(selected)
        roi["service_anchor"] = selected_anchor
    Path(config_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return save_roi_preview(video, config_path)

def main() -> None:
    """Parse CLI arguments and select or preview configured ROIs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--preview-only", action="store_true", help="save the current ROI polygons without opening the selector")
    args = parser.parse_args()
    if args.preview_only:
        print(f"Saved ROI preview: {save_roi_preview(args.video, args.config).resolve()}")
    else:
        print(f"Saved ROI preview: {select_rois(args.video, args.config).resolve()}")

if __name__ == "__main__":
    main()
