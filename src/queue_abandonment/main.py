"""Run detection, queue state tracking, annotation and event export."""
import argparse
import csv
import json
import random
import shutil
import subprocess
import time
from types import SimpleNamespace
from collections import defaultdict, deque
from pathlib import Path
from math import hypot
from typing import Any
import cv2
import numpy as np
from .annotator import Annotator
from .config_loader import load_config
from .counter import FirstComeCounter
from .detector_tracker import DetectorTracker
from .event_logger import write_outputs, write_summary_chart
from .identity_recovery import IdentityRecovery, StaffRegistry
from .state_machine import State, VisitMachine

def inside(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    """Check a point against a polygon with OpenCV's point-in-polygon test."""
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.int32), point, False) >= 0

def in_exit_region(point: tuple[float, float], polygon: list[list[int]], margin: float) -> bool:
    """Treat an optional outside band as still within the queue for exit debounce."""
    distance = cv2.pointPolygonTest(np.asarray(polygon, dtype=np.int32), point, True)
    return distance >= -margin

def signed_roi_distance(point: tuple[float, float], polygon: list[list[int]]) -> float:
    """Return negative distance inside an ROI and positive distance outside."""
    signed = cv2.pointPolygonTest(np.asarray(polygon, dtype=np.int32), point, True)
    return -float(signed)

def within_service_margin(point: tuple[float, float], rois: list[dict[str, Any]], margin: float) -> bool:
    """Check whether a point is inside any service ROI or its outside margin."""
    return any(signed_roi_distance(point, roi["service"]) <= margin for roi in rois)

def close_lost_visit(machine: VisitMachine, visit: Any, now: float, point: tuple[float, float],
                     rois: list[dict[str, Any]], margin: float,
                     anchor: tuple[float, float] | None = None, radius: float = 0.0,
                     allow_inferred_serve: bool = False, minimum_dwell: float = 1.8,
                     maximum_speed: float = 25.0) -> bool:
    """Close an identity-split visit as served when its last foot point is at the desk."""
    inferred = within_service_margin(point, rois, margin) or (anchor is not None and hypot(point[0]-anchor[0], point[1]-anchor[1]) <= 2*radius)
    dwell_qualified = service_dwell_qualified(float(getattr(visit, "service_dwell_seconds", 0.0)),
        float(getattr(visit, "service_average_speed_px_s", float("inf")) or float("inf")), minimum_dwell, maximum_speed)
    if inferred:
        machine.touch_service(visit.track_id, now)
        # Keep the visit open through identity recovery; the counter-order rule
        # decides service only after the configured lost-track timeout.
    else:
        machine.close_missing(visit, now, service_inferred=False)
    return inferred

def load_tracks_cache(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate frame-indexed detections saved by ``--save-tracks``."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise ValueError(f"Track cache {source} must contain a frames list.")
    for index, frame in enumerate(frames):
        if frame.get("frame_index") != index or not isinstance(frame.get("tracks"), list):
            raise ValueError(f"Invalid track cache frame at index {index}.")
    return frames

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a deterministic CSV diagnostic or review table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

def pending_episode_rows(visits: dict[str, Any], timeline: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize every pending interval and flag low-motion/near-edge cases."""
    output: list[dict[str, Any]] = []
    for visit in visits.values():
        transitions = sorted(visit.transitions, key=lambda row: row["timestamp"])
        starts = [row for row in transitions if row["new_state"] == State.PENDING_ABANDONMENT.value]
        for start in starts:
            end = next((row for row in transitions if row["timestamp"] >= start["timestamp"] and row["new_state"] in {State.CANCELLED.value, State.ABANDONED.value}), None)
            end_time = float(end["timestamp"]) if end else None
            if end and end["new_state"] == State.CANCELLED.value:
                outcome = "CANCELLED"
            elif end:
                outcome = "ABANDONED"
            else:
                outcome = "PENDING_AT_END"
                end_time = max((float(row["time"]) for row in timeline if row["visit_id"] == visit.visit_id), default=float(start["timestamp"]))
            interval = [row for row in timeline if row["visit_id"] == visit.visit_id and float(row["time"]) >= float(start["timestamp"]) - 1e-6 and float(row["time"]) <= end_time + 1e-6]
            by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in interval: by_track[int(row["track_id"])].append(row)
            track_id = int(visit.track_id)
            if by_track:
                track_id = max(by_track, key=lambda tid: len(by_track[tid]))
            points = [(float(row["smoothed_foot_x"]), float(row["smoothed_foot_y"])) for row in sorted(by_track.get(track_id, []), key=lambda row: row["time"])]
            movement = sum(hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]) for index in range(1, len(points)))
            max_distance = max((float(row["queue_signed_distance_px"]) for row in interval), default=0.0)
            suspect = movement < float(config["suspect_move_px"])
            visit_points = [row for row in timeline if row.get("visit_id") == visit.visit_id]
            closest_anchor = min((float(row.get("anchor_distance_px", float("inf"))) for row in visit_points), default=float("inf"))
            output.append({"visit_id": visit.visit_id, "track_id": track_id, "start_time": float(start["timestamp"]), "end_time": end_time,
                           "duration_seconds": max(0.0, end_time - float(start["timestamp"])), "outcome": outcome,
                           "max_foot_point_distance_px": max_distance, "total_foot_point_movement_px": movement,
                           "closest_anchor_distance_px": closest_anchor,
                           "abandonment_review": "SUSPECT" if outcome == "ABANDONED" and closest_anchor < 100.0 else "",
                           "suspect_false": "SUSPECT_FALSE" if suspect else ""})
    return sorted(output, key=lambda row: row["start_time"])

def service_check_rows(observations: dict[int, list[dict[str, Any]]], config: dict[str, Any], fps: float) -> list[dict[str, Any]]:
    """Measure actual and margin service occupancy and explain each outcome."""
    rows: list[dict[str, Any]] = []
    dt = 1.0 / max(fps, 1e-6)
    for tid, observations_for_track in sorted(observations.items()):
        values = sorted(observations_for_track, key=lambda item: item["time"])
        if not any(item["service_expanded"] for item in values): continue
        inside_seconds = sum(dt for item in values if item["inside_service"])
        margin_seconds = sum(dt for item in values if item["service_expanded"])
        entries = leaves = 0; previous_inside = False; current_streak = max_streak = 0.0; last_time = None
        for item in values:
            contiguous = last_time is not None and item["time"] - last_time <= dt * 1.6
            active_inside = bool(item["inside_service"])
            if active_inside and (not previous_inside or not contiguous): entries += 1
            if previous_inside and (not active_inside or not contiguous): leaves += 1
            if active_inside and contiguous: current_streak += dt
            elif active_inside: current_streak = dt
            else: current_streak = 0.0
            max_streak = max(max_streak, current_streak)
            previous_inside = active_inside; last_time = item["time"]
        if previous_inside: leaves += 1
        served = any(item["state"] == State.AT_SERVICE_POINT.value for item in values)
        final_state = values[-1]["state"]
        if served:
            reason = f"Reached AT_SERVICE_POINT after service_touch_seconds={config['service_touch_seconds']:.2f}s of anchor/patch contact; served visits stay terminal."
        elif inside_seconds <= 0:
            reason = "Not served: foot point never entered the service ROI; margin can preserve a timer but cannot start it."
        elif max_streak < float(config["min_service_seconds"]):
            reason = f"Not served: longest continuous inside dwell {max_streak:.2f}s is below min_service_seconds={config['min_service_seconds']:.2f}s."
        else:
            reason = "Not served: sufficient geometric dwell occurred without an active eligible queue visit at completion."
        rows.append({"track_id": tid, "total_seconds_inside_service": inside_seconds,
                     "longest_continuous_seconds_inside": max_streak, "total_seconds_within_service_margin": margin_seconds,
                     "entries": entries, "leaves": leaves, "state_at_end": final_state, "reason": reason,
                     "never_marked_served": not served})
    return rows

def write_ground_truth_template(path: Path, track_intervals: dict[int, list[float]]) -> None:
    """Create editable labels for every track that joined or entered pending."""
    if path.exists():
        return
    rows = [{"track_id": tid, "start_time": interval[0], "end_time": interval[1], "person_label": "",
             "really_joined": "", "really_served": "", "really_left": ""}
            for tid, interval in sorted(track_intervals.items())]
    write_csv(path, ["track_id", "start_time", "end_time", "person_label", "really_joined", "really_served", "really_left"], rows)

def create_review_material(video_path: Path, output: Path, fps: float, frame_count: int, transitions: list[dict[str, Any]]) -> None:
    """Save a 2-second annotated frame series, its contact sheet, and first service clip."""
    review_dir = output / "review_frames"; review_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output / "annotated.mp4"
    review_source = annotated_path if annotated_path.is_file() else video_path
    cap = cv2.VideoCapture(str(review_source))
    thumbs = []
    for frame_index in range(0, frame_count, max(1, round(fps * 2))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = cap.read()
        if not ok: continue
        timestamp = frame_index / fps
        label = f"t={timestamp:.1f}s"
        cv2.rectangle(frame, (frame.shape[1] - 180, 8), (frame.shape[1] - 8, 44), (20, 24, 30), -1)
        cv2.putText(frame, label, (frame.shape[1] - 168, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        path = review_dir / f"frame_{frame_index:06d}_{timestamp:07.2f}s.jpg"
        cv2.imwrite(str(path), frame)
        thumb = cv2.resize(frame, (320, 180))
        cv2.putText(thumb, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    cap.release()
    if thumbs:
        columns = 4; rows = (len(thumbs) + columns - 1) // columns
        sheet = np.full((rows * 190, columns * 320, 3), 245, dtype=np.uint8)
        for index, thumb in enumerate(thumbs):
            y, x = (index // columns) * 190, (index % columns) * 320
            sheet[y:y + 180, x:x + 320] = thumb
        cv2.imwrite(str(output / "diagnostics" / "review_contact_sheet.jpg"), sheet)
    service_events = [item for item in transitions if item["new_state"] == State.AT_SERVICE_POINT.value]
    if service_events:
        timestamp = float(service_events[0]["timestamp"])
        start_frame = max(0, round((timestamp - 1.0) * fps)); end_frame = min(frame_count - 1, round((timestamp + 2.0) * fps))
        source = cv2.VideoCapture(str(review_source)); source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = source.read()
        if ok:
            height, width = frame.shape[:2]
            clip = cv2.VideoWriter(str(review_dir / "first_service_clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            while ok and start_frame <= end_frame:
                clip.write(frame); start_frame += 1
                if start_frame > end_frame: break
                ok, frame = source.read()
            clip.release()
        source.release()

def create_customer_review_frames(annotated_path: Path, output: Path, rows: list[dict[str, Any]], fps: float) -> None:
    """Extract arrival, served and fixed desk-check frames from the real annotated video."""
    destination = output / "diagnostics" / "service_review_frames"
    destination.mkdir(parents=True, exist_ok=True)
    for previous in destination.glob("*.jpg"):
        previous.unlink()
    capture = cv2.VideoCapture(str(annotated_path))
    requested: dict[str, float] = {f"check_{second:02d}s": float(second) for second in (8, 12, 16, 20, 40, 44, 58)}
    for row in rows:
        for key in ("counter_arrival_time", "served_time"):
            if row.get(key) not in (None, ""):
                requested[f"{row['visit_id']}_{key}"] = float(row[key])
    for name, timestamp in requested.items():
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(timestamp * fps)))
        ok, frame = capture.read()
        if ok:
            cv2.imwrite(str(destination / f"{name}_{timestamp:.2f}s.jpg"), frame)
    capture.release()

def stationary_service_segments(observations: dict[int, list[dict[str, Any]]], fps: float,
                               maximum_speed: float, radius_px: float) -> list[dict[str, Any]]:
    """Find three-second slow foot-position clusters around the service anchor."""
    segments: list[dict[str, Any]] = []
    step_limit = 1.6 / max(fps, 1e-6)
    for track_id, values in sorted(observations.items()):
        ordered = sorted(values, key=lambda item: float(item["time"]))
        active: list[dict[str, Any]] = []
        for sample in ordered + [{"time": float("inf"), "speed": float("inf"), "distance": float("inf")}]:
            is_stationary = float(sample.get("speed", float("inf"))) <= maximum_speed and float(sample.get("distance", float("inf"))) <= 2.0*radius_px
            contiguous = not active or float(sample["time"])-float(active[-1]["time"]) <= step_limit
            if is_stationary and contiguous:
                active.append(sample)
                continue
            if len(active) > 1:
                duration = float(active[-1]["time"])-float(active[0]["time"])
                if duration >= 3.0:
                    segments.append({"track_id": track_id, "start_time": active[0]["time"], "end_time": active[-1]["time"],
                        "duration_seconds": duration, "centroid_x": sum(float(item["x"]) for item in active)/len(active),
                        "centroid_y": sum(float(item["y"]) for item in active)/len(active),
                        "median_distance_to_anchor_px": float(np.median([float(item["distance"]) for item in active])),
                        "median_speed_px_s": float(np.median([float(item["speed"]) for item in active])), "samples": len(active)})
            active = [sample] if is_stationary else []
    return segments

def box_intersects_polygon(box: list[int], polygon: list[list[int]]) -> bool:
    """Check box overlap for staff-only classification when feet are hidden."""
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), ((x1 + x2) / 2, (y1 + y2) / 2)]
    if any(inside(point, polygon) for point in corners):
        return True
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in polygon)

def should_start_pending(state: State, in_queue: bool, in_service: bool) -> bool:
    """Return whether a queue exit should start abandonment debounce."""
    return state == State.IN_QUEUE and not in_queue and not in_service

def lost_track_exit_eligible(state: State, queue_signed_distance_px: float,
                             queue_exit_margin_px: float, in_service_reach: bool,
                             outward_motion: bool = False) -> bool:
    """A lost track outside the queue while moving outward enters pending, not TRACK_LOST."""
    beyond_band = queue_signed_distance_px > queue_exit_margin_px
    clear_departure = queue_signed_distance_px > 0 and outward_motion
    return state == State.IN_QUEUE and (beyond_band or clear_departure) and not in_service_reach

def pending_return_allowed(state: State, in_queue: bool, in_service_margin: bool) -> bool:
    """Pending can cancel only on strict queue re-entry or a service handoff."""
    return state == State.PENDING_ABANDONMENT and (in_queue or in_service_margin)

def running_event_totals(visits: dict[str, Any]) -> dict[str, int]:
    """Count completed visit events from immutable visit transition history."""
    return {
        "served_total": sum(bool(getattr(visit, "served", visit.state == State.AT_SERVICE_POINT)) for visit in visits.values()),
        "reached_service_brief_total": sum(visit.last_outcome == "REACHED_SERVICE_BRIEF" and visit.brief_finalized for visit in visits.values()),
        "abandoned_total": sum(any(item["new_state"] == State.ABANDONED.value for item in visit.transitions)
                                for visit in visits.values()),
        "cancelled_total": sum(item["new_state"] == State.CANCELLED.value
                                for visit in visits.values() for item in visit.transitions),
        "track_lost_total": sum(visit.last_outcome == "TRACK_LOST" for visit in visits.values()),
    }

def assert_panel_log_totals(rows: list[dict[str, Any]], counters: dict[str, int]) -> None:
    """Fail if final panel queue/direct counts differ from serialized event rows."""
    expected = event_panel_totals(rows)
    for key, value in expected.items():
        if int(counters.get(key, 0)) != value:
            raise AssertionError(f"Panel/log mismatch for {key}: panel={counters.get(key, 0)}, events={value}")

def event_panel_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Derive every displayed running total from the exact event-row collection."""
    served = [row for row in rows if row.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"}]
    return {"served_total": len(served),
            "served_queue_total": sum(row.get("visit_type", "QUEUE") == "QUEUE" for row in served),
            "served_direct_total": sum(row.get("visit_type") == "DIRECT" for row in served),
            "reached_service_brief_total": sum(row.get("event_outcome") == "REACHED_SERVICE_BRIEF" for row in rows),
            "abandoned_total": sum(row.get("event_outcome") == "ABANDONED" for row in rows),
            "cancelled_total": sum(bool(row.get("cancelled", False)) for row in rows)}

def assert_totals_monotonic(previous: dict[str, int], current: dict[str, int]) -> None:
    """Running event totals may increase, but must never fall between frames."""
    for key in ("served_total", "served_queue_total", "served_direct_total", "abandoned_total", "cancelled_total"):
        if int(current.get(key, 0)) < int(previous.get(key, 0)):
            raise AssertionError(f"Panel total {key} decreased: {previous.get(key)} -> {current.get(key)}")

def direct_approach_eligible(first_distance_px: float, current_distance_px: float,
                             direct_approach_px: float, inside_staff_zone: bool) -> bool:
    """A direct customer must be observed approaching from outside the desk area."""
    return not inside_staff_zone and first_distance_px >= direct_approach_px and current_distance_px < first_distance_px - 40.0

def staff_box_bottom_inside(box: list[int], polygon: list[Any]) -> bool:
    """Frame-wise staff display rule based only on bbox bottom-center in the zone."""
    bottom = ((box[0] + box[2]) / 2.0, float(box[3]))
    return any(inside(bottom, area) for area in staff_zone_polygons(polygon))

def queue_exit_elapsed(state: State, signed_distance_px: float, exit_margin_px: float,
                       in_service_margin: bool, elapsed_seconds: float,
                       debounce_seconds: float) -> bool:
    """Check that a visit stayed beyond the expanded queue edge for its debounce."""
    return (state == State.IN_QUEUE and signed_distance_px > exit_margin_px
            and not in_service_margin and elapsed_seconds >= debounce_seconds)

def near_service_touch(point: tuple[float, float], anchor: tuple[float, float], radius: float,
                       service_polygons: list[list[list[int]]]) -> bool:
    """Service touch means sustained anchor proximity or entry into the service patch."""
    return hypot(point[0] - anchor[0], point[1] - anchor[1]) <= radius or any(inside(point, poly) for poly in service_polygons)

def queue_join_eligible(point: tuple[float, float], queue_polygons: list[list[list[int]]],
                        anchor: tuple[float, float], service_exclusion_px: float,
                        inside_staff_zone: bool) -> bool:
    """Require queue-floor evidence away from the counter and staff floor."""
    return (not inside_staff_zone and hypot(point[0]-anchor[0], point[1]-anchor[1]) > service_exclusion_px
            and any(inside(point, polygon) for polygon in queue_polygons))

def service_candidate_eligible(state: State | None, is_staff: bool, in_staff_zone: bool,
                               has_queue_history: bool, direct_touch_complete: bool,
                               at_touch: bool) -> bool:
    """Bystanders do not compete until a non-queue customer completes anchor touch."""
    if is_staff or in_staff_zone or not at_touch:
        return False
    if state in {State.IN_QUEUE, State.PENDING_ABANDONMENT}:
        return True
    return not has_queue_history and direct_touch_complete

def service_completion_time(started_at: float, now: float, minimum: float) -> float | None:
    """Return service transition time after continuous dwell is met."""
    return started_at + minimum if now - started_at >= minimum else None

def service_dwell_qualified(duration_seconds: float, average_speed_px_s: float,
                            minimum_seconds: float, maximum_speed_px_s: float) -> bool:
    """Count service only after enough continuous dwell while moving slowly."""
    return duration_seconds >= minimum_seconds and average_speed_px_s <= maximum_speed_px_s

def protect_near_service_departure(visit: Any, now: float, observations: dict[int, list[dict[str, Any]]], threshold: float) -> bool:
    """Close a close-approach leaver as inferred served instead of pending."""
    ids = [int(visit.track_id), *[int(value) for value in visit.previous_track_ids]]
    closest = min((float(point["distance"]) for tid in ids for point in observations.get(tid, [])), default=float("inf"))
    if closest > threshold or visit.served:
        return False
    old = visit.state
    visit.state = State.AT_SERVICE_POINT
    visit.reached_service = visit.served = visit.service_inferred = True
    visit.contact_type = "PASS_THROUGH"
    visit.served_time = visit.queue_exit_time = float(now)
    visit.counter_leave_time = float(now)
    visit.time_at_counter_seconds = None
    visit.last_outcome = "SERVED"
    visit.queue_closed = True
    visit.transitions.append({"timestamp": float(now), "visit_id": visit.visit_id, "old_state": old.value,
                              "new_state": State.AT_SERVICE_POINT.value, "reason": "closest_approach_protection"})
    return True

def brief_upgrade_allowed(visit: Any, now: float, point: tuple[float, float],
                          anchor: tuple[float, float], radius: float, window_seconds: float) -> bool:
    """Keep a brief visit upgradeable only for its original track and within the spatial/time window."""
    start = getattr(visit, "brief_since", None)
    return (getattr(visit, "state", None) == State.REACHED_SERVICE_BRIEF and not getattr(visit, "brief_finalized", True)
            and start is not None and now - float(start) <= window_seconds
            and hypot(point[0]-anchor[0], point[1]-anchor[1]) <= 2.0 * radius)

def desk_staff_should_classify(near_desk: bool, ever_queued: bool, elapsed_seconds: float,
                               staff_dwell_seconds: float) -> bool:
    """Recognize a long-staying desk-side person without queue history as staff."""
    return near_desk and not ever_queued and elapsed_seconds >= staff_dwell_seconds

def identity_split_last_point(previous_point: tuple[float, float] | None,
                              replacement_point: tuple[float, float]) -> tuple[float, float]:
    """Use the lost identity's last position, never its replacement's location."""
    return previous_point if previous_point is not None else replacement_point

def torso_signature(frame: Any, box: list[int] | tuple[int, ...]) -> np.ndarray | None:
    """Return a normalized HSV histogram for the upper-body part of a person box."""
    if frame is None:
        return None
    x1, y1, x2, y2 = map(int, box)
    height = max(1, y2 - y1)
    width = max(1, x2 - x1)
    crop = frame[max(0, y1 + int(height * 0.18)):max(0, y1) + int(height * 0.68),
                 max(0, x1 + int(width * 0.15)):max(0, x1) + int(width * 0.85)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    chromatic = cv2.inRange(hsv, (0, 35, 35), (180, 255, 255))
    if cv2.countNonZero(chromatic) < 5:
        return None
    hist = cv2.calcHist([hsv], [0, 1], chromatic, [18, 4], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def box_iou(first: list[int] | tuple[int, ...], second: list[int] | tuple[int, ...]) -> float:
    """Intersection over union for xyxy boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1, area_a + area_b - intersection)

def overlapping_track_ids(detections: list[dict[str, Any]], threshold: float) -> set[int]:
    """Return both IDs for each pair whose boxes overlap above the ambiguity limit."""
    ambiguous: set[int] = set()
    for index, first in enumerate(detections):
        for second in detections[index + 1:]:
            if box_iou(first["box"], second["box"]) >= threshold:
                ambiguous.update((int(first["track_id"]), int(second["track_id"])))
    return ambiguous

def merge_duplicate_detections(detections: list[dict[str, Any]], threshold: float,
                               active_track_ids: set[int] | None = None,
                               containment_threshold: float = 0.8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Suppress near-identical boxes before tracking logic; prefer an active visit owner."""
    active = active_track_ids or set()
    kept: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: (int(item["track_id"]) not in active, -float(item.get("confidence", 0.0)))):
        def overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
            iou = box_iou(a["box"], b["box"])
            x1, y1 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
            x2, y2 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            smaller = min(max(1, (a["box"][2]-a["box"][0])*(a["box"][3]-a["box"][1])),
                          max(1, (b["box"][2]-b["box"][0])*(b["box"][3]-b["box"][1])))
            return iou, inter / smaller
        duplicate = next((item for item in kept if (lambda score: score[0] >= threshold or score[1] >= containment_threshold)(overlap(item, det))), None)
        if duplicate is None:
            kept.append(det)
            continue
        merged.append({"kept_track_id": int(duplicate["track_id"]), "suppressed_track_id": int(det["track_id"]),
                       "iou": box_iou(duplicate["box"], det["box"]), "containment": overlap(duplicate, det)[1]})
    return kept, merged

def in_expanded_polygon(point: tuple[float, float], polygon: list[list[int]], margin: float) -> bool:
    """Return true inside a polygon or within a pixel margin outside it."""
    return bool(polygon) and signed_roi_distance(point, polygon) <= margin

def staff_zone_polygons(value: list[Any]) -> list[list[list[float]]]:
    """Accept either one polygon or a list of disjoint staff polygons."""
    if not value:
        return []
    if len(value) >= 3 and isinstance(value[0], (list, tuple)) and len(value[0]) == 2 and all(isinstance(v, (int, float)) for v in value[0]):
        return [value]
    return list(value)

def staff_zone_contains(box: list[int], foot: tuple[float, float], polygon: list[Any], margin: float) -> bool:
    """Staff zone membership uses either the tracked foot or box-bottom center."""
    polygons = staff_zone_polygons(polygon)
    if not polygons:
        return False
    bottom = ((box[0] + box[2]) / 2.0, float(box[3]))
    return any(in_expanded_polygon(foot, area, margin) or in_expanded_polygon(bottom, area, margin)
               for area in polygons)

class StaffZoneRegistry:
    """Classify stationary staff from scene position, independent of tracker ID history."""
    def __init__(self) -> None:
        self.since: dict[int, float] = {}
        self.staff_ids: set[int] = set()
        self.rules: dict[int, str] = {}

    def observe(self, track_id: int, in_zone: bool, now: float, born_in_zone: bool,
                ever_queued: bool, minimum_seconds: float, already_served: bool = False) -> bool:
        """Return staff status; queued customers cannot become staff by entering the polygon."""
        tid = int(track_id)
        if tid in self.staff_ids:
            return True
        if already_served:
            self.since.pop(tid, None)
            return False
        if ever_queued:
            self.since.pop(tid, None)
            return False
        if not in_zone:
            self.since.pop(tid, None)
            return False
        self.since.setdefault(tid, now)
        if born_in_zone:
            self.staff_ids.add(tid); self.rules[tid] = "staff_zone_new_track_inside"
        elif now - self.since[tid] >= minimum_seconds:
            self.staff_ids.add(tid); self.rules[tid] = "staff_zone_dwell"
        return tid in self.staff_ids

def service_owners(candidates: list[tuple], anchor: tuple[float, float],
                   radius: float, maximum: int) -> set[int]:
    """Choose the closest eligible customer in anchor radius or service patch."""
    eligible = sorted((hypot(point[0]-anchor[0], point[1]-anchor[1]), int(tid))
                      for item in candidates for tid, point in [item[:2]]
                      if hypot(point[0]-anchor[0], point[1]-anchor[1]) <= radius or (len(item) > 2 and bool(item[2])))
    return {tid for _, tid in eligible[:max(0, int(maximum))]}

class DirectServiceRegistry:
    """Count non-queue service dwell once across short detector-ID fragments."""
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.track_to_record: dict[int, dict[str, Any]] = {}

    def observe(self, track_id: int, point: tuple[float, float], now: float,
                min_seconds: float, merge_px: float, merge_seconds: float,
                speed: float = 0.0, max_speed: float = float("inf"),
                contact_start: float | None = None) -> dict[str, Any]:
        """Update a DIRECT record, merging a nearby short-gap track fragment."""
        tid = int(track_id); record = self.track_to_record.get(tid)
        if record is None:
            options = [(float(np.linalg.norm(np.asarray(point)-np.asarray(item["last_point"]))), item)
                       for item in self.records if now - item["last_time"] <= merge_seconds]
            options = [(distance, item) for distance, item in options if distance <= merge_px]
            record = min(options, key=lambda item: item[0])[1] if options else None
        if record is None:
            record = {"visit_id": f"direct_{len(self.records)+1:04d}", "track_id": tid, "track_ids": set(),
                      "start_time": contact_start if contact_start is not None else now, "last_time": now, "last_point": point, "served_time": None,
                      "stable_since": now if speed <= max_speed else None, "max_speed": max_speed, "closed": False,
                      "service_dwell_seconds": 0.0, "average_speed_px_s": None, "stable_speeds": []}
            self.records.append(record)
        record["track_ids"].add(tid); self.track_to_record[tid] = record
        record.update({"last_time": now, "last_point": point, "closed": False})
        if speed <= max_speed:
            record["stable_since"] = record.get("stable_since") if record.get("stable_since") is not None else now
            record["stable_speeds"].append(float(speed))
            if record["served_time"] is None and now - float(record["stable_since"]) >= min_seconds:
                record["served_time"] = float(record["stable_since"])
        else:
            record["stable_since"] = None
        record["last_speed"] = speed
        if record["stable_since"] is not None:
            record["service_dwell_seconds"] = max(0.0, now - float(record["stable_since"]))
            record["average_speed_px_s"] = sum(record["stable_speeds"]) / len(record["stable_speeds"])
        return record

    @property
    def served(self) -> list[dict[str, Any]]:
        return [item for item in self.records if item["served_time"] is not None]

    @property
    def brief(self) -> list[dict[str, Any]]:
        return [item for item in self.records if item["served_time"] is None and item["closed"]]

def run(config_path: str, video_path: str, save_tracks: bool = False, load_tracks: bool = False,
        config_overrides: dict[str, Any] | None = None, render_video: bool = True,
        write_artifacts: bool = True) -> dict[str, Any]:
    """Process a video and write annotated video, events and transitions."""
    started = time.perf_counter()
    config = load_config(config_path)
    if config_overrides:
        config.update(config_overrides)
    if save_tracks and load_tracks:
        raise ValueError("Choose either --save-tracks or --load-tracks, not both.")
    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"Video not found: {video}. Add a clip under data/videos/ or generate data/videos/demo.mp4.")
    random.seed(config["seed"]); np.random.seed(config["seed"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}. Check that the file is a valid video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release(); raise ValueError("Video has invalid FPS or frame size.")
    try:
        config = load_config(config_path, (width, height))
        if config_overrides:
            config.update(config_overrides)
        cache_path = Path(config["output_dir"]) / "tracks_cache.json"
        cached_frames = load_tracks_cache(cache_path) if load_tracks else None
        detector = None if load_tracks else DetectorTracker(config)
    except Exception:
        capture.release(); raise
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    writer = None
    writer_codec = "not written"
    if render_video:
        for codec in ("avc1", "H264"):
            candidate = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            if candidate.isOpened():
                writer = candidate; writer_codec = f"{codec}/H.264"; break
            candidate.release()
        if writer is None:
            writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            writer_codec = "mp4v fallback (H.264 encoder unavailable)"
    cache_tmp = cache_path.with_suffix(".json.tmp")
    cache_file = cache_tmp.open("w", encoding="utf-8") if save_tracks else None
    if cache_file:
        cache_file.write('{"schema_version":1,"frames":[\n')
    machine = VisitMachine(float(config["grace_period_seconds"]))
    recovery = IdentityRecovery(float(config["lost_track_timeout_seconds"]), float(config["reassociate_distance_px"]),
                                float(config["appearance_min_similarity"]), float(config["max_link_direction_angle"]))
    staff = StaffRegistry(config["ignore_track_ids"], config["staff_min_dwell_seconds"], config["staff_max_speed"],
                          config["lost_track_timeout_seconds"], config["reassociate_distance_px"],
                          config["staff_initial_dwell_seconds"])
    histories: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=int(config["trajectory_smoothing_window"])))
    candidate_since: dict[int, float] = {}; candidate_paths: dict[int, list[tuple[float, float]]] = {}; outside_since: dict[int, float] = {}; service_since: dict[str, tuple[float, float, int]] = {}; service_dwell_since: dict[str, float] = {}
    speed_window_samples = max(1, round(0.5 * fps))
    direct_dwell_since: dict[int, float] = {}; speed_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=speed_window_samples)); desk_staff_since: dict[int, float] = {}
    direct_since: dict[int, float] = {}; direct_observations: dict[int, list[tuple[float, tuple[float, float]]]] = defaultdict(list)
    direct_first_distance: dict[int, float] = {}; direct_approached_ids: set[int] = set(); direct_born_in_staff_zone: set[int] = set()
    tight_service_since: dict[str, float] = {}
    talking_owner_visit_id: str | None = None
    talking_owner_last_seen: float | None = None
    counter_leave_since: dict[str, float] = {}
    staff_zone_registry = StaffZoneRegistry()
    first_frame_staff_ids: set[int] = set()
    first_frame_queue_ids: set[int] = set()
    direct_registry = DirectServiceRegistry(); served_hidden: set[int] = set(); served_tags: dict[int, float] = {}
    counter = FirstComeCounter(float(config["service_touch_seconds"]), float(config["service_reach_grace_seconds"]), float(config["max_counter_hold_seconds"]))
    direct_counter_owner: int | None = None; direct_counter_since: float | None = None
    offscreen_rows: dict[int, dict[str, Any]] = {}
    last_position: dict[int, tuple[float, float]] = {}; last_seen: dict[int, float] = {}; known_ids: set[int] = set()
    annotator = Annotator(config, config["queue_rois"], fps, width, height)
    frame_index = 0; alert_history: list[tuple[float, Any]] = []; last_panel_counters: dict[str, int] = {}
    previous_totals = {key: 0 for key in ("served_total", "served_queue_total", "served_direct_total", "abandoned_total", "cancelled_total")}
    panel_total_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    service_observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    anchor_observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    staff_zone_observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    service_speed_samples: dict[str, list[float]] = defaultdict(list)
    counter_timeline_rows: list[dict[str, Any]] = []
    visit_track_ids: dict[str, set[int]] = defaultdict(set)
    multi_live_visit_frames: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    ambiguity_frames: dict[int, int] = defaultdict(int)
    ambiguity_visits: dict[int, set[str]] = defaultdict(set)
    ambiguous_track_ids: set[int] = set()
    overlap_pending: set[int] = set()
    track_start_end: dict[int, list[float]] = {}
    track_first_seen: dict[int, float] = {}; track_last_seen: dict[int, float] = {}; track_frame_counts: dict[int, int] = defaultdict(int)
    service_only_records: dict[int, dict[str, Any]] = {}; staff_observations: dict[int, dict[str, Any]] = {}
    def protect_queue_identity(track_id: int) -> None:
        """A detector ID linked to a queue visit cannot remain classified as staff."""
        tid = int(track_id)
        staff.staff_ids.discard(tid); staff.staff_rules.pop(tid, None); staff.staff_positions.pop(tid, None)
        staff.ever_queued.add(tid)
        staff_zone_registry.staff_ids.discard(tid); staff_zone_registry.rules.pop(tid, None); staff_zone_registry.since.pop(tid, None)
        staff_observations.pop(tid, None)
    separator = ""
    cache_complete = False
    try:
        while True:
            if load_tracks:
                if frame_index >= len(cached_frames): break
                cache_record = cached_frames[frame_index]
                if render_video:
                    ok, frame = capture.read()
                    if not ok: break
                else:
                    frame = None
                detections = [{**track, "foot": tuple(track.get("foot", ((track["box"][0] + track["box"][2]) / 2, track["box"][3])))} for track in cache_record["tracks"]]
            else:
                ok, frame = capture.read()
                if not ok: break
                detections = detector.track(frame)
            now = frame_index / fps
            detections, duplicates = merge_duplicate_detections(
                detections, float(config.get("duplicate_iou_threshold", 0.70)), set(machine.active),
                float(config.get("duplicate_containment_threshold", 0.8)))
            for item in duplicates:
                duplicate_rows.append({"frame_index": frame_index, "time": now, **item})
            if cache_file:
                cache_record = {"frame_index": frame_index, "time": now, "tracks": [
                    {"track_id": int(det["track_id"]), "box": [int(v) for v in det["box"]], "confidence": float(det["confidence"]), "foot": [float(det["foot"][0]), float(det["foot"][1])]}
                    for det in detections
                ]}
                cache_file.write(separator); json.dump(cache_record, cache_file, separators=(",", ":")); separator = ",\n"
                if (frame_index + 1) % 50 == 0:
                    cache_file.flush()
                    print(f"Track cache progress: {frame_index + 1} frames", flush=True)
            visible = {int(det["track_id"]) for det in detections}
            # Ambiguity affects only active queue visits outside both service patch and margin.
            eligible_ambiguous: set[int] = set()
            for det in detections:
                tid = int(det["track_id"]); visit = machine.active.get(tid)
                foot = det["foot"]
                near_service = within_service_margin(tuple(foot), config["queue_rois"], float(config["service_margin_px"]))
                if visit is not None and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT} and not near_service:
                    eligible_ambiguous.add(tid)
            overlapped = overlapping_track_ids(detections, float(config["overlap_iou_threshold"]))
            ambiguous_track_ids = overlapped & eligible_ambiguous
            for tid in ambiguous_track_ids:
                ambiguity_frames[tid] += 1
                visit = machine.active.get(tid)
                if visit is not None: ambiguity_visits[tid].add(visit.visit_id)
            signature_current = {int(det["track_id"]): (None if int(det["track_id"]) in ambiguous_track_ids else torso_signature(frame, det["box"])) for det in detections}
            recovery.reconcile_live(machine.active, signature_current, now,
                                    {int(det["track_id"]): tuple(det["foot"]) for det in detections})
            recovered_ids: set[int] = set()
            # Restore a known queue visit before the first-frame staff-zone rule runs.
            # Otherwise an ID switch inside the staff polygon can relabel a customer as staff.
            for det in detections:
                tid = int(det["track_id"])
                track_first_seen.setdefault(tid, now)
                if tid in machine.active or now - track_first_seen[tid] < float(config["min_track_age_seconds"]):
                    continue
                live_visit_ids = {visit.visit_id for owner, visit in machine.active.items() if owner in visible and owner != tid}
                sample = recovery.track_samples.get(tid)
                match = recovery.recover(tid, tuple(det["foot"]), now,
                    sample[3] if sample is not None else signature_current.get(tid), live_visit_ids,
                    {counter.owner_visit_id} if counter.owner_visit_id else set())
                if match is None or not (machine.accepts_alias(match.visit) or counter.is_owner(match.visit.visit_id)):
                    continue
                for old_owner, old_visit in list(machine.active.items()):
                    if old_visit is match.visit:
                        machine.active.pop(old_owner, None)
                machine.active[tid] = match.visit
                protect_queue_identity(tid)
                recovered_ids.add(tid)
            smoothed_current: dict[int, tuple[float, float]] = {}
            geometry_current: dict[int, tuple[float, float, bool, bool, bool, bool]] = {}
            service_candidates: dict[str, int] = {}
            service_nearby: set[str] = set()
            service_outside: set[str] = set()
            service_speeds: dict[str, float] = {}
            unconfirmed_ids: set[int] = set()
            anchor = tuple(map(float, config["queue_rois"][0].get("service_anchor", (0, 0))))
            has_service_anchor = config["queue_rois"][0].get("service_anchor") is not None
            eligible_service: list[tuple] = []
            reach_by_id: dict[int, bool] = {}
            direct_eligible_by_id: dict[int, bool] = {}
            queue_observations: list[dict[str, Any]] = []
            for item in detections:
                tid0 = int(item["track_id"]); point0 = tuple(map(float, item["foot"]))
                inside_patch = any(inside(point0, roi["service"]) for roi in config["queue_rois"])
                distance_to_anchor = hypot(point0[0] - anchor[0], point0[1] - anchor[1])
                inside_tight_circle0 = inside_patch or distance_to_anchor <= float(config["service_radius_px"])
                zone0 = any(staff_zone_contains(item["box"], point0, roi.get("staff_zone", []), float(config["staff_zone_margin_px"])) for roi in config["queue_rois"])
                direct_first_distance.setdefault(tid0, distance_to_anchor)
                if tid0 not in track_first_seen and zone0:
                    direct_born_in_staff_zone.add(tid0)
                if direct_approach_eligible(direct_first_distance[tid0], distance_to_anchor,
                                            float(config["direct_approach_px"]), zone0 or tid0 in direct_born_in_staff_zone):
                    direct_approached_ids.add(tid0)
                if frame_index == 0 and zone0:
                    first_frame_staff_ids.add(tid0)
                reach0 = any(signed_roi_distance(point0, roi["service"]) <= float(config["service_reach_px"]) for roi in config["queue_rois"])
                reach_by_id[tid0] = reach0
                active_visit = machine.active.get(tid0)
                has_queue_history0 = tid0 in staff.ever_queued or active_visit is not None or bool(active_visit and active_visit.served)
                direct_record0 = direct_registry.track_to_record.get(tid0)
                direct_already_served0 = direct_record0 is not None and direct_record0["served_time"] is not None
                staff0 = (zone0 or tid0 in staff.staff_ids or tid0 in staff_zone_registry.staff_ids) and not (has_queue_history0 or direct_already_served0)
                direct_eligible = False
                existing_direct = direct_registry.track_to_record.get(tid0)
                if staff0 or zone0:
                    direct_since.pop(tid0, None)
                elif (active_visit is None and tid0 not in staff.ever_queued and not staff0 and reach0
                      and tid0 in direct_approached_ids
                      and not (existing_direct is not None and existing_direct["served_time"] is not None)):
                    direct_since.setdefault(tid0, now)
                    direct_observations[tid0].append((now, point0))
                    direct_eligible = now-direct_since[tid0] >= float(config["direct_confirm_seconds"])
                elif not reach0:
                    direct_since.pop(tid0, None)
                direct_eligible_by_id[tid0] = direct_eligible
                visit_state = active_visit.state if active_visit is not None else None
                if active_visit is not None and visit_state in {State.IN_QUEUE, State.PENDING_ABANDONMENT, State.AT_SERVICE_POINT}:
                    queue_observations.append({"visit_id": active_visit.visit_id, "track_id": tid0,
                                               "in_reach": reach0 and not inside_tight_circle0,
                                               "in_service_patch": inside_tight_circle0, "staff": staff0})
                if active_visit is None and reach0 and not staff0 and tid0 not in staff.ever_queued and (direct_eligible or tid0 == direct_counter_owner):
                    eligible_service.append((tid0, point0, inside_patch))
            for tid, visit in machine.active.items():
                if tid in visible or tid not in last_position or tid not in last_seen:
                    continue
                lost_for = now - last_seen[tid]
                point = last_position[tid]
                if lost_for <= float(config["lost_track_timeout_seconds"]) and any(
                        signed_roi_distance(point, roi["service"]) <= float(config["service_reach_px"]) for roi in config["queue_rois"]):
                    queue_observations.append({"visit_id": visit.visit_id, "track_id": tid, "in_reach": True, "staff": False})
            for absent_id in set(direct_since) - visible:
                direct_since.pop(absent_id, None)
            old_counter_owner = counter.owner_visit_id
            acquired_visit_id = counter.update(now, queue_observations)
            if old_counter_owner and old_counter_owner != counter.owner_visit_id:
                old_record = machine.visits.get(old_counter_owner)
                if old_record is not None and old_record.counter_leave_time is None:
                    old_record.counter_leave_time = now
                    old_record.time_at_counter_seconds = max(0.0, now - float(old_record.counter_arrival_time or now))
            if direct_counter_owner is not None and direct_counter_since is not None and now - direct_counter_since >= float(config["max_counter_hold_seconds"]):
                direct_counter_owner = None; direct_counter_since = None
            owner_visit_id = counter.owner_visit_id
            owner_track_id = counter.track_for_owner()
            service_owner_ids = {owner_track_id} if owner_track_id is not None else set()
            counter_timeline_rows.append({"frame_index": frame_index, "time": now,
                "owner_visit_id": owner_visit_id or "", "owner_track_id": owner_track_id if owner_track_id is not None else "",
                "in_reach_visit_ids": ";".join(sorted({str(item["visit_id"]) for item in queue_observations if item.get("in_reach") and not item.get("staff")})),
                "newly_acquired_visit_id": acquired_visit_id or ""})
            # DIRECT customers use the slot only when no queue visit owns it.
            if owner_visit_id is None and eligible_service:
                available_direct_ids = {int(item[0]) for item in eligible_service}
                if direct_counter_owner is not None and direct_counter_owner in available_direct_ids:
                    service_owner_ids.add(direct_counter_owner)
                else:
                    direct_counter = min(eligible_service, key=lambda item: hypot(item[1][0]-anchor[0], item[1][1]-anchor[1]))
                    direct_counter_owner = int(direct_counter[0]); direct_counter_since = now
                    service_owner_ids.add(direct_counter_owner)
            elif direct_counter_owner is not None:
                direct_counter_owner = None; direct_counter_since = None
            fired = machine.tick(now)
            for visit in fired:
                position = last_position.get(int(visit.track_id))
                alert_history.append((now, {"visit_id": visit.visit_id, "wait_seconds": visit.duration(now),
                                            "position": position, "track_id": int(visit.track_id)}))
            for det in detections:
                tid = int(det["track_id"]); known_ids.add(tid); previous_seen = last_seen.get(tid); last_seen[tid] = now
                track_first_seen.setdefault(tid, now); track_last_seen[tid] = now; track_frame_counts[tid] += 1
                previous_position = last_position.get(tid)
                foot = det["foot"]; histories[tid].append(foot)
                smoothed = (sum(p[0] for p in histories[tid]) / len(histories[tid]), sum(p[1] for p in histories[tid]) / len(histories[tid]))
                if previous_position is not None and previous_seen is not None and now > previous_seen:
                    speed_history[tid].append(hypot(smoothed[0]-previous_position[0], smoothed[1]-previous_position[1]) / (now-previous_seen))
                current_speed = float(np.median(speed_history[tid])) if speed_history[tid] else 0.0
                last_position[tid] = smoothed
                anchor_distance_now = hypot(smoothed[0]-anchor[0], smoothed[1]-anchor[1])
                anchor_observations[tid].append({"time": now, "x": smoothed[0], "y": smoothed[1], "distance": anchor_distance_now, "speed": current_speed,
                    "inside_radius": anchor_distance_now <= float(config["service_radius_px"]),
                    "within_2x_radius": anchor_distance_now <= 2.0 * float(config["service_radius_px"])})
                smoothed_current[tid] = smoothed
                box = det["box"]
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                service_point = center if config["service_use_box_center_fallback"] else smoothed
                queue_distance = min(signed_roi_distance(smoothed, roi["queue"]) for roi in config["queue_rois"])
                service_distance = min(signed_roi_distance(service_point, roi["service"]) for roi in config["queue_rois"])
                queue_pair = queue_distance <= 0
                queue_expanded = queue_distance <= float(config["queue_exit_margin_px"])
                service_pair = service_distance <= 0
                service_expanded = service_distance <= float(config["service_margin_px"])
                geometry_current[tid] = (queue_distance, service_distance, queue_pair, service_pair, queue_expanded, service_expanded)
                if frame_index == 0 and queue_pair:
                    first_frame_queue_ids.add(tid)
                visit = machine.active.get(tid)
                in_staff_zone = any(staff_zone_contains(box, smoothed, roi.get("staff_zone", []), float(config["staff_zone_margin_px"])) for roi in config["queue_rois"])
                if frame_index == 0 and in_staff_zone:
                    first_frame_staff_ids.add(tid)
                staff_zone_observations[tid].append({"time": now, "x": smoothed[0], "y": smoothed[1], "in_staff_zone_or_margin": in_staff_zone})
                born_in_zone = tid in first_frame_staff_ids
                direct_record = direct_registry.track_to_record.get(tid)
                already_direct_served = direct_record is not None and direct_record["served_time"] is not None
                has_queue_history = tid in staff.ever_queued or visit is not None
                if staff_zone_registry.observe(tid, in_staff_zone, now, born_in_zone, has_queue_history,
                                               float(config["staff_zone_min_seconds"]), already_served=already_direct_served or bool(visit and visit.served)):
                    staff.staff_ids.add(tid); staff.staff_rules[tid] = staff_zone_registry.rules.get(tid, "staff_zone_dwell")
                near_desk = hypot(smoothed[0]-anchor[0], smoothed[1]-anchor[1]) <= 2.0 * float(config["service_radius_px"])
                if near_desk and tid not in staff.ever_queued and tid not in staff.staff_ids:
                    desk_staff_since.setdefault(tid, now)
                    if desk_staff_should_classify(True, tid in staff.ever_queued, now - desk_staff_since[tid], float(config["staff_dwell_seconds"])):
                        staff.staff_ids.add(tid); staff.staff_rules[tid] = "staff_desk_dwell_without_queue_history"
                else:
                    desk_staff_since.pop(tid, None)
                signature = signature_current[tid]
                if tid in ambiguous_track_ids:
                    overlap_pending.add(tid)
                elif visit is not None and signature is not None and tid in overlap_pending:
                    expected = recovery.visit_appearance.get(visit.visit_id)
                    similarity = recovery.appearance_similarity(expected, signature)
                    if expected is not None and similarity is not None and similarity < float(config["appearance_min_similarity"]):
                        recovery.decisions.append({"old_track_id": tid, "new_track_id": tid, "timestamp": now,
                            "distance_px": 0.0, "appearance_similarity": similarity, "direction_angle_degrees": None,
                            "accepted": False, "decision": "rejected: track appearance changed; split identity", "visit_id": visit.visit_id})
                        # The current box may be the person who replaced this identity
                        # during an overlap. Only the old track's last stable point can
                        # describe where its visit ended.
                        inferred_point = identity_split_last_point(previous_position, smoothed)
                        inferred_service = within_service_margin(inferred_point, config["queue_rois"], float(config["service_margin_px"])) or hypot(inferred_point[0]-anchor[0], inferred_point[1]-anchor[1]) <= 2*float(config["service_radius_px"])
                        if inferred_service:
                            close_lost_visit(machine, visit, now, inferred_point, config["queue_rois"], float(config["service_margin_px"]), anchor, float(config["service_radius_px"]), minimum_dwell=float(config["min_service_seconds"]), maximum_speed=float(config["service_max_speed"]))
                        elif visit.state == State.IN_QUEUE:
                            prior_distance = min(signed_roi_distance(previous_position or smoothed, roi["queue"]) for roi in config["queue_rois"])
                            if prior_distance > float(config["queue_exit_margin_px"]):
                                if not protect_near_service_departure(visit, now, anchor_observations, float(config["reach_protect_px"])):
                                    machine.leave(tid, now)
                            else:
                                machine.close_missing(visit, now)
                        if visit is None or not visit.reached_service:
                            machine.active.pop(tid, None)
                            visit = None
                        outside_since.pop(tid, None)
                    overlap_pending.discard(tid)
                recovery.observe(tid, smoothed, now, signature, visit, overlap=tid in ambiguous_track_ids)
                min_track_age = float(config["min_track_age_seconds"])
                track_age = now - track_first_seen[tid]
                if track_age + 1e-9 < min_track_age and tid not in staff.staff_ids:
                    if frame_index == 0:
                        staff.observe(tid, center, now, service_pair, visit is not None, visible, first_video_frame=True)
                    unconfirmed_ids.add(tid)
                    continue
                if tid in ambiguous_track_ids:
                    # Freeze join/cancel/exit decisions until the overlapping boxes separate.
                    outside_since.pop(tid, None)
                    unconfirmed_ids.add(tid)
                    continue
                is_staff = tid in staff.staff_ids or tid in staff_zone_registry.staff_ids
                if in_staff_zone and not is_staff and visit is None:
                    unconfirmed_ids.add(tid)
                    continue
                if service_pair and visit is None and tid not in staff.ever_queued and not is_staff:
                    record = service_only_records.setdefault(tid, {"track_id": tid, "first_service_time": now,
                        "last_service_time": now, "total_seconds_inside_service": 0.0, "max_confidence": float(det["confidence"])})
                    if now - float(record["last_service_time"]) <= 1.6 / fps:
                        record["total_seconds_inside_service"] += 1.0 / fps
                    record["last_service_time"] = now
                    record["max_confidence"] = max(float(record["max_confidence"]), float(det["confidence"]))
                if is_staff:
                    outside_since.pop(tid, None); candidate_since.pop(tid, None); candidate_paths.pop(tid, None)
                    observed = staff_observations.setdefault(tid, {"track_id": tid, "first_time": now,
                        "first_x": smoothed[0], "first_y": smoothed[1], "first_region": "service" if service_pair else "other",
                        "rule_fired": staff.staff_rules.get(tid, "unknown"), "ever_in_queue": tid in staff.ever_queued})
                    observed.update({"last_time": now, "last_x": smoothed[0], "last_y": smoothed[1],
                                     "last_region": "service" if service_pair else "other",
                                     "ever_in_queue": bool(observed["ever_in_queue"] or tid in staff.ever_queued)})
                    continue
                join_eligible = queue_join_eligible(smoothed, [roi["queue"] for roi in config["queue_rois"]],
                    anchor, (2.0 * float(config["service_radius_px"]) if has_service_anchor else 0.0), in_staff_zone or is_staff)
                if visit is None and join_eligible and not staff.blocks_queue(tid):
                    if tid not in candidate_since:
                        candidate_since[tid] = now; candidate_paths[tid] = [smoothed]
                    elif candidate_paths[tid][-1] != smoothed:
                        candidate_paths[tid].append(smoothed)
                    elapsed = now - candidate_since[tid]
                    path = candidate_paths[tid]
                    path_length = sum(float(np.linalg.norm(np.asarray(path[index]) - np.asarray(path[index - 1]))) for index in range(1, len(path)))
                    speed = path_length / max(elapsed, 1 / fps)
                    if elapsed >= config["min_join_seconds"] and speed < config["max_join_speed"]:
                        at_video_start = tid in first_frame_queue_ids
                        estimated_entry = max(0.0, now - float(config["queue_entry_backdated_seconds"])) if at_video_start else now
                        visit = machine.join(tid, now, queue_entry_estimated=at_video_start)
                        visit.queue_entry_time = estimated_entry
                        staff.mark_queued(tid)
                        candidate_since.pop(tid, None); candidate_paths.pop(tid, None)
                else:
                    candidate_since.pop(tid, None); candidate_paths.pop(tid, None)
                visit = machine.active.get(tid)
                if visit is not None:
                    recovery.observe(tid, smoothed, now, signature, visit, overlap=False)
                if visit and pending_return_allowed(visit.state, queue_pair, service_expanded):
                    machine.return_to_queue(tid, now)
                    visit = machine.active.get(tid)
                in_reach_now = reach_by_id.get(tid, False)
                inside_tight_service = (any(inside(smoothed, roi["service"]) for roi in config["queue_rois"])
                                        or hypot(smoothed[0]-anchor[0], smoothed[1]-anchor[1]) <= float(config["service_radius_px"]))
                talk_spot = service_distance <= float(config.get("talk_radius_px", 30.0))
                if visit and talk_spot:
                    if talking_owner_visit_id is None or (talking_owner_last_seen is not None and now-talking_owner_last_seen > 1.0):
                        talking_owner_visit_id = visit.visit_id
                    if talking_owner_visit_id == visit.visit_id:
                        talking_owner_last_seen = now
                if visit and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT} and talk_spot:
                    visit.counter_arrival_time = visit.counter_arrival_time or now
                    if visit.state == State.PENDING_ABANDONMENT:
                        machine.return_to_queue(tid, now)
                    stopped_since = tight_service_since.get(visit.visit_id)
                    if talking_owner_visit_id == visit.visit_id and current_speed <= float(config["service_max_speed"]):
                        if stopped_since is None:
                            tight_service_since[visit.visit_id] = now
                            stopped_since = now
                        if now - stopped_since >= float(config["min_service_seconds"]):
                            served_at = float(stopped_since)
                            visit.contact_type = "STOPPED"
                            visit.service_inferred = False
                            visit.service_dwell_seconds = now - served_at
                            visit.service_average_speed_px_s = current_speed
                            visit.served_time = served_at
                            machine.service(tid, served_at, reason="stopped_talking_spot_dwell")
                            visit.counter_leave_time = None
                            counter.mark_served(visit.visit_id)
                            served_tags[tid] = now + 2.0
                            outside_since.pop(tid, None)
                    else:
                        tight_service_since.pop(visit.visit_id, None)
                elif visit and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT}:
                    tight_service_since.pop(visit.visit_id, None)
                visit = machine.active.get(tid)
                if visit and visit.state == State.AT_SERVICE_POINT and visit.counter_arrival_time is not None and visit.counter_leave_time is None:
                    if talk_spot:
                        counter_leave_since.pop(visit.visit_id, None)
                    else:
                        leave_since = counter_leave_since.setdefault(visit.visit_id, now)
                        if now-leave_since >= float(config["service_reach_grace_seconds"]):
                            visit.counter_leave_time = leave_since
                            visit.time_at_counter_seconds = max(0.0, leave_since - float(visit.counter_arrival_time))
                            served_tags[tid] = now + 2.0
                if talking_owner_visit_id and visit and visit.visit_id == talking_owner_visit_id and not talk_spot:
                    if talking_owner_last_seen is not None and now-talking_owner_last_seen > float(config["service_reach_grace_seconds"]):
                        talking_owner_visit_id = None
                        talking_owner_last_seen = None
                if visit and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT} and in_reach_now:
                    if visit.visit_id == counter.owner_visit_id:
                        reach_record = counter.records.get(visit.visit_id)
                        if visit.counter_arrival_time is None and reach_record is not None:
                            visit.counter_arrival_time = reach_record.first_reach
                    outside_since.pop(tid, None)
                    if visit.state == State.PENDING_ABANDONMENT:
                        machine.return_to_queue(tid, now)
                        visit = machine.active.get(tid)
                    if visit.visit_id == counter.owner_visit_id:
                        stopped_since = service_dwell_since.get(visit.visit_id)
                        required_stop = float(config["min_service_seconds"]) if talk_spot else float(config["ring_service_seconds"])
                        if current_speed <= float(config["service_max_speed"]):
                            if stopped_since is None:
                                service_dwell_since[visit.visit_id] = now
                                stopped_since = now
                            if now - stopped_since >= required_stop:
                                served_at = float(stopped_since)
                                visit.service_dwell_seconds = now - served_at
                                visit.service_average_speed_px_s = float(np.median(speed_history[tid])) if speed_history[tid] else current_speed
                                visit.contact_type = "STOPPED"
                                visit.counter_leave_time = None
                                machine.service(tid, served_at, reason="stopped_dwell_at_counter")
                                visit.time_at_counter_seconds = max(0.0, now - float(visit.counter_arrival_time or now))
                                visit.service_inferred = False
                                counter.mark_served(visit.visit_id)
                        else:
                            service_dwell_since.pop(visit.visit_id, None)
                elif visit and visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT} and not inside_tight_service:
                    service_outside.add(visit.visit_id)
                if tid in service_owner_ids and visit is None and tid not in staff.staff_ids and tid not in staff.ever_queued and direct_eligible_by_id.get(tid, False):
                    direct_record = direct_registry.observe(tid, smoothed, now, float(config["min_service_seconds"]),
                        float(config["track_merge_px"]), float(config["track_merge_seconds"]),
                        speed=current_speed, max_speed=float(config["service_max_speed"]),
                        contact_start=direct_since.get(tid, now))
                visit = machine.active.get(tid)
                if visit and visit.state == State.ABANDONED and queue_pair:
                    machine.active.pop(tid, None)
                    visit = None
                if visit and (queue_pair or service_expanded or in_reach_now):
                    if pending_return_allowed(visit.state, queue_pair, service_expanded):
                        machine.return_to_queue(tid, now)
                    outside_since.pop(tid, None)
                elif visit and visit.state == State.IN_QUEUE and not queue_expanded and not service_expanded and not in_reach_now:
                    outside_since.setdefault(tid, now)
                    closest = min((float(item["distance"]) for alias in [int(visit.track_id), *[int(v) for v in visit.previous_track_ids]] for item in anchor_observations.get(alias, [])), default=float("inf"))
                    if closest <= float(config["reach_protect_px"]):
                        visit.service_inferred = True; visit.contact_type = "PASS_THROUGH"
                        visit.served_time = now; visit.queue_exit_time = now
                        visit.last_outcome = "SERVED"; visit.reached_service = True; visit.served = True
                        visit.queue_closed = True; visit.counter_leave_time = now
                        visit.time_at_counter_seconds = None
                        visit.transitions.append({"timestamp": now, "visit_id": visit.visit_id, "old_state": visit.state.value, "new_state": State.AT_SERVICE_POINT.value, "reason": "reach_protection_inferred"})
                        visit.state = State.AT_SERVICE_POINT
                        counter.mark_served(visit.visit_id)
                        continue
                    if queue_exit_elapsed(visit.state, queue_distance, float(config["queue_exit_margin_px"]),
                                          bool(service_expanded), now - outside_since[tid],
                                          float(config["exit_debounce_seconds"])):
                        outside_since.pop(tid)
                        if not protect_near_service_departure(visit, now, anchor_observations, float(config["reach_protect_px"])):
                            machine.leave(tid, now)
                elif visit and visit.state == State.IN_QUEUE:
                    outside_since.pop(tid, None)
            for record in direct_registry.records:
                if not (record["track_ids"] & visible) and now - float(record["last_time"]) > 1.6 / fps:
                    record["closed"] = True
            for visit_id in service_nearby:
                timer = service_since.get(visit_id)
                if timer is None:
                    continue
                started_at, _, tid = timer
                service_since[visit_id] = (started_at, now, tid)
                completed_at = service_completion_time(started_at, now, float(config["service_touch_seconds"]))
                if completed_at is not None:
                    machine.touch_service(tid, completed_at)
                dwell_start = service_dwell_since.get(visit_id)
                if dwell_start is not None:
                    dwell_done = service_completion_time(dwell_start, now, float(config["min_service_seconds"]))
                    visit = machine.active.get(tid)
                    if dwell_done is not None and visit is not None and service_dwell_qualified(dwell_done-dwell_start, service_speeds.get(visit_id, float("inf")), float(config["min_service_seconds"]), float(config["service_max_speed"])):
                        visit.service_dwell_seconds = dwell_done - dwell_start
                        samples = service_speed_samples.get(visit_id, [])
                        visit.service_average_speed_px_s = sum(samples) / len(samples) if samples else service_speeds.get(visit_id)
                        machine.service(tid, dwell_done, reason="service_dwell_and_low_speed_met")
                    elif visit is not None:
                        visit.service_dwell_seconds = max(visit.service_dwell_seconds, now - dwell_start)
                        samples = service_speed_samples.get(visit_id, [])
                        if samples:
                            visit.service_average_speed_px_s = sum(samples) / len(samples)
            for visit_id in service_outside - service_nearby:
                service_since.pop(visit_id, None)
                service_dwell_since.pop(visit_id, None)
            for visit_id, (_, last_candidate_time, _) in list(service_since.items()):
                if visit_id not in service_nearby and now - last_candidate_time > float(config["lost_track_timeout_seconds"]):
                    service_since.pop(visit_id, None)
            # Preserve brief losses, then close stale ghosts instead of leaving open visits.
            for tid in known_ids - visible:
                if tid in machine.active and tid in last_seen and now - last_seen[tid] <= 1 / fps + 1e-6:
                    visit = machine.active[tid]
                    if visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT}:
                        qdist = min(signed_roi_distance(last_position[tid], roi["queue"]) for roi in config["queue_rois"])
                        in_queue = qdist <= 0
                        moving_away = qdist > float(config["queue_exit_margin_px"])
                        in_reach = any(signed_roi_distance(last_position[tid], roi["service"]) <= float(config["service_reach_px"]) for roi in config["queue_rois"])
                        if visit.state == State.IN_QUEUE and moving_away and not in_reach:
                            if not protect_near_service_departure(visit, now, anchor_observations, float(config["reach_protect_px"])):
                                machine.leave(tid, now)
                            visit = machine.active[tid]
                        sample = recovery.track_samples.get(tid)
                        velocity = sample[2] if sample else (0.0, 0.0)
                        recovery.mark_lost(tid, last_position[tid], last_seen[tid], visit, in_queue,
                                           timeout_seconds=float(config["lost_track_timeout_seconds"]) if in_reach else min(float(config["lost_track_timeout_seconds"]), float(config["max_occlusion_seconds"])),
                                           velocity=velocity, moving_away=moving_away)
                    elif visit.state == State.AT_SERVICE_POINT and now - float(visit.queue_exit_time or now) <= float(config["service_grace_seconds"]):
                        in_queue = any(inside(last_position[tid], roi["queue"]) for roi in config["queue_rois"])
                        recovery.mark_lost(tid, last_position[tid], last_seen[tid], visit, in_queue,
                                           timeout_seconds=float(config["lost_track_timeout_seconds"]))
            for det in detections:
                tid = int(det["track_id"])
                if tid not in machine.active and now - track_first_seen.get(tid, now) >= float(config["min_track_age_seconds"]):
                    live_visit_ids = {visit.visit_id for owner, visit in machine.active.items() if owner in visible and owner != tid}
                    current_sample = recovery.track_samples.get(tid)
                    candidate_appearance = current_sample[3] if current_sample is not None else signature_current.get(tid)
                    match = recovery.recover(tid, det["foot"], now, candidate_appearance, live_visit_ids,
                        {counter.owner_visit_id} if counter.owner_visit_id else set())
                    if tid not in recovered_ids and match and (machine.accepts_alias(match.visit) or counter.is_owner(match.visit.visit_id)):
                        for old_owner, active_visit in list(machine.active.items()):
                            if active_visit is match.visit:
                                machine.active.pop(old_owner, None)
                        if match.visit.state == State.IN_QUEUE and match.visit.last_outcome == "TRACK_LOST" and not match.visit.served:
                            match.visit.queue_exit_time = None
                            match.visit.last_outcome = None
                        machine.active[tid] = match.visit; protect_queue_identity(tid)
            # A lost non-pending visit is closed after the occlusion limit. Pending keeps its grace clock.
            for tid, visit in list(machine.active.items()):
                if visit.state == State.REACHED_SERVICE_BRIEF and not visit.brief_finalized:
                    if now > float(visit.brief_since or now) + float(config.get("brief_upgrade_seconds", 5.0)):
                        machine.finalize_brief(visit)
                    continue
                point = last_position.get(tid)
                near_reach = point is not None and any(signed_roi_distance(point, roi["service"]) <= float(config["service_reach_px"]) for roi in config["queue_rois"])
                if visit.state == State.IN_QUEUE and point is not None:
                    queue_distance_at_loss = min(signed_roi_distance(point, roi["queue"]) for roi in config["queue_rois"])
                    recent_timeline = [row for row in timeline_rows if int(row["track_id"]) == int(tid)
                                       and now - float(row["time"]) <= 0.8]
                    outward_motion = (len(recent_timeline) >= 2 and
                                      float(recent_timeline[-1]["queue_signed_distance_px"]) >
                                      float(recent_timeline[0]["queue_signed_distance_px"]) + 5.0)
                    if lost_track_exit_eligible(visit.state, queue_distance_at_loss,
                                                float(config["queue_exit_margin_px"]), near_reach, outward_motion):
                        # The track was last observed beyond the queue band; retain an
                        # abandonment timer instead of silently closing it as TRACK_LOST.
                        if not protect_near_service_departure(visit, float(last_seen[tid]), anchor_observations, float(config["reach_protect_px"])):
                            machine.leave(tid, float(last_seen[tid]))
                loss_limit = float(config["lost_track_timeout_seconds"]) if near_reach else float(config["max_occlusion_seconds"])
                if tid in visible or tid not in last_seen or now - last_seen[tid] <= loss_limit:
                    continue
                if visit.state == State.PENDING_ABANDONMENT and not near_reach:
                    continue
                if near_reach and counter.is_owner_or_next(visit.visit_id):
                    if not visit.served:
                        machine.service(tid, now, reason="counter_visit_inferred_after_unrecovered_track_loss")
                    visit.service_inferred = True
                    visit.contact_type = visit.contact_type or "STOPPED"
                    visit.time_at_counter_seconds = None
                    visit.counter_leave_time = now
                else:
                    machine.close_missing(visit, now, service_inferred=False)
                    if visit.counter_arrival_time is not None and visit.counter_leave_time is None:
                        visit.service_inferred = True
                        visit.contact_type = visit.contact_type or "STOPPED"
                        visit.time_at_counter_seconds = None
                        visit.counter_leave_time = now
                machine.active.pop(tid, None)
            for lost in recovery.expire(now):
                departed = IdentityRecovery.likely_departure(lost.position, (width, height), lost.inside_queue, int(config["frame_edge_margin_px"]))
                if departed and lost.visit.state == State.IN_QUEUE:
                    if not protect_near_service_departure(lost.visit, now, anchor_observations, float(config["reach_protect_px"])):
                        machine.leave(lost.track_id, now)
            for det in detections:
                tid = int(det["track_id"])
                queue_distance, service_distance, in_queue, in_service, queue_expanded, service_expanded = geometry_current[tid]
                visit = machine.active.get(tid)
                if tid in unconfirmed_ids:
                    state = "AMBIGUOUS" if tid in ambiguous_track_ids else "UNCONFIRMED"; visit_id = visit.visit_id if visit is not None else ""
                elif tid in staff.staff_ids:
                    state = "STAFF"; visit_id = ""
                elif visit is not None:
                    state = visit.state.value; visit_id = visit.visit_id
                    visit_track_ids[visit_id].add(tid)
                elif tid in service_only_records:
                    state = "WALK_UP"; visit_id = ""
                elif tid in candidate_since:
                    state = "Candidate"; visit_id = ""
                else:
                    state = State.OUTSIDE_QUEUE.value; visit_id = ""
                foot = smoothed_current[tid]
                timeline_rows.append({
                    "frame_index": frame_index, "time": now, "track_id": tid,
                    "smoothed_foot_x": foot[0], "smoothed_foot_y": foot[1],
                    "queue_signed_distance_px": queue_distance, "inside_queue_roi": in_queue,
                    "inside_service_roi": in_service, "expanded_roi_flag": queue_expanded,
                    "service_expanded_roi_flag": service_expanded, "service_signed_distance_px": service_distance,
                    "anchor_distance_px": hypot(foot[0]-anchor[0], foot[1]-anchor[1]),
                    "state": state, "visit_id": visit_id, "detection_visible": True,
                    "identity_ambiguous": tid in ambiguous_track_ids,
                })
                service_observations[tid].append({"time": now, "service_distance": service_distance,
                                                  "inside_service": in_service, "service_expanded": service_expanded,
                                                  "state": state, "visit_id": visit_id})
                if state in {State.IN_QUEUE.value, State.PENDING_ABANDONMENT.value}:
                    interval = track_start_end.setdefault(tid, [now, now]); interval[1] = now
            visible_visit_ids = {active_visit.visit_id for owner, active_visit in machine.active.items() if owner in visible}
            live_counts: dict[str, list[int]] = defaultdict(list)
            for owner, active_visit in machine.active.items():
                if owner in visible:
                    live_counts[active_visit.visit_id].append(owner)
            for visit_id, owners in live_counts.items():
                if len(owners) > 1:
                    multi_live_visit_frames.append({"frame_index": frame_index, "time": now,
                                                    "visit_id": visit_id, "track_ids": ";".join(map(str, owners))})
            for visit in machine.visits.values():
                if visit.visit_id in visible_visit_ids or visit.state not in {State.IN_QUEUE, State.PENDING_ABANDONMENT, State.AT_SERVICE_POINT}:
                    continue
                aliases = [tid for tid, active_visit in machine.active.items() if active_visit is visit and tid in last_position]
                if not aliases:
                    continue
                tid = max(aliases, key=lambda item: last_seen.get(item, -1.0))
                foot = last_position[tid]
                queue_distance = min(signed_roi_distance(foot, roi["queue"]) for roi in config["queue_rois"])
                service_distance = min(signed_roi_distance(foot, roi["service"]) for roi in config["queue_rois"])
                timeline_rows.append({"frame_index": frame_index, "time": now, "track_id": tid,
                    "smoothed_foot_x": foot[0], "smoothed_foot_y": foot[1], "queue_signed_distance_px": queue_distance,
                    "inside_queue_roi": queue_distance <= 0, "inside_service_roi": service_distance <= 0,
                    "expanded_roi_flag": queue_distance <= float(config["queue_exit_margin_px"]),
                    "service_expanded_roi_flag": service_distance <= float(config["service_margin_px"]),
                    "service_signed_distance_px": service_distance,
                    "anchor_distance_px": hypot(foot[0]-anchor[0], foot[1]-anchor[1]), "state": visit.state.value,
                    "visit_id": visit.visit_id, "detection_visible": False, "identity_ambiguous": False})
                if visit.state in {State.IN_QUEUE, State.PENDING_ABANDONMENT}:
                    interval = track_start_end.setdefault(tid, [now, now]); interval[1] = now
            final_frame = frame_index >= source_frame_count - 1
            if final_frame:
                for final_visit in machine.visits.values():
                    if final_visit.state == State.REACHED_SERVICE_BRIEF:
                        machine.finalize_brief(final_visit)
            current = list({id(visit): visit for visit in machine.active.values()}.values())
            projected_rows = machine.finish(now)
            projected_rows.extend({"visit_id": item["visit_id"], "visit_type": "DIRECT", "event_outcome": "SERVED", "cancelled": False}
                                  for item in direct_registry.records if item["served_time"] is not None
                                  and not (item["track_ids"] & staff.staff_ids))
            counters = event_panel_totals(projected_rows)
            assert_totals_monotonic(previous_totals, counters)
            previous_totals = {key: counters.get(key, 0) for key in previous_totals}
            panel_total_rows.append({"frame_index": frame_index, "time": now, **previous_totals})
            counters.update({"in_queue": sum(v.state == State.IN_QUEUE and not v.queue_closed for v in current),
                             "pending": sum(v.state == State.PENDING_ABANDONMENT for v in current),
                             "walkup_total": len(service_only_records),
                             "track_lost_total": sum(v.last_outcome == "TRACK_LOST" for v in machine.visits.values())})
            last_panel_counters = counters.copy()
            alerts = [item for stamp, item in alert_history if now - stamp <= max(3.0, float(config["alert_seconds"]))]
            feet = {int(det["track_id"]): det["foot"] for det in detections}
            visible_direct_ids = {tid for tid in direct_registry.track_to_record if tid in visible}
            display_visits = {tid: visit for tid, visit in machine.active.items()
                              if (visit.state not in {State.REACHED_SERVICE_BRIEF, State.AT_SERVICE_POINT} and not visit.queue_closed)
                              or (visit.counter_arrival_time is not None and visit.counter_leave_time is None)}
            if owner_track_id is not None and owner_track_id in machine.active:
                display_visits[owner_track_id] = machine.active[owner_track_id]
            for tid in visible_direct_ids:
                record = direct_registry.track_to_record[tid]
                served = record["served_time"] is not None
                display_visits[tid] = SimpleNamespace(state=State.AT_SERVICE_POINT if served else State.OUTSIDE_QUEUE,
                    visit_id=record["visit_id"], pending_start_time=None,
                    counter_arrival_time=record.get("start_time"), counter_leave_time=record.get("last_time") if now-float(record.get("last_time", now)) > float(config["service_reach_grace_seconds"]) else None,
                    duration=lambda timestamp, start=record["start_time"]: timestamp-start)
            served_now = {tid for tid in visible if tid in machine.active and machine.active[tid].state in {State.AT_SERVICE_POINT, State.REACHED_SERVICE_BRIEF}
                          and not (machine.active[tid].counter_arrival_time is not None and machine.active[tid].counter_leave_time is None)}
            served_now.update(tid for tid in visible_direct_ids if direct_registry.track_to_record[tid]["served_time"] is not None)
            for tid in served_now - service_owner_ids - served_hidden:
                served_hidden.add(tid)
                served_tags[tid] = now + 2.0
            hidden_ids = set(served_hidden)
            for tid, visit in machine.active.items():
                if visit.state == State.REACHED_SERVICE_BRIEF:
                    served_tags.setdefault(tid, now + 2.0)
            for tid, expiry in list(served_tags.items()):
                if now > expiry:
                    served_tags.pop(tid, None)
            for det in detections:
                tid = int(det["track_id"]); box = det["box"]
                if int(box[3]) >= height - 1:
                    offscreen_rows[tid] = {"track_id": tid, "first_time": offscreen_rows.get(tid, {}).get("first_time", now),
                        "last_time": now, "foot_x": float(last_position[tid][0]), "foot_y": float(last_position[tid][1]), "foot_status": "off-screen"}
            frame_staff_ids = set(staff.staff_ids) | set(staff_zone_registry.staff_ids)
            for det in detections:
                if any(staff_box_bottom_inside(det["box"], roi.get("staff_zone", [])) for roi in config["queue_rois"]):
                    frame_staff_ids.add(int(det["track_id"]))
            if writer is not None:
                counter_markers = {int(visit.track_id): last_position[int(visit.track_id)] for visit in machine.visits.values()
                                   if visit.counter_arrival_time is not None and visit.counter_leave_time is None
                                   and int(visit.track_id) not in visible and int(visit.track_id) in last_position
                                   and now-float(visit.counter_arrival_time) <= float(config["max_counter_hold_seconds"])}
                writer.write(annotator.draw(frame, detections, display_visits, now, counters, alerts,
                    frame_staff_ids, set(candidate_since), feet, set(), unconfirmed_ids, ambiguous_track_ids,
                    frame_index=frame_index, visit_labels={tid: (f"AT COUNTER {max(0.0, now - float(record.counter_arrival_time or now)):.1f}s" if record.state == State.AT_SERVICE_POINT and record.counter_arrival_time is not None else f"V{int(record.visit_id.split('_')[-1])}") for tid, record in machine.active.items()},
                    direct_track_ids=visible_direct_ids, hidden_track_ids=hidden_ids, served_tags=served_tags,
                    served_tag_labels={tid: "SERVED" for tid, visit in machine.active.items() if visit.state == State.AT_SERVICE_POINT},
                    counter_markers=counter_markers))
            frame_index += 1
        cache_complete = True
    finally:
        capture.release()
        if writer is not None: writer.release()
        if cache_file is not None:
            if cache_complete:
                cache_file.write("\n]}")
            cache_file.close()
            if cache_complete:
                cache_tmp.replace(cache_path)
    if render_video:
        probe_args = [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,pix_fmt", "-show_entries", "format=size", "-of", "json",
            str(output / "annotated.mp4")]
        probe = subprocess.run(probe_args, check=True, capture_output=True, text=True)
        probe_data = json.loads(probe.stdout)
        codec = probe_data["streams"][0]["codec_name"]
        if codec.lower() != "h264":
            temporary_h264 = output / "annotated.h264.tmp.mp4"
            command = [shutil.which("ffmpeg"), "-y", "-i", str(output / "annotated.mp4"), "-c:v", "libx264",
                       "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(temporary_h264)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            temporary_h264.replace(output / "annotated.mp4")
            probe = subprocess.run(probe_args, check=True, capture_output=True, text=True)
            probe_data = json.loads(probe.stdout)
            codec = probe_data["streams"][0]["codec_name"]
        if codec.lower() != "h264":
            raise RuntimeError(f"H.264 verification failed after ffmpeg conversion: {codec}")
        writer_codec = "H.264 verified by ffprobe"
        print(f"ffprobe: {probe.stdout.strip()}")
    duration = max(0.0, (frame_index - 1) / fps)
    for visit in machine.visits.values():
        if visit.state == State.REACHED_SERVICE_BRIEF:
            machine.finalize_brief(visit)
    rows = machine.finish(duration)
    direct_events = [record for record in direct_registry.records if record["served_time"] is not None and not (record["track_ids"] & staff.staff_ids)]
    direct_events.sort(key=lambda item: float(item["served_time"]))
    for index, record in enumerate(direct_events, start=1):
        record["visit_id"] = f"direct_{index:04d}"
    queue_track_ids = {int(value) for row in rows if row.get("visit_type", "QUEUE") == "QUEUE"
                       for value in [row["track_id"], *row.get("previous_track_ids", [])]}
    valid_service_only_records = {tid: row for tid, row in service_only_records.items()
                                  if tid not in staff.staff_ids and tid not in queue_track_ids}
    for record in direct_events:
        served_direct = True
        rows.append({"visit_id": record["visit_id"], "track_id": record["track_id"], "previous_track_ids": sorted(record["track_ids"] - {record["track_id"]}),
            "queue_entry_time": None, "queue_exit_time": record["served_time"] if served_direct else record["last_time"], "queue_duration_seconds": 0.0,
            "queue_entry_estimated": False, "reached_service": True, "service_inferred": False, "visit_type": "DIRECT",
            "counter_arrival_time": record.get("start_time"), "served_time": record["served_time"],
            "counter_leave_time": record.get("last_time"), "time_at_counter_seconds": max(0.0, float(record.get("last_time", 0))-float(record.get("start_time", 0))),
            "contact_type": "STOPPED", "event_outcome": "SERVED",
            "pending_start_time": None, "abandonment_confirmed_time": None})
    for row in rows:
        row.setdefault("visit_type", "QUEUE")
        if row.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"}:
            if row.get("counter_arrival_time") in (None, ""):
                row["counter_arrival_time"] = row.get("served_time")
            if row.get("service_inferred"):
                row["time_at_counter_seconds"] = None
                row["contact_type"] = row.get("contact_type") or "STOPPED"
            elif row.get("time_at_counter_seconds") in (None, "") and row.get("counter_leave_time") not in (None, ""):
                row["time_at_counter_seconds"] = max(0.0, float(row["counter_leave_time"])-float(row["counter_arrival_time"]))
    final_totals = event_panel_totals(rows)
    assert_panel_log_totals(rows, final_totals)
    assert_panel_log_totals(rows, last_panel_counters)
    print("Final panel totals (from event rows):", final_totals)
    transitions = [transition for visit in machine.visits.values() for transition in visit.transitions]
    transitions.extend({"timestamp": item["served_time"], "visit_id": item["visit_id"], "old_state": State.OUTSIDE_QUEUE.value,
                        "new_state": State.AT_SERVICE_POINT.value, "reason": "direct_service_dwell_met"}
                        for item in direct_events if item["served_time"] is not None)
    episodes = pending_episode_rows(machine.visits, timeline_rows, config)
    for episode in episodes:
        if episode["outcome"] == "ABANDONED":
            print(f"Abandonment review: {episode['visit_id']} closest_anchor={episode['closest_anchor_distance_px']:.1f}px {episode['abandonment_review'] or 'not flagged'}")
    service_rows = service_check_rows(service_observations, config, fps)
    served_queue_count = sum(visit.served for visit in machine.visits.values())
    served_count = served_queue_count + sum(item["served_time"] is not None for item in direct_events)
    abandoned_count = sum(row["event_outcome"] == "ABANDONED" for row in rows)
    pending_count = sum(row["event_outcome"] == "PENDING_AT_END" for row in rows)
    old_served_times = {"visit_0001": 6.671706290242694, "visit_0002": 9.040307082714214,
                        "visit_0003": 22.517979197622587, "visit_0004": 27.08878652798415,
                        "visit_0006": 36.129502228826155, "visit_0008": 52.876102030708275,
                        "direct_0001": 63.35173105497771}
    truth_path = Path(config["ground_truth_path"])
    truth_labels = {str(item.get("visit_id")): item.get("person_label", "")
                    for item in csv.DictReader(truth_path.open(encoding="utf-8"))} if truth_path.is_file() else {}
    review_rows = []
    for row in rows:
        track_ids = [int(row["track_id"]), *[int(value) for value in row.get("previous_track_ids", [])]]
        observations = [point for track_id in track_ids for point in anchor_observations.get(track_id, [])]
        closest = min((float(point["distance"]) for point in observations), default=None)
        arrival = row.get("counter_arrival_time")
        nearest = min(observations, key=lambda point: abs(float(point["time"])-float(arrival))) if arrival is not None and observations else None
        arrival_speed = nearest.get("speed") if nearest else None
        nearby = sorted((point for point in observations if arrival is not None and abs(float(point["time"])-float(arrival)) <= 0.5), key=lambda point: point["time"])
        moving_away = len(nearby) >= 2 and float(nearby[-1]["distance"]) > float(nearby[0]["distance"])
        flag = "REVIEW" if row.get("visit_type", "QUEUE") == "QUEUE" and row.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"} and (moving_away or row.get("contact_type") == "PASS_THROUGH") else ""
        review_rows.append({"visit_id": row["visit_id"], "person_label": truth_labels.get(str(row["visit_id"]), ""),
            "event_outcome": row.get("event_outcome"), "counter_arrival_time": arrival, "served_time": row.get("served_time"),
            "old_served_time": old_served_times.get(str(row["visit_id"])), "speed_at_arrival_px_s": arrival_speed,
            "closest_approach_px": closest, "time_at_counter_seconds": row.get("time_at_counter_seconds"),
            "contact_type": row.get("contact_type"), "service_inferred": row.get("service_inferred"), "review_flag": flag})
        print(f"Visit {row['visit_id']} closest approach: {closest if closest is not None else 'unknown'} px")
    inferred_served_count = sum(bool(row.get("service_inferred")) and row.get("event_outcome") in {"SERVED", "CANCELLED_THEN_SERVED"} for row in rows)
    print("Served events inferred:", inferred_served_count)
    calibration_segments = stationary_service_segments(anchor_observations, fps, float(config["service_max_speed"]), float(config["service_radius_px"]))
    print(f"Stationary service calibration segments (>=3s): {len(calibration_segments)}")
    for segment in calibration_segments:
        print(f"  track {segment['track_id']}: {segment['duration_seconds']:.2f}s centroid=({segment['centroid_x']:.1f},{segment['centroid_y']:.1f}) distance={segment['median_distance_to_anchor_px']:.1f}px speed={segment['median_speed_px_s']:.1f}px/s")
    cancelled_visits = {visit.visit_id for visit in machine.visits.values() if any(item["new_state"] == State.CANCELLED.value for item in visit.transitions)}
    track_age_rows = []
    for tid in sorted(track_first_seen):
        age_seconds = track_last_seen[tid] - track_first_seen[tid] + 1.0 / fps
        track_age_rows.append({"track_id": tid, "first_time": track_first_seen[tid], "last_time": track_last_seen[tid],
            "detection_frames": track_frame_counts[tid], "duration_seconds": age_seconds,
            "below_min_track_age": age_seconds < float(config["min_track_age_seconds"]),
            "state_eligible": age_seconds >= float(config["min_track_age_seconds"])})
    if write_artifacts:
        write_outputs(output, rows, transitions)
        write_summary_chart(output, rows, duration)
        if render_video and (output / "annotated.mp4").is_file():
            create_customer_review_frames(output / "annotated.mp4", output, rows, fps)
        diagnostics = output / "diagnostics"; diagnostics.mkdir(parents=True, exist_ok=True)
        write_csv(diagnostics / "service_review.csv", ["visit_id", "person_label", "event_outcome", "counter_arrival_time", "served_time", "old_served_time", "speed_at_arrival_px_s", "closest_approach_px", "time_at_counter_seconds", "contact_type", "service_inferred", "review_flag"], review_rows)
        write_csv(diagnostics / "service_talk_spot_calibration.csv", ["track_id", "start_time", "end_time", "duration_seconds", "centroid_x", "centroid_y", "median_distance_to_anchor_px", "median_speed_px_s", "samples"], calibration_segments)
        timeline_fields = ["frame_index", "time", "track_id", "smoothed_foot_x", "smoothed_foot_y", "queue_signed_distance_px", "inside_queue_roi", "inside_service_roi", "expanded_roi_flag", "service_expanded_roi_flag", "service_signed_distance_px", "anchor_distance_px", "state", "visit_id", "detection_visible", "identity_ambiguous"]
        write_csv(diagnostics / "track_timeline.csv", timeline_fields, timeline_rows)
        write_csv(diagnostics / "panel_totals_by_frame.csv", ["frame_index", "time", *previous_totals.keys()], panel_total_rows)
        episode_fields = ["visit_id", "track_id", "start_time", "end_time", "duration_seconds", "outcome", "max_foot_point_distance_px", "total_foot_point_movement_px", "closest_anchor_distance_px", "abandonment_review", "suspect_false"]
        write_csv(diagnostics / "episodes.csv", episode_fields, episodes)
        service_fields = ["track_id", "total_seconds_inside_service", "longest_continuous_seconds_inside", "total_seconds_within_service_margin", "entries", "leaves", "state_at_end", "reason", "never_marked_served"]
        write_csv(diagnostics / "service_check.csv", service_fields, service_rows)
        write_csv(diagnostics / "staff_tracks.csv", ["track_id", "first_time", "last_time", "rule_fired", "first_x", "first_y", "first_region", "last_x", "last_y", "last_region", "ever_in_queue"],
                  [staff_observations[tid] | {"rule_fired": staff.staff_rules.get(tid, staff_observations[tid]["rule_fired"])} for tid in sorted(staff_observations)])
        queue_last_seen: dict[str, dict[str, Any]] = {}
        for item in timeline_rows:
            if item.get("visit_id"):
                prior = queue_last_seen.get(item["visit_id"])
                if prior is None or float(item["time"]) > float(prior["time"]):
                    queue_last_seen[item["visit_id"]] = item
        staff_near_queue_last = []
        for staff_point in (item for item in timeline_rows if item["state"] == "STAFF"):
            for visit_id, last in queue_last_seen.items():
                delta = float(staff_point["time"]) - float(last["time"])
                if abs(delta) <= 2.0:
                    staff_near_queue_last.append({"staff_track_id": staff_point["track_id"], "staff_time": staff_point["time"],
                        "staff_x": staff_point["smoothed_foot_x"], "staff_y": staff_point["smoothed_foot_y"],
                        "queue_visit_id": visit_id, "queue_last_track_id": last["track_id"], "queue_last_time": last["time"],
                        "time_delta_seconds": delta, "distance_px": hypot(float(staff_point["smoothed_foot_x"])-float(last["smoothed_foot_x"]),
                                                                              float(staff_point["smoothed_foot_y"])-float(last["smoothed_foot_y"]))})
        write_csv(diagnostics / "staff_near_queue_last_position.csv", ["staff_track_id", "staff_time", "staff_x", "staff_y", "queue_visit_id", "queue_last_track_id", "queue_last_time", "time_delta_seconds", "distance_px"], staff_near_queue_last)
        write_csv(diagnostics / "service_only.csv", ["track_id", "first_service_time", "last_service_time", "total_seconds_inside_service", "max_confidence", "queue_visit_created", "outcome"],
                  [{**valid_service_only_records[tid], "queue_visit_created": False, "outcome": "WALK_UP_AT_SERVICE"} for tid in sorted(valid_service_only_records)])
        write_csv(diagnostics / "track_age.csv", ["track_id", "first_time", "last_time", "detection_frames", "duration_seconds", "below_min_track_age", "state_eligible"], track_age_rows)
        write_csv(diagnostics / "counter_slot_timeline.csv", ["frame_index", "time", "owner_visit_id", "owner_track_id", "in_reach_visit_ids", "newly_acquired_visit_id"], counter_timeline_rows)
        event_by_track: dict[int, dict[str, Any]] = {}
        for row in rows:
            for event_tid in [int(row["track_id"]), *[int(value) for value in row.get("previous_track_ids", [])]]:
                event_by_track[event_tid] = row
        anchor_dwell_rows = []
        for tid, points in sorted(anchor_observations.items()):
            near = [item for item in points if item["within_2x_radius"]]
            if not near: continue
            event = event_by_track.get(tid)
            staff_zone_points = staff_zone_observations.get(tid, [])
            anchor_dwell_rows.append({"track_id": tid, "track_chain": ";".join(map(str, sorted({tid} | (set(map(int, event.get("previous_track_ids", []))) if event else set())))),
                "seconds_within_2x_radius": len(near)/fps, "seconds_within_service_radius": sum(item["inside_radius"] for item in near)/fps,
                "average_speed_within_2x_radius_px_s": sum(item["speed"] for item in near)/len(near),
                "event_outcome": event["event_outcome"] if event else ("STAFF" if tid in staff.staff_ids else "NO_SERVICE_EVENT"),
                "visit_type": event.get("visit_type", "") if event else "", "ever_in_staff_zone_or_margin": any(item["in_staff_zone_or_margin"] for item in staff_zone_points)})
        write_csv(diagnostics / "anchor_service_dwell.csv", ["track_id", "track_chain", "seconds_within_2x_radius", "seconds_within_service_radius", "average_speed_within_2x_radius_px_s", "event_outcome", "visit_type", "ever_in_staff_zone_or_margin"], anchor_dwell_rows)
        write_csv(diagnostics / "staff_163_197_positions.csv", ["track_id", "time", "x", "y", "in_staff_zone_or_margin"],
                  [{"track_id": tid, **item} for tid in (163, 197) for item in staff_zone_observations.get(tid, [])])
        service_event_rows = []
        counted_prior = []
        service_outcomes = {"SERVED", "CANCELLED_THEN_SERVED", "REACHED_SERVICE_BRIEF", "ABANDONED"}
        for row in sorted((item for item in rows if item["event_outcome"] in service_outcomes), key=lambda item: float(item.get("queue_exit_time") or item.get("abandonment_confirmed_time") or 0)):
            tid = int(row["track_id"]); chain = [tid, *[int(value) for value in row.get("previous_track_ids", [])]]
            event_time = float(row.get("queue_exit_time") or row.get("abandonment_confirmed_time") or row.get("pending_start_time") or 0)
            points = [point for alias in chain for point in anchor_observations.get(alias, [])]
            point = min(points, key=lambda item: abs(float(item["time"])-event_time)) if points else None
            previous = None; distance = None; gap = None
            if point is not None:
                options = []
                for old in counted_prior:
                    old_point = old.get("point")
                    dt = event_time - old["time"]
                    if old_point is None or dt < 0 or dt > 8.0: continue
                    dist = hypot(point["x"]-old_point["x"], point["y"]-old_point["y"])
                    if dist <= 100.0: options.append((dist, old, dt))
                if options:
                    distance, previous, gap = min(options, key=lambda item: item[0])
            service_event_rows.append({"visit_id": row["visit_id"], "event_time": event_time, "outcome": row["event_outcome"],
                "visit_type": row["visit_type"], "track_chain": ";".join(map(str, chain)),
                "dwell_seconds": row.get("service_dwell_seconds", 0.0), "average_speed_px_s": row.get("service_average_speed_px_s"),
                "previous_counted_event_within_8s_100px": previous["visit_id"] if previous else "",
                "previous_event_gap_seconds": gap, "previous_event_distance_px": distance,
                "suspected_duplicate": bool(previous)})
            if row["event_outcome"] in {"SERVED", "CANCELLED_THEN_SERVED"}:
                counted_prior.append({"visit_id": row["visit_id"], "time": event_time, "point": point})
        write_csv(diagnostics / "service_event_audit.csv", ["visit_id", "event_time", "outcome", "visit_type", "track_chain", "dwell_seconds", "average_speed_px_s", "previous_counted_event_within_8s_100px", "previous_event_gap_seconds", "previous_event_distance_px", "suspected_duplicate"], service_event_rows)
        starts = {tid: track_first_seen.get(tid, 0.0) for tid in track_first_seen}
        all_track_ids = sorted(anchor_observations)
        pair_rows = []
        for current_tid in all_track_ids:
            current_start = starts.get(current_tid, 0.0)
            current_points = anchor_observations[current_tid]
            current_first = min(current_points, key=lambda item: item["time"])
            for previous_tid in all_track_ids:
                if current_tid == previous_tid or starts.get(previous_tid, 0.0) >= current_start: continue
                previous_points = anchor_observations[previous_tid]
                previous_last = max(previous_points, key=lambda item: item["time"])
                gap = max(0.0, current_start - float(track_last_seen.get(previous_tid, current_start)))
                dist = hypot(current_first["x"]-previous_last["x"], current_first["y"]-previous_last["y"])
                if gap <= float(config["track_merge_seconds"]) and dist <= float(config["track_merge_px"]):
                    pair_rows.append({"previous_track_id": previous_tid, "new_track_id": current_tid, "gap_seconds": gap,
                        "distance_px": dist, "previous_first_time": starts.get(previous_tid), "previous_last_time": track_last_seen.get(previous_tid),
                        "new_first_time": current_start, "suspected_duplicate": True,
                        "previous_outcome": event_by_track.get(previous_tid, {}).get("event_outcome", ""),
                        "new_outcome": event_by_track.get(current_tid, {}).get("event_outcome", "")})
        write_csv(diagnostics / "nearby_identity_pairs.csv", ["previous_track_id", "new_track_id", "gap_seconds", "distance_px", "previous_first_time", "previous_last_time", "new_first_time", "suspected_duplicate", "previous_outcome", "new_outcome"], pair_rows)
        write_csv(diagnostics / "identity_recovery_links.csv", ["old_track_id", "new_track_id", "timestamp", "visit_id"], recovery.links)
        write_csv(diagnostics / "reassociation_log.csv", ["old_track_id", "new_track_id", "timestamp", "distance_px", "appearance_similarity", "direction_angle_degrees", "accepted", "decision", "visit_id"], recovery.decisions)
        write_csv(diagnostics / "multi_live_visit_tracks.csv", ["frame_index", "time", "visit_id", "track_ids"], multi_live_visit_frames)
        write_csv(diagnostics / "duplicate_boxes.csv", ["frame_index", "time", "kept_track_id", "suppressed_track_id", "iou", "containment"], duplicate_rows)
        write_csv(diagnostics / "offscreen_tracks.csv", ["track_id", "first_time", "last_time", "foot_x", "foot_y", "foot_status"], list(offscreen_rows.values()))
        ambiguity_rows = [{"track_id": tid, "ambiguous_frames": count, "ambiguous_seconds": count / fps,
                           "visit_ids": ";".join(sorted(ambiguity_visits[tid])),
                           "over_2_seconds": count / fps > 2.0}
                          for tid, count in sorted(ambiguity_frames.items())]
        write_csv(diagnostics / "ambiguity_durations.csv", ["track_id", "ambiguous_frames", "ambiguous_seconds", "visit_ids", "over_2_seconds"], ambiguity_rows)
        nonserved_rows = []
        outcome_by_visit = {row["visit_id"]: row for row in rows if row["event_outcome"] not in {"SERVED", "CANCELLED_THEN_SERVED"}}
        for visit_id, event in outcome_by_visit.items():
            points = [item for item in timeline_rows if item["visit_id"] == visit_id]
            if not points: continue
            event_end = event["abandonment_confirmed_time"] if event["abandonment_confirmed_time"] is not None else event["queue_exit_time"]
            last_time = float(event_end if event_end is not None else max(item["time"] for item in points))
            first_time = max(float(event["queue_entry_time"] or 0), last_time - 10.0)
            aliases = sorted({int(item["track_id"]) for item in points} | {int(value) for value in event.get("previous_track_ids", [])})
            for item in points:
                if first_time - 1e-6 <= float(item["time"]) <= last_time + 1e-6:
                    nonserved_rows.append({"visit_id": visit_id, "time": item["time"], "track_ids": ";".join(map(str, aliases)),
                        "track_id": item["track_id"], "foot_x": item["smoothed_foot_x"], "foot_y": item["smoothed_foot_y"],
                        "state": item["state"], "ambiguous": item.get("identity_ambiguous", False), "observation_kind": "DETECTED" if item["detection_visible"] else "HELD_TRACK_STATE"})
            latest = max(points, key=lambda item: float(item["time"]))
            latest_time = float(latest["time"])
            if last_time - latest_time > 1.0 / fps:
                held_state = "PENDING_ABANDONMENT" if event["event_outcome"] == "ABANDONED" else str(latest["state"])
                for held_time in np.arange(latest_time + 1.0 / fps, last_time + 1e-6, 1.0 / fps):
                    if held_time < first_time: continue
                    nonserved_rows.append({"visit_id": visit_id, "time": float(held_time), "track_ids": ";".join(map(str, aliases)),
                        "track_id": latest["track_id"], "foot_x": latest["smoothed_foot_x"], "foot_y": latest["smoothed_foot_y"],
                        "state": held_state, "ambiguous": latest.get("identity_ambiguous", False), "observation_kind": "LAST_POINT_HELD"})
        nonserved_rows.sort(key=lambda row: (row["visit_id"], float(row["time"])))
        write_csv(diagnostics / "nonserved_last10_seconds.csv", ["visit_id", "time", "track_ids", "track_id", "foot_x", "foot_y", "state", "ambiguous", "observation_kind"], nonserved_rows)
        (diagnostics / "track_id_count_note.txt").write_text(
            "Staff and walk-up diagnostics count detector track IDs, not unique real people. Duplicate suppression is reported in duplicate_boxes.csv.\n",
            encoding="utf-8")
        near_pending = []
        for episode in episodes:
            points = [item for item in timeline_rows if item["visit_id"] == episode["visit_id"]]
            if points:
                nearest = min(points, key=lambda item: abs(float(item["time"]) - episode["start_time"]))
                if float(nearest["service_signed_distance_px"]) <= float(config["service_margin_px"]):
                    near_pending.append({**episode, "service_distance_px_at_start": nearest["service_signed_distance_px"]})
        write_csv(diagnostics / "pending_near_service.csv", episode_fields + ["service_distance_px_at_start"], near_pending)
        never_served_near = [item for item in service_rows if item["never_marked_served"] and item["total_seconds_within_service_margin"] > float(config["min_service_seconds"])]
        write_csv(diagnostics / "near_service_not_served.csv", service_fields, never_served_near)
        write_ground_truth_template(Path(config["ground_truth_path"]), track_start_end)
        if render_video:
            create_review_material(video, output, fps, frame_index, transitions)
    summary = {"frames": frame_index, "fps": fps, "duration": duration, "visits": len(rows), "served": served_count,
               "abandoned": abandoned_count, "cancelled_visits": len(cancelled_visits), "pending_at_end": pending_count,
               "pending_episodes": len(episodes), "suspect_false": sum(row["suspect_false"] == "SUSPECT_FALSE" for row in episodes),
               "unique_track_ids": len(track_age_rows), "short_track_ids": sum(row["below_min_track_age"] for row in track_age_rows),
               "identity_recovery_links": len(recovery.links), "staff_track_count": len(staff_observations),
               "service_only_count": len(valid_service_only_records), "service_check": service_rows,
               "episodes": episodes, "events": rows, "transitions": transitions}
    print(f"YOLO settings: model={config['model_name']} input_size={config['inference_size']} confidence_threshold={config['confidence_threshold']}")
    print(f"Run complete: frames={frame_index}, fps={fps:.3f}, frame_size={width}x{height}")
    if render_video:
        print(f"Annotated output: {width}x{height} at {fps:.3f} fps; codec={writer_codec}; bytes={(output / 'annotated.mp4').stat().st_size if (output / 'annotated.mp4').exists() else 0}")
    print(f"Tracks: {summary['unique_track_ids']} unique | shorter than {config['min_track_age_seconds']:.2f}s: {summary['short_track_ids']} | identity recovery links: {summary['identity_recovery_links']}")
    print("STAFF tracks:")
    for item in (staff_observations[tid] for tid in sorted(staff_observations)):
        print(f"  ID {item['track_id']}: {item['first_time']:.2f}-{item['last_time']:.2f}s rule={item['rule_fired']} first=({item['first_x']:.0f},{item['first_y']:.0f})/{item['first_region']} last=({item['last_x']:.0f},{item['last_y']:.0f})/{item['last_region']} ever_IN_QUEUE={item['ever_in_queue']}")
    reached_desk = [item for item in service_rows if item["state_at_end"] == State.AT_SERVICE_POINT.value]
    print("Service durations for tracks that reached the desk:", [(item["track_id"], round(item["total_seconds_inside_service"], 2)) for item in reached_desk])
    print(f"Walk-up at service track count: {len(valid_service_only_records)}; listed in service_only.csv (per track id, not per person)")
    print("Service-only tracks:", [(tid, round(float(row["first_service_time"]), 2), round(float(row["total_seconds_inside_service"]), 2)) for tid, row in sorted(valid_service_only_records.items())])
    print(f"Visits: {len(rows)} | served: {served_count} | abandoned: {abandoned_count} | cancelled visits: {len(cancelled_visits)} | pending at end: {pending_count}")
    print(f"Pending episodes: {len(episodes)} | suspect false: {summary['suspect_false']}")
    print("Service check (all tracks observed within service margin):")
    for item in service_rows:
        print(f"  track {item['track_id']}: inside={item['total_seconds_inside_service']:.2f}s longest={item['longest_continuous_seconds_inside']:.2f}s margin={item['total_seconds_within_service_margin']:.2f}s entries={item['entries']} leaves={item['leaves']} end={item['state_at_end']} reason={item['reason']}")
    print("Never served despite >min_service_seconds near desk:", [item["track_id"] for item in service_rows if item["never_marked_served"] and item["total_seconds_within_service_margin"] > float(config["min_service_seconds"])])
    print("Pending/abandoned episodes starting near service:", [(item["track_id"], item["outcome"], round(item["start_time"], 2)) for item in near_pending] if write_artifacts else [])
    print("Suspect-false episodes:", [(item["track_id"], item["outcome"], round(item["start_time"], 2)) for item in episodes if item["suspect_false"]])
    print("Re-association decisions (old,new,time,distance,appearance,direction,accepted,reason):")
    for item in recovery.decisions:
        print(f"  {item['old_track_id']} -> {item['new_track_id']} @ {item['timestamp']:.2f}s dist={item['distance_px']} app={item['appearance_similarity']} angle={item['direction_angle_degrees']} accepted={item['accepted']} {item['decision']}")
    print("Visits with more than one live track:", multi_live_visit_frames)
    print("Duplicate boxes suppressed:", len(duplicate_rows), "(details: diagnostics/duplicate_boxes.csv)")
    print("Ambiguous frames per track:", [(row["track_id"], row["ambiguous_frames"], round(row["ambiguous_seconds"], 2)) for row in ambiguity_rows] if write_artifacts else [])
    print("Tracks ambiguous >2 seconds:", [row["track_id"] for row in ambiguity_rows if row["over_2_seconds"]] if write_artifacts else [])
    print("Track-lost visits:", [row["visit_id"] for row in rows if row["event_outcome"] == "TRACK_LOST"])
    print("Service inferred after track loss:", [(row["visit_id"], row["track_id"]) for row in rows if row.get("service_inferred")])
    if write_artifacts:
        anchor_rows = list(csv.DictReader((output / "diagnostics" / "anchor_service_dwell.csv").open(encoding="utf-8")))
        print("Anchor dwell for every track within 2x service_radius_px (time, avg speed, outcome):")
        for item in anchor_rows:
            print(f"  track {item['track_id']} chain={item['track_chain']} near={float(item['seconds_within_2x_radius']):.2f}s in_radius={float(item['seconds_within_service_radius']):.2f}s avg_speed={float(item['average_speed_within_2x_radius_px_s']):.1f}px/s outcome={item['event_outcome']}")
        audit_rows = list(csv.DictReader((output / "diagnostics" / "service_event_audit.csv").open(encoding="utf-8")))
        print("Served/brief/abandoned event audit (visit, dwell, speed, type, id chain, prior event, duplicate flag):")
        for item in audit_rows:
            print(f"  {item['visit_id']} {item['outcome']} dwell={item['dwell_seconds']} speed={item['average_speed_px_s']} type={item['visit_type']} tracks=[{item['track_chain']}] prior={item['previous_counted_event_within_8s_100px']} suspected_duplicate={item['suspected_duplicate']}")
        for tid in (163, 197):
            points = staff_zone_observations.get(tid, [])
            if points:
                first_point, last_point = points[0], points[-1]
                print(f"Staff-zone check ID {tid}: {first_point['time']:.2f}- {last_point['time']:.2f}s first=({first_point['x']:.1f},{first_point['y']:.1f}) last=({last_point['x']:.1f},{last_point['y']:.1f}) ever_inside_or_margin={any(p['in_staff_zone_or_margin'] for p in points)} classified_staff={tid in staff.staff_ids}")
        pairs = list(csv.DictReader((output / "diagnostics" / "nearby_identity_pairs.csv").open(encoding="utf-8")))
        print("Nearby short-gap ID pairs (possible duplicates; distance/time heuristic only):", [(item["previous_track_id"], item["new_track_id"], round(float(item["gap_seconds"]), 2), round(float(item["distance_px"]), 1)) for item in pairs])
        print("Visit 0006 effect: inspect its actual service event and the staff overlap in service_event_audit.csv / staff_163_197_positions.csv; staff IDs are excluded from direct and served rows.")
    print(f"Processing time: {time.perf_counter() - started:.2f}s; output: {output.resolve()}")
    return summary

def main() -> None:
    """Check CLI inputs and show actionable startup errors."""
    parser = argparse.ArgumentParser(description="Queue abandonment detection")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--save-tracks", action="store_true", help="save every frame's detections to output/tracks_cache.json")
    parser.add_argument("--load-tracks", action="store_true", help="skip YOLO and replay output/tracks_cache.json")
    parser.add_argument("--no-video-output", action="store_true", help="skip rendering during cached parameter sweeps")
    parser.add_argument("--strict-service", action="store_true", help="use locked-staff, single-counter, observed-dwell replay")
    parser.add_argument("--output-dir", default=None, help="override output directory (strict replay only)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        video = args.video or config["video_source"]
        if args.strict_service:
            if not args.load_tracks:
                parser.error("--strict-service requires --load-tracks; YOLO inference is not used")
            from .fixed_replay import replay as strict_replay
            strict_replay(args.config, video, args.output_dir)
            return
        run(args.config, video, save_tracks=args.save_tracks,
            load_tracks=args.load_tracks, render_video=not args.no_video_output)
    except Exception as exc:
        parser.exit(2, f"Startup or processing error: {exc}\n")

if __name__ == "__main__":
    main()
