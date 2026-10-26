"""Load and validate the application's YAML configuration."""
from pathlib import Path
from typing import Any
import yaml

TIME_KEYS = ("min_join_seconds", "exit_debounce_seconds", "min_service_seconds", "service_min_dwell_sec", "service_exit_grace_sec", "service_flicker_grace_sec", "min_track_sec", "min_queue_sec", "reassoc_gap_sec", "reentry_sec", "staff_lock_sec", "staff_ghost_sec", "service_touch_seconds", "ring_service_seconds", "direct_confirm_seconds", "staff_initial_dwell_seconds", "staff_dwell_seconds", "grace_period_seconds", "lost_track_timeout_seconds", "max_occlusion_seconds", "evaluation_tolerance_seconds", "service_time_tolerance_seconds", "alert_seconds", "synthetic_fps", "synthetic_duration_seconds", "service_grace_seconds", "brief_upgrade_seconds", "service_reach_grace_seconds", "short_contact_seconds")

def load_config(path: str | Path, frame_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Read YAML and reject malformed or unsafe configuration values."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Config not found: {source}. Create config/config.yaml or pass a valid --config path.")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must contain a YAML mapping.")
    data.setdefault("queue_exit_margin_px", float(data.get("exit_margin_px", 0.0)))
    data.setdefault("service_margin_px", 40.0)
    data.setdefault("service_use_box_center_fallback", False)
    data.setdefault("service_grace_seconds", 3.0)
    data.setdefault("brief_upgrade_seconds", 5.0)
    data.setdefault("suspect_move_px", 40.0)
    data.setdefault("staff_min_dwell_seconds", 16.0)
    data.setdefault("min_track_age_seconds", 0.5)
    data.setdefault("max_occlusion_seconds", 1.5)
    data.setdefault("appearance_min_similarity", 0.6)
    data.setdefault("max_link_direction_angle", 110.0)
    data.setdefault("overlap_iou_threshold", 0.15)
    data.setdefault("staff_initial_dwell_seconds", 2.0)
    data.setdefault("staff_zone_margin_px", 18.0)
    data.setdefault("staff_zone_min_seconds", 1.0)
    data.setdefault("service_radius_px", 50.0)
    data.setdefault("talk_radius_px", float(data["service_radius_px"]))
    data.setdefault("reach_protect_px", 80.0)
    data.setdefault("service_time_tolerance_seconds", 3.0)
    data.setdefault("ring_service_seconds", 2.5)
    data.setdefault("talk_radius_px", 30.0)
    data.setdefault("reach_protect_px", 80.0)
    data.setdefault("service_touch_seconds", 0.5)
    data.setdefault("direct_approach_px", 250.0)
    data.setdefault("direct_confirm_seconds", 1.5)
    data.setdefault("service_max_speed", 25.0)
    data.setdefault("service_reach_px", 100.0)
    data.setdefault("service_reach_grace_seconds", 0.5)
    data.setdefault("max_counter_hold_seconds", 30.0)
    data.setdefault("queue_entry_backdated_seconds", 1.0)
    data.setdefault("short_contact_seconds", 1.0)
    data.setdefault("staff_dwell_seconds", 8.0)
    data.setdefault("max_simultaneous_service", 1)
    data.setdefault("track_merge_px", 75.0)
    data.setdefault("track_merge_seconds", 2.0)
    data.setdefault("duplicate_containment_threshold", 0.8)
    data.setdefault("staff_roi", data.get("queue_rois", [{}])[0].get("staff_zone", []) if data.get("queue_rois") else [])
    data.setdefault("staff_lock_sec", 2.0)
    data.setdefault("staff_ghost_sec", 3.0)
    data.setdefault("staff_max_bbox_height_px", 75.0)
    data.setdefault("staff_max_motion_px", 25.0)
    data.setdefault("service_min_dwell_sec", 2.0)
    data.setdefault("service_exit_grace_sec", 0.75)
    data.setdefault("min_track_sec", 0.5)
    data.setdefault("min_queue_sec", 1.0)
    data.setdefault("reassoc_gap_sec", 1.5)
    data.setdefault("reassoc_dist_px", float(data.get("reassociate_distance_px", 80.0)))
    data.setdefault("max_reassoc_links_per_visit", 1)
    data.setdefault("reentry_sec", 10.0)
    data.setdefault("service_flicker_grace_sec", 0.5)
    data.setdefault("reassoc_overlap_iou", 0.5)
    data.setdefault("reassoc_appearance_min_similarity", 0.7)
    data.setdefault("reassoc_min_birth_gap_sec", 0.1)
    data.setdefault("reassoc_min_height_ratio", 1.2)
    for key in TIME_KEYS:
        if key not in data or float(data[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero.")
    if data.get("tracker") not in {"bytetrack", "botsort"}:
        raise ValueError("tracker must be one of: bytetrack, botsort.")
    if data.get("device") not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: cpu, cuda.")
    if not data.get("queue_rois"):
        raise ValueError("At least one queue/service ROI pair is required.")
    for item in data["queue_rois"]:
        for name in ("queue", "service"):
            polygon = item.get(name, [])
            if len(polygon) < 3:
                raise ValueError(f"ROI '{item.get('name', '?')}.{name}' needs at least 3 points.")
            for point in polygon:
                if len(point) != 2 or any(not isinstance(v, (int, float)) for v in point):
                    raise ValueError(f"ROI '{item.get('name', '?')}.{name}' points must be [x, y] numbers.")
                if frame_size:
                    width, height = frame_size
                    if not (0 <= point[0] < width and 0 <= point[1] < height):
                        raise ValueError(f"ROI '{item.get('name', '?')}.{name}' has a point outside the {width}x{height} video frame.")
        for name in ("staff_zone",):
            polygon = item.get(name, [])
            polygons = [] if not polygon else ([polygon] if len(polygon) >= 3 and isinstance(polygon[0], (list, tuple)) and len(polygon[0]) == 2 and all(isinstance(v, (int, float)) for v in polygon[0]) else polygon)
            if any(len(area) < 3 for area in polygons):
                raise ValueError(f"ROI '{item.get('name', '?')}.{name}' polygons need at least 3 points.")
            if frame_size:
                width, height = frame_size
                if any(not (0 <= x < width and 0 <= y < height) for area in polygons for x, y in area):
                    raise ValueError(f"ROI '{item.get('name', '?')}.{name}' has a point outside the {width}x{height} video frame.")
    staff_roi = data["staff_roi"]
    if not isinstance(staff_roi, list) or len(staff_roi) < 3:
        raise ValueError("staff_roi needs at least 3 [x, y] points.")
    if frame_size:
        width, height = frame_size
        if any(len(point) != 2 or not (0 <= point[0] < width and 0 <= point[1] < height) for point in staff_roi):
            raise ValueError(f"staff_roi has a point outside the {width}x{height} video frame.")
    if int(data.get("trajectory_smoothing_window", 0)) < 1:
        raise ValueError("trajectory_smoothing_window must be at least 1.")
    if float(data.get("max_join_speed", 0)) <= 0:
        raise ValueError("max_join_speed must be greater than zero.")
    if float(data.get("staff_max_speed", 0)) <= 0:
        raise ValueError("staff_max_speed must be greater than zero.")
    if float(data["service_max_speed"]) <= 0:
        raise ValueError("service_max_speed must be greater than zero.")
    if float(data["staff_max_bbox_height_px"]) <= 0 or float(data["staff_max_motion_px"]) < 0:
        raise ValueError("staff_max_bbox_height_px must be positive and staff_max_motion_px non-negative.")
    if float(data["reassoc_dist_px"]) <= 0:
        raise ValueError("reassoc_dist_px must be greater than zero.")
    if int(data["max_reassoc_links_per_visit"]) < 0:
        raise ValueError("max_reassoc_links_per_visit must be non-negative.")
    if not 0.0 <= float(data["reassoc_overlap_iou"]) <= 1.0 or not 0.0 <= float(data["reassoc_appearance_min_similarity"]) <= 1.0:
        raise ValueError("reassociation overlap and appearance thresholds must be between 0 and 1.")
    if float(data["staff_min_dwell_seconds"]) < 0 or float(data.get("min_track_age_seconds", -1)) < 0:
        raise ValueError("staff_min_dwell_seconds and min_track_age_seconds must be non-negative.")
    if int(data.get("inference_size", 0)) < 160:
        raise ValueError("inference_size must be at least 160 pixels.")
    ignored = data.get("ignore_track_ids", [])
    if not isinstance(ignored, list) or any(not isinstance(track_id, int) or isinstance(track_id, bool) for track_id in ignored):
        raise ValueError("ignore_track_ids must be a list of integer track IDs.")
    if any(float(data.get(key, -1)) < 0 for key in ("exit_margin_px", "queue_exit_margin_px", "service_margin_px", "suspect_move_px")) or float(data.get("reassociate_distance_px", 0)) <= 0 or float(data["talk_radius_px"]) <= 0 or float(data["reach_protect_px"]) <= 0:
        raise ValueError("ROI margins and suspect_move_px must be non-negative, and reassociate_distance_px must be greater than zero.")
    if not isinstance(data["service_use_box_center_fallback"], bool):
        raise ValueError("service_use_box_center_fallback must be true or false.")
    if not 0 <= float(data.get("confidence_threshold", -1)) <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not 0 <= float(data["appearance_min_similarity"]) <= 1:
        raise ValueError("appearance_min_similarity must be between 0 and 1.")
    if not 0 <= float(data["max_link_direction_angle"]) <= 180:
        raise ValueError("max_link_direction_angle must be between 0 and 180 degrees.")
    if not 0 <= float(data["overlap_iou_threshold"]) <= 1:
        raise ValueError("overlap_iou_threshold must be between 0 and 1.")
    return data
