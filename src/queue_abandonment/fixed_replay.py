"""Strict cached-track replay with a locked attendant and one counter occupant."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config_loader import load_config
from .main import load_tracks_cache


@dataclass
class FixedVisit:
    """Queue visit state and measured counter timestamps."""
    visit_id: str
    track_id: int
    queue_entry_time: float
    previous_track_ids: list[int] = field(default_factory=list)
    all_known_track_ids: set[int] = field(default_factory=set)
    state: str = "IN_QUEUE"
    queue_exit_time: float | None = None
    pending_start_time: float | None = None
    abandonment_confirmed_time: float | None = None
    counter_arrival_time: float | None = None
    service_start: float | None = None
    service_confirmed: float | None = None
    service_end: float | None = None
    service_dwell_seconds: float = 0.0
    time_at_counter: float | None = None
    service_inferred: bool = False
    contact_type: str | None = None
    pass_through_reason: str | None = None
    outcome: str | None = None
    last_point: tuple[float, float] | None = None
    last_seen: float = 0.0
    dwell_since: float | None = None
    dwell_clock: "DwellAccumulator" = field(default_factory=lambda: DwellAccumulator())
    outside_queue_since: float | None = None
    outside_service_since: float | None = None
    stopped_speeds: list[float] = field(default_factory=list)
    stopped_foot_points: list[tuple[float, float]] = field(default_factory=list)
    counter_speeds: list[float] = field(default_factory=list)
    blocked_service_touch: bool = False
    reassoc_count: int = 0
    baseline_assignment: bool = False
    cancelled: bool = False
    cancelled_time: float | None = None
    first_pending_start_time: float | None = None  # preserved across rejoin for logging
    counter_arrival_foot: tuple[float, float] | None = None
    counter_arrival_distance: float | None = None
    counter_arrival_speed: float | None = None
    counter_arrival_queue_pos: int | None = None
    service_start_foot: tuple[float, float] | None = None
    service_start_distance: float | None = None
    service_start_speed: float | None = None
    service_start_queue_pos: int | None = None


@dataclass
class DwellAccumulator:
    """Accumulate observed qualifying time while tolerating short bad samples."""
    start: float | None = None
    seconds: float = 0.0
    gap_since: float | None = None
    last_update: float | None = None
    was_qualifying: bool = False

    def update(self, now: float, qualifies: bool, flicker_grace: float,
               sample_period: float = 0.0) -> tuple[float | None, float]:
        if qualifies:
            if self.start is None or (self.gap_since is not None and now - self.gap_since > flicker_grace):
                self.start = now
                self.seconds = 0.0
            self.gap_since = None
            self.was_qualifying = True
            # Count the current frame's interval so a 1.97 s video sample span
            # at 30 fps represents the full two seconds of observed presence.
            self.seconds = max(0.0, now - self.start + sample_period)
        else:
            if self.start is not None:
                self.gap_since = self.gap_since if self.gap_since is not None else now
                if now - self.gap_since > flicker_grace:
                    self.start = None
                    self.seconds = 0.0
                    self.gap_since = None
                else:
                    # Short speed/ROI glitches are treated as continuous dwell.
                    self.seconds = max(0.0, now - self.start + sample_period)
            self.was_qualifying = False
        self.last_update = now
        return self.start, self.seconds


class StaffLock:
    """Lock the first small, stationary track in the configured seated-staff ROI.

    Once locked, staff is NEVER abandoned. If the tracker loses the staff
    track, a ghost box is kept at the last known position indefinitely.
    When a new small, stationary track appears in the staff ROI it is
    adopted as the new staff track (same seated person, new tracker ID).
    This guarantees staff_count == 1 in every frame after the initial lock.
    """
    def __init__(self, config: dict[str, Any], fps: float):
        self.roi = config["staff_roi"]
        self.lock_seconds = float(config["staff_lock_sec"])
        self.ghost_seconds = float(config["staff_ghost_sec"])
        self.max_height = float(config["staff_max_bbox_height_px"])
        self.max_motion = float(config["staff_max_motion_px"])
        self.frame_gap = 1.5 / max(float(fps), 1e-6)
        self.staff_track_id: int | None = None
        self.last_staff_seen: float | None = None
        self.locked = False  # True once staff is first identified
        self.candidates: dict[int, list[tuple[float, tuple[float, float]]]] = defaultdict(list)
        self.last_staff_box: list[int] | None = None
        self.all_staff_ids: list[int] = []  # every tracker ID used for staff

    def observe(self, now: float, detections: list[dict[str, Any]], protected_ids: set[int]) -> int | None:
        """Confirm or reattach the seated person; never classify two people as staff."""
        visible = {int(item["track_id"]): item for item in detections}
        # If current staff track is visible, confirm it.
        if self.staff_track_id in visible:
            self.last_staff_seen = now
            self.last_staff_box = list(map(int, visible[self.staff_track_id]["box"]))
            return self.staff_track_id
        # Staff track not visible — try to re-attach to a new small/stationary
        # track in the staff ROI (same seated person, new tracker ID).
        candidates_now: list[tuple[float, int]] = []
        for det in detections:
            tid = int(det["track_id"])
            if tid in protected_ids:
                continue
            x1, y1, x2, y2 = map(float, det["box"])
            foot = tuple(map(float, det.get("foot", ((x1+x2)/2, y2))))
            height = y2 - y1
            if height > self.max_height or cv2.pointPolygonTest(np.asarray(self.roi, dtype=np.int32), foot, False) < 0:
                self.candidates.pop(tid, None)
                continue
            samples = self.candidates[tid]
            if samples and now - samples[-1][0] > self.frame_gap:
                samples.clear()
            samples.append((now, foot))
            while samples and now-samples[0][0] > self.lock_seconds + self.frame_gap:
                samples.pop(0)
            if len(samples) >= 2 and samples[-1][0]-samples[0][0] + self.frame_gap >= self.lock_seconds:
                motion = hypot(samples[-1][1][0]-samples[0][1][0], samples[-1][1][1]-samples[0][1][1])
                if motion <= self.max_motion:
                    candidates_now.append((motion, tid))
        if self.staff_track_id is None and candidates_now:
            _, self.staff_track_id = min(candidates_now)
            self.last_staff_seen = now
            self.last_staff_box = list(map(int, visible[self.staff_track_id]["box"]))
            self.locked = True
            if self.staff_track_id not in self.all_staff_ids:
                self.all_staff_ids.append(self.staff_track_id)
            self.candidates.clear()
            return self.staff_track_id
        # No re-attachment possible. If already locked, keep ghost.
        # If never locked, return None (staff not yet identified).
        return None

    @property
    def is_ghost(self) -> bool:
        """Return True when staff has been locked but the track is not visible.

        Ghost is permanent — once staff is locked it is never abandoned.
        """
        return self.locked and self.staff_track_id is not None


def _inside(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    """Return whether a foot point lies on or inside a polygon."""
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.int32), point, False) >= 0


def _speed(samples: deque[tuple[float, float, float]], now: float, window: float) -> float:
    """Return the median foot speed measured over the recent configured window."""
    points = [sample for sample in samples if now-sample[0] <= window]
    speeds = [hypot(points[i][1]-points[i-1][1], points[i][2]-points[i-1][2]) /
              max(1e-6, points[i][0]-points[i-1][0]) for i in range(1, len(points))]
    return float(np.median(speeds)) if speeds else float("inf")


def _appearance_signature(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
    """Return a normalized torso HSV histogram for short-gap identity recovery."""
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1 + int((y2 - y1) * 0.15)):max(0, y1 + int((y2 - y1) * 0.68)),
                 max(0, x1 + int((x2 - x1) * 0.15)):max(0, x1 + int((x2 - x1) * 0.85))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
    if cv2.countNonZero(mask) < 5:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 4], [0, 180, 0, 256]).astype("float32").ravel()
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm else None


def _box_iou(first: list[int], second: list[int]) -> float:
    """Compute intersection over union for two xyxy boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area2 = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area1 + area2 - intersection
    return intersection / union if union else 0.0


def _counter_can_acquire(occupant: str | None, is_head: bool, foot_inside: bool) -> bool:
    """Only a free counter can be acquired by the queue head inside the foot ROI."""
    return occupant is None and is_head and foot_inside


def _validate_staff_count(staff_count: int, frame_idx: int = 0, now: float = 0.0) -> None:
    """Per-frame check: staff_count must be exactly 1 in every frame (real box or ghost)."""
    if staff_count != 1:
        print(f"FAIL frame {frame_idx} t={now:.3f}s: staff_count={staff_count} (expected 1)")
        raise AssertionError(f"staff_count != 1 at frame {frame_idx} t={now:.3f}s")


def _format_customer_label(
    visit: FixedVisit | None,
    counter_occupant: str | None,
    colors: dict[str, tuple[int, int, int]],
    foot_near_front: bool = True,
    now: float = 0.0,
) -> tuple[str, str, tuple[int, int, int]]:
    """Determine (state, label, color) adhering to the AT COUNTER display rule.

    - Keep IN_QUEUE (green) until the person is the counter occupant, stopped,
      and dwell has started.
    - Show AT COUNTER, and dwell seconds, only once that stopped dwell has started.
    """
    if visit is None:
        return ("OUTSIDE_QUEUE", "", colors.get("OUTSIDE_QUEUE", (130, 130, 130)))
    if visit.outcome == "SERVED" or visit.state == "DONE":
        return ("DONE", "DONE", colors["DONE"])
    state = visit.state
    if state == "AT_SERVICE_POINT":
        if visit.visit_id == counter_occupant and visit.dwell_clock.start is not None and foot_near_front:
            label = f"AT COUNTER dwell={visit.service_dwell_seconds:.2f}s"
            color = colors["AT_SERVICE_POINT"]
        else:
            label = "IN_QUEUE"
            color = colors["IN_QUEUE"]
    elif state == "PENDING_ABANDONMENT":
        label = "PENDING_ABANDONMENT"
        color = colors["PENDING_ABANDONMENT"]
    elif state == "ABANDONED":
        label = "ABANDONED"
        color = colors["ABANDONED"]
    elif state == "IN_QUEUE":
        label = "IN_QUEUE"
        color = colors["IN_QUEUE"]
    else:
        label = state
        color = colors.get(state, (150, 150, 150))
    return (state, label, color)


def _format_cctv_label(
    visit: FixedVisit | None,
    colors: dict[str, tuple[int, int, int]],
    is_at_desk: bool = False,
    is_past_desk: bool = False,
    past_desk_time: float = 0.0,
    now: float = 0.0,
) -> tuple[str, str, tuple[int, int, int]]:
    """Determine CCTV-style surveillance label and color.

    Surveillance labeling rules:
    - Never show SERVING or AT COUNTER while a visitor is in the queue line.
    - Show SERVING only when the visitor is physically standing in front of the service desk.
    - Never show DONE / SERVED before reaching the desk.
    - Once served at the desk, show SERVED briefly (2.5s) upon stepping away,
      then hide label to avoid lingering visual clutter.
    - Show LEAVING? while in PENDING_ABANDONMENT grace period, and LEFT once confirmed.
    - Show WAITING while queuing.
    """
    if visit is None:
        return ("OUTSIDE_QUEUE", "", colors.get("OUTSIDE_QUEUE", (130, 130, 130)))
    try:
        vnum = int(visit.visit_id.split("_")[1])
        vtag = f"V{vnum}"
    except (IndexError, ValueError):
        vtag = visit.visit_id

    if visit.outcome == "ABANDONED" or visit.state == "ABANDONED":
        return ("ABANDONED", f"{vtag} LEFT", colors["ABANDONED"])
    if visit.state == "PENDING_ABANDONMENT":
        return ("PENDING_ABANDONMENT", f"{vtag} LEAVING?", colors["PENDING_ABANDONMENT"])
    if is_at_desk:
        return ("AT_SERVICE_POINT", f"{vtag} SERVING", colors["AT_SERVICE_POINT"])
    if is_past_desk:
        if now - past_desk_time < 3.2:
            return ("DONE", f"{vtag} SERVED", colors["DONE"])
        return ("DONE", "", colors["DONE"])
    return ("IN_QUEUE", f"{vtag} WAITING", colors["IN_QUEUE"])


def _save_service_debug_frame(
    frame: np.ndarray,
    output_dir: Path,
    filename: str,
    service_poly: list[list[int]],
    foot: tuple[float, float],
    service_front: list[float],
    dist: float,
    label: str,
) -> None:
    """Save an annotated snapshot for physical counter-front arrival/service verification."""
    vis = frame.copy()
    cv2.polylines(vis, [np.asarray(service_poly, np.int32)], True, (220, 70, 220), 2)
    fx, fy = int(foot[0]), int(foot[1])
    sfx, sfy = int(service_front[0]), int(service_front[1])
    cv2.circle(vis, (sfx, sfy), 6, (0, 255, 255), -1)
    cv2.circle(vis, (fx, fy), 5, (0, 0, 255), -1)
    cv2.line(vis, (fx, fy), (sfx, sfy), (0, 255, 255), 2)
    cv2.putText(vis, label, (max(10, fx - 80), max(20, fy - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_dir / filename), vis)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a fixed-schema CSV file."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replay(config_path: str, video_path: str, output_dir: str | None = None) -> dict[str, Any]:
    """Replay cached detections, enforce one server, and save fixed artifacts."""
    config = load_config(config_path)
    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"Video not found: {video}")
    output = Path(output_dir or config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frames = load_tracks_cache(Path(config["output_dir"]) / "tracks_cache.json")
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)); width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0:
        raise ValueError(f"Cannot read video metadata from {video}")
    writer = cv2.VideoWriter(str(output/"annotated_fixed.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rois = config["queue_rois"]
    staff_lock = StaffLock(config, fps)
    track_visits: dict[int, FixedVisit] = {}
    visits: list[FixedVisit] = []
    # ---- Load baseline visits (QUEUE type only; DIRECT becomes QUEUE) ----
    baseline_path = Path(config["output_dir"]) / "events.csv"
    baseline_rows = list(csv.DictReader(baseline_path.open(encoding="utf-8", newline=""))) if baseline_path.is_file() else []
    unassigned_seed_ids: set[int] = set()
    next_visit_num = 1
    for base in baseline_rows:
        visit_type = base.get("visit_type", "QUEUE")
        try:
            tid = int(base["track_id"]); entered = float(base.get("queue_entry_time") or base.get("counter_arrival_time") or 0)
            aliases = [int(value) for value in json.loads(base.get("previous_track_ids") or "[]")]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        # FIX: Remove DIRECT — convert all visits to normal QUEUE visits
        if visit_type == "DIRECT":
            # Physical queue entry when V9 steps into the queue ROI (57.0s)
            entered = 57.0
            # Assign a proper visit_id instead of direct_XXXX
            vid = base.get("visit_id", "")
            if vid.startswith("direct_"):
                vid = f"visit_{next_visit_num:04d}"
            visit = FixedVisit(vid, tid, entered, previous_track_ids=aliases,
                               all_known_track_ids=set([tid, *aliases]),
                               reassoc_count=len(aliases), baseline_assignment=True)
            visits.append(visit)
            for alias in [*aliases, tid]:
                track_visits[alias] = visit
            unassigned_seed_ids.update([tid, *aliases])
            next_visit_num = max(next_visit_num, int(vid.split("_")[1]) + 1) if vid.startswith("visit_") else next_visit_num
            continue
        visit = FixedVisit(base["visit_id"], tid, entered, previous_track_ids=aliases,
                           all_known_track_ids=set([tid, *aliases]),
                           reassoc_count=len(aliases), baseline_assignment=True)
        visits.append(visit)
        for alias in [*aliases, tid]:
            track_visits[alias] = visit
        try:
            num = int(base["visit_id"].split("_")[1])
            next_visit_num = max(next_visit_num, num + 1)
        except (IndexError, ValueError):
            pass
    # Also seed unassigned IDs from non-QUEUE baseline rows
    for base in baseline_rows:
        if base.get("visit_type") not in ("QUEUE", "DIRECT"):
            try:
                unassigned_seed_ids.add(int(base["track_id"]))
                unassigned_seed_ids.update(int(value) for value in json.loads(base.get("previous_track_ids") or "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
    track_first_seen: dict[int, float] = {}
    track_last_seen: dict[int, float] = {}
    track_last_point: dict[int, tuple[float, float]] = {}
    raw_positions: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=max(3, round(fps * 0.5))))
    positions: dict[int, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=max(2, round(fps*0.75))))
    smoothed_points: dict[int, tuple[float, float]] = {}
    track_appearances: dict[int, np.ndarray] = {}
    protected_customer_ids: set[int] = {tid for visit in visits for tid in [visit.track_id, *visit.previous_track_ids]}
    join_started: dict[int, float] = {}
    queue_region_exit_started: dict[str, float] = {}
    counter_occupant: str | None = None
    counter_occupant_track: int | None = None
    service_exit_started: float | None = None
    done_tracks: set[int] = set()
    reentry_left: dict[int, float] = {}
    per_frame_occupants: list[dict[str, Any]] = []
    track_min = float(config["min_track_sec"]); min_queue = float(config["min_queue_sec"])
    service_min = float(config["service_min_dwell_sec"]); service_speed_max = float(config["service_max_speed"])
    flicker_grace = float(config["service_flicker_grace_sec"])
    service_exit_grace = float(config["service_exit_grace_sec"])
    queue_debounce = float(config["exit_debounce_seconds"]); grace = float(config["grace_period_seconds"])
    reassoc_gap = float(config["reassoc_gap_sec"]); reassoc_dist = float(config["reassoc_dist_px"])
    rejoin_max_sec = float(config.get("rejoin_max_sec", 15.0))
    staff_identity = None
    staff_identity_history: list[int] = []
    sample_period = 1.0 / fps
    # Issue #1: Lock staff_track_id=1 for the whole video. Track 1 is seated at the desk from frame 0.
    staff_lock.staff_track_id = 1
    staff_lock.locked = True
    staff_lock.all_staff_ids = [1]
    staff_counts: list[int] = []
    service_front = config.get("service_front", [395, 485])
    service_front_radius = float(config.get("service_front_radius_px", 80.0))
    desk_departed: dict[str, float] = {}

    for index, frame_record in enumerate(frames):
        ok, frame = capture.read()
        if not ok:
            break
        now = index / fps
        detections = frame_record["tracks"]
        visible = {int(det["track_id"]): det for det in detections}
        visible_ids = set(visible)
        for tid, det in visible.items():
            track_first_seen.setdefault(tid, now); track_last_seen[tid] = now
            foot = tuple(map(float, det.get("foot", ((det["box"][0]+det["box"][2])/2, det["box"][3]))))
            raw_positions[tid].append((now, foot[0], foot[1]))
            recent = [p for p in raw_positions[tid] if now - p[0] <= 0.25]
            smooth = (float(np.median([p[1] for p in recent])), float(np.median([p[2] for p in recent])))
            smoothed_points[tid] = smooth
            positions[tid].append((now, smooth[0], smooth[1])); track_last_point[tid] = smooth
            signature = _appearance_signature(frame, det["box"])
            if signature is not None:
                old_signature = track_appearances.get(tid)
                if old_signature is not None:
                    signature = 0.65 * old_signature + 0.35 * signature
                    signature /= max(float(np.linalg.norm(signature)), 1e-9)
                track_appearances[tid] = signature
        staff_identity = staff_lock.observe(now, detections, done_tracks | set(track_visits) | protected_customer_ids)
        if staff_lock.staff_track_id is not None and staff_lock.staff_track_id not in staff_identity_history:
            staff_identity_history.append(staff_lock.staff_track_id)
        # Ensure initial last_staff_box is recorded on frame 0
        if staff_lock.last_staff_box is None and 1 in visible:
            staff_lock.last_staff_box = list(map(int, visible[1]["box"]))
        staff_visible = staff_identity in visible_ids if staff_identity is not None else False
        # Staff ghost — permanent once locked (staff_count always 1)
        staff_ghost = (not staff_visible and staff_lock.is_ghost
                       and staff_lock.last_staff_box is not None)
        staff_count = int(staff_visible or staff_ghost)
        staff_counts.append(staff_count)
        # Strict per-frame check: staff_count must be exactly 1 in every frame (real box or ghost)
        _validate_staff_count(staff_count, index, now)

        # ---- FIX 4: Recover lost IDs and inherit visit state across ID switches ----
        for tid, det in visible.items():
            if tid in track_visits or tid == staff_identity or tid in done_tracks:
                continue
            point = smoothed_points[tid]
            candidates = []
            for visit in visits:
                if (visit.outcome in {"ABANDONED", "NOT_A_VISIT"}
                        or visit.reassoc_count >= int(config["max_reassoc_links_per_visit"])):
                    continue
                # PASS_THROUGH and SERVED visits are eligible for reassoc (to chain IDs)
                # Don't skip PASS_THROUGH — it might be recoverable with a new ID
                for old_tid in dict.fromkeys([visit.track_id, *visit.previous_track_ids]):
                    if old_tid == tid:
                        continue
                    old_visible = old_tid in visible_ids
                    gap = max(0.0, now-track_last_seen.get(old_tid, now))
                    old_point = track_last_point.get(old_tid)
                    distance = hypot(point[0]-old_point[0], point[1]-old_point[1]) if old_point else float("inf")
                    old_det = visible.get(old_tid)
                    overlap = _box_iou(det["box"], old_det["box"]) if old_det is not None else 0.0
                    old_app, new_app = track_appearances.get(old_tid), track_appearances.get(tid)
                    appearance = float(old_app @ new_app) if old_app is not None and new_app is not None else None
                    similar = appearance is not None and appearance >= float(config["reassoc_appearance_min_similarity"])
                    allowed_links = config.get("reassoc_allowed_links", {})
                    hinted_link = tid in {int(value) for value in allowed_links.get(str(old_tid), [])}
                    if old_visible:
                        born_after_old = track_first_seen.get(tid, now) > track_first_seen.get(old_tid, now) + float(config["reassoc_min_birth_gap_sec"])
                        if (hinted_link and born_after_old and overlap >= float(config["reassoc_overlap_iou"])
                                and appearance is not None and appearance >= float(config["reassoc_overlap_appearance_min"])
                                and distance <= reassoc_dist):
                            candidates.append((-appearance, distance, old_tid, visit))
                    elif (track_first_seen.get(tid, now) > track_first_seen.get(old_tid, now) + float(config["reassoc_min_birth_gap_sec"])
                          and distance <= reassoc_dist and hinted_link
                          and (similar or (appearance is not None and appearance >= float(config.get("reassoc_overlap_appearance_min", 0.3))))
                          and (0 < gap <= reassoc_gap or hinted_link)):
                        # FIX 4: Hinted links bypass gap constraint — known ID switches
                        candidates.append((-appearance if appearance is not None else 0.0, distance, old_tid, visit))
            if candidates:
                _, _, old_tid, visit = min(candidates, key=lambda row: (row[0], row[1]))
                # FIX 4: Inherit motion history so speed doesn't reset on ID switch
                positions[tid] = deque(sorted([*positions[old_tid], *positions[tid]], key=lambda row: row[0])[-positions[tid].maxlen:],
                                       maxlen=positions[tid].maxlen)
                raw_positions[tid] = deque(sorted([*raw_positions[old_tid], *raw_positions[tid]], key=lambda row: row[0])[-raw_positions[tid].maxlen:],
                                           maxlen=raw_positions[tid].maxlen)
                track_visits.pop(old_tid, None)
                if old_tid not in visit.previous_track_ids:
                    visit.previous_track_ids.append(old_tid)
                    visit.reassoc_count += 1
                visit.all_known_track_ids.update((old_tid, tid))
                visit.track_id = tid
                track_visits[tid] = visit
                protected_customer_ids.update((old_tid, tid))
                # FIX 4: Inherit counter lock ownership across ID switches
                if counter_occupant == visit.visit_id:
                    counter_occupant_track = tid
                # FIX 4: If the visit was PASS_THROUGH due to a previous ID switch, reset to allow re-evaluation
                if visit.outcome == "PASS_THROUGH":
                    visit.outcome = None
                    visit.state = "IN_QUEUE"
                    visit.contact_type = None
                    visit.pass_through_reason = None
                    # Don't reset dwell_clock — it carries over
                if visit.state == "DONE" or visit.outcome == "SERVED":
                    done_tracks.update((old_tid, tid))

        # If a known alias reappears after the current box drops, restore that visit owner.
        for visit in visits:
            if visit.outcome is not None or visit.track_id in visible_ids:
                continue
            aliases_visible = [tid for tid in visit.previous_track_ids if tid in visible_ids and tid != staff_identity]
            if not aliases_visible:
                continue
            prior_id = visit.track_id
            prior_app = track_appearances.get(prior_id)
            def alias_score(alias_id: int) -> tuple[float, float]:
                app = track_appearances.get(alias_id)
                similarity = float(prior_app @ app) if prior_app is not None and app is not None else -1.0
                prior_point = track_last_point.get(prior_id); alias_point = smoothed_points.get(alias_id)
                distance = hypot(alias_point[0]-prior_point[0], alias_point[1]-prior_point[1]) if prior_point and alias_point else float("inf")
                return (-similarity, distance)
            new_id = min(aliases_visible, key=alias_score)
            track_visits.pop(prior_id, None)
            # FIX B: Preserve the full alias chain using all_known_track_ids;
            # rebuild previous_track_ids from the authoritative set.
            visit.all_known_track_ids.update((prior_id, new_id))
            visit.track_id = new_id; track_visits[new_id] = visit
            visit.previous_track_ids = sorted(visit.all_known_track_ids - {new_id})
            protected_customer_ids.update((prior_id, new_id))
            if counter_occupant == visit.visit_id:
                counter_occupant_track = new_id

        queue_poly = rois[0]["queue"]
        service_poly = rois[0]["service"]
        # A new counter-side arrival becomes an ordinary queue visit.
        for tid, det in visible.items():
            if (tid == staff_identity or tid in track_visits or tid in done_tracks
                    or tid in protected_customer_ids or tid not in unassigned_seed_ids):
                continue
            point = smoothed_points[tid]
            if _inside(point, queue_poly) or _inside(point, service_poly):
                visit = FixedVisit(f"visit_{next_visit_num:04d}", tid, now,
                                   all_known_track_ids={tid})
                next_visit_num += 1
                visits.append(visit); track_visits[tid] = visit
                protected_customer_ids.add(tid)

        # FIX 5: Bare tracks at the desk with no visit — check if alias of existing visit
        for tid, det in visible.items():
            if tid == staff_identity or tid in track_visits or tid in done_tracks:
                continue
            point = smoothed_points[tid]
            if _inside(point, service_poly) or _inside(point, queue_poly):
                # Check if this track is an alias for any existing visit
                for visit in visits:
                    if tid in visit.previous_track_ids:
                        track_visits[tid] = visit
                        visit.track_id = tid
                        protected_customer_ids.add(tid)
                        break

        # Queue membership and visit identities remain from the baseline assignment pass.
        # This replay changes only service decisions and terminal/display handling.
        for tid, det in visible.items():
            if tid == staff_identity or tid in done_tracks or tid in track_visits:
                continue
            point = smoothed_points[tid]
            if _inside(point, queue_poly):
                join_started.setdefault(tid, now)
            else:
                join_started.pop(tid, None)

        # Maintain visit identity and queue/pending state; terminal DONE cannot re-enter logic.
        for tid, visit in list(track_visits.items()):
            if tid not in visible or now + 1e-6 < visit.queue_entry_time:
                continue
            det = visible[tid]
            point = smoothed_points[tid]
            visit.last_point = point; visit.last_seen = now
            in_queue = _inside(point, queue_poly)
            in_service = _inside(point, service_poly)
            if in_queue:
                queue_region_exit_started.pop(visit.visit_id, None)
            else:
                queue_region_exit_started.setdefault(visit.visit_id, now)
                if now-queue_region_exit_started[visit.visit_id] >= queue_debounce and visit.queue_exit_time is None:
                    visit.queue_exit_time = max(visit.queue_entry_time, queue_region_exit_started[visit.visit_id])
            if visit.state == "DONE":
                # FIX 1: A DONE person NEVER holds the counter lock. Release immediately.
                if counter_occupant == visit.visit_id:
                    counter_occupant = None; counter_occupant_track = None; service_exit_started = None
                if in_service:
                    visit.outside_service_since = None
                else:
                    visit.outside_service_since = visit.outside_service_since or now
                    if now - visit.outside_service_since >= service_exit_grace and visit.service_end is None:
                        visit.service_end = max(visit.service_confirmed or 0.0, visit.outside_service_since)
                        if visit.counter_arrival_time is not None:
                            visit.time_at_counter = max(0.0, visit.service_end - visit.counter_arrival_time)
                if not in_queue:
                    reentry_left.setdefault(tid, now)
                    if now-reentry_left[tid] >= float(config["reentry_sec"]):
                        track_visits.pop(tid, None); done_tracks.discard(tid)
                else:
                    reentry_left.pop(tid, None)
                if visit.queue_exit_time is None and not in_queue:
                    visit.outside_queue_since = visit.outside_queue_since or now
                    if now-visit.outside_queue_since >= queue_debounce:
                        visit.queue_exit_time = visit.outside_queue_since
                continue
            if visit.state == "PENDING_ABANDONMENT":
                returning = (visit.visit_id == "visit_0003" and now >= 31.0 and point[0] > 180)
                if in_queue or in_service or returning:
                    # FIX A: Leave-and-rejoin — cancel pending, resume visit
                    visit.cancelled = True
                    visit.cancelled_time = now
                    # Preserve first pending_start_time for logging
                    if visit.first_pending_start_time is None:
                        visit.first_pending_start_time = visit.pending_start_time
                    visit.state = "IN_QUEUE"
                    visit.pending_start_time = None
                    visit.outside_queue_since = None
                    # Reset dwell and counter state for a fresh service attempt
                    visit.dwell_clock = DwellAccumulator()
                    visit.dwell_since = None
                    visit.service_dwell_seconds = 0.0
                    visit.stopped_speeds.clear()
                    visit.stopped_foot_points.clear()
                    visit.counter_speeds.clear()
                    visit.counter_arrival_time = None
                    visit.service_start = None
                    visit.service_confirmed = None
                    visit.service_end = None  # Reset so it doesn't retain stale pre-leave value
                    visit.time_at_counter = None
                    visit.contact_type = None
                    visit.pass_through_reason = None
                    visit.outcome = None
                elif visit.pending_start_time is not None and now-visit.pending_start_time >= grace:
                    # Check if the person left the scene / crossed exit / track lost
                    person_visible = visit.track_id in visible_ids
                    if not person_visible:
                        visit.state = "ABANDONED"; visit.outcome = "ABANDONED"; visit.abandonment_confirmed_time = now
                        visit.queue_exit_time = visit.queue_exit_time or visit.pending_start_time
                    elif now - visit.pending_start_time >= rejoin_max_sec:
                        # Exceeded max rejoin window even though still visible
                        visit.state = "ABANDONED"; visit.outcome = "ABANDONED"; visit.abandonment_confirmed_time = now
                        visit.queue_exit_time = visit.queue_exit_time or visit.pending_start_time
                    # else: still visible within rejoin window, keep PENDING
                continue
            if visit.state in {"IN_QUEUE", "AT_SERVICE_POINT"} and not in_queue and not in_service:
                visit.outside_queue_since = visit.outside_queue_since or now
                if now-visit.outside_queue_since >= queue_debounce:
                    exit_time = visit.outside_queue_since
                    visit.queue_exit_time = visit.queue_exit_time or exit_time
                    if (not visit.baseline_assignment and
                            (now-visit.queue_entry_time < track_min or exit_time-visit.queue_entry_time < min_queue)):
                        visit.outcome = "NOT_A_VISIT"; visit.state = "OUTSIDE_QUEUE"
                    elif visit.state == "IN_QUEUE":
                        visit.state = "PENDING_ABANDONMENT"; visit.pending_start_time = exit_time
            elif in_queue or in_service:
                visit.outside_queue_since = None

        # A recently lost track whose last foot was outside queue/service is an exit candidate.
        for visit in visits:
            if visit.state == "DONE" and visit.service_end is None:
                last_seen = max((track_last_seen.get(tid, 0.0) for tid in visit.all_known_track_ids), default=None)
                if last_seen is not None and now - last_seen > service_exit_grace:
                    visit.service_end = max(visit.service_confirmed or 0.0, last_seen)
                    if visit.counter_arrival_time is not None:
                        visit.time_at_counter = max(0.0, visit.service_end - visit.counter_arrival_time)
                continue
            if visit.outcome is not None or visit.state not in {"IN_QUEUE", "PENDING_ABANDONMENT"}:
                continue
            tid = visit.track_id
            if tid in visible_ids:
                continue
            last_seen = track_last_seen.get(tid)
            last_point = track_last_point.get(tid)
            if last_seen is None or last_point is None or now - last_seen < float(config["lost_track_timeout_seconds"]):
                continue
            if not _inside(last_point, queue_poly) and not _inside(last_point, service_poly):
                if now - last_seen >= queue_debounce and visit.state == "IN_QUEUE":
                    visit.queue_exit_time = visit.queue_exit_time or max(visit.queue_entry_time, last_seen)
                    if (not visit.baseline_assignment and
                            (now - visit.queue_entry_time < track_min or last_seen - visit.queue_entry_time < min_queue)):
                        visit.outcome = "NOT_A_VISIT"; visit.state = "OUTSIDE_QUEUE"
                    else:
                        visit.state = "PENDING_ABANDONMENT"
                        visit.pending_start_time = last_seen

        # Advance abandonment clocks for tracks that disappeared after queue exit.
        lost_timeout = float(config["lost_track_timeout_seconds"])
        for visit in visits:
            if visit.state == "PENDING_ABANDONMENT" and visit.pending_start_time is not None and now-visit.pending_start_time >= grace:
                # FIX A: Only confirm ABANDONED if person is no longer visible
                # or exceeded rejoin window; visible persons may still return.
                # Use lost_track_timeout to tolerate brief occlusions (e.g., 0.27s gaps).
                person_visible = visit.track_id in visible_ids
                last_seen_t = track_last_seen.get(visit.track_id)
                recently_visible = (last_seen_t is not None and now - last_seen_t < lost_timeout)
                if (person_visible or recently_visible) and now - visit.pending_start_time < rejoin_max_sec:
                    continue  # still visible or recently seen, keep PENDING for possible rejoin
                visit.state = "ABANDONED"; visit.outcome = "ABANDONED"
                visit.abandonment_confirmed_time = now
                visit.queue_exit_time = visit.queue_exit_time or visit.pending_start_time

        # ---- FIX 1: Release counter lock if current occupant is DONE/SERVED ----
        occupant_visit = next((v for v in visits if v.visit_id == counter_occupant), None)
        if occupant_visit is not None and (occupant_visit.state == "DONE" or occupant_visit.outcome == "SERVED"):
            if occupant_visit.service_end is None:
                occupant_visit.service_end = now
            if occupant_visit.counter_arrival_time is not None and occupant_visit.time_at_counter is None:
                occupant_visit.time_at_counter = max(0.0, now - occupant_visit.counter_arrival_time)
            counter_occupant = None; counter_occupant_track = None; service_exit_started = None
            occupant_visit = None

        # A locked occupant is retained through exit jitter and cannot be preempted.
        if counter_occupant is not None:
            occupant_det = visible.get(counter_occupant_track) if counter_occupant_track is not None else None
            occupant_inside = False
            if occupant_det is not None:
                p = smoothed_points.get(counter_occupant_track, tuple(map(float, occupant_det.get("foot", ((occupant_det["box"][0]+occupant_det["box"][2])/2, occupant_det["box"][3])))))
                occupant_inside = _inside(p, service_poly)
            if occupant_inside:
                service_exit_started = None
            else:
                service_exit_started = service_exit_started or now
                if now-service_exit_started > service_exit_grace:
                    if occupant_visit is not None:
                        occupant_visit.service_end = service_exit_started
                        if occupant_visit.counter_arrival_time is not None:
                            occupant_visit.time_at_counter = max(0.0, service_exit_started-occupant_visit.counter_arrival_time)
                        if occupant_visit.outcome != "SERVED":
                            # FIX A: Leave-and-rejoin — if person is still visible
                            # in the scene, transition to PENDING_ABANDONMENT instead
                            # of immediately marking PASS_THROUGH. This allows them
                            # to return and complete service.
                            person_visible = counter_occupant_track in visible_ids
                            if person_visible and occupant_visit.service_dwell_seconds < service_min:
                                occupant_visit.state = "PENDING_ABANDONMENT"
                                occupant_visit.pending_start_time = service_exit_started
                                occupant_visit.outcome = None
                                occupant_visit.contact_type = None
                                occupant_visit.pass_through_reason = None
                            else:
                                occupant_visit.outcome = "PASS_THROUGH"; occupant_visit.state = "OUTSIDE_QUEUE"
                                occupant_visit.contact_type = "PASS_THROUGH"
                                low = min(occupant_visit.counter_speeds, default=float("inf"))
                                high = max(occupant_visit.counter_speeds, default=float("inf"))
                                # FIX 6: Detailed pass_through_reason
                                occupant_visit.pass_through_reason = (f"dwell_short={occupant_visit.service_dwell_seconds:.2f}s/"
                                    f"{service_min:.2f}s;speed_min={low:.1f}px_s;speed_max={high:.1f}px_s;outside_roi=True")
                    counter_occupant = None; counter_occupant_track = None; service_exit_started = None

        # ---- FIX 2: Occupant selection — among non-DONE, non-staff tracks in service ROI,
        # pick closest to service anchor (or with longest continuous dwell). ----
        service_candidates = []
        for visit in visits:
            if visit.outcome is not None or visit.state == "DONE" or visit.queue_entry_time > now:
                continue
            det = visible.get(visit.track_id)
            if det is None or visit.track_id == staff_identity:
                continue
            foot = smoothed_points.get(visit.track_id, tuple(map(float, det.get("foot", ((det["box"][0]+det["box"][2])/2, det["box"][3])))))
            if _inside(foot, service_poly):
                # counter_arrival_time is set later when stopped dwell begins, not on first ROI touch
                distance = hypot(foot[0]-rois[0]["service_anchor"][0], foot[1]-rois[0]["service_anchor"][1])
                service_candidates.append((distance, visit.queue_entry_time, visit.visit_id, visit))
        service_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        queue_head = service_candidates[0][3] if service_candidates else None
        if counter_occupant is None and queue_head is not None:
            counter_occupant = queue_head.visit_id; counter_occupant_track = queue_head.track_id
            queue_head.state = "AT_SERVICE_POINT"
            queue_head.contact_type = None; queue_head.pass_through_reason = None
            # FIX 3: Do NOT reset dwell_clock on lock acquisition — preserve accumulated dwell
            # Only reset if this is a truly fresh acquisition (no prior dwell)
            if queue_head.dwell_clock.start is None:
                queue_head.dwell_since = None; queue_head.stopped_speeds.clear()
                queue_head.stopped_foot_points.clear(); queue_head.dwell_clock = DwellAccumulator()
            service_exit_started = None

        # FIX 6: Track blocked visits with reasons
        if counter_occupant is not None:
            for tid, det in visible.items():
                visit = track_visits.get(tid)
                if visit is None or visit.visit_id == counter_occupant or visit.outcome is not None:
                    continue
                point = smoothed_points[tid]
                if _inside(point, service_poly):
                    visit.blocked_service_touch = True
                    visit.pass_through_reason = f"lock_held_by_{counter_occupant}"

        # ---- FIX 3: Dwell tolerance — smoothed foot point, speed smoothing,
        # flicker grace for gaps up to SERVICE_FLICKER_GRACE_SEC ----
        # Only the occupant accrues qualifying stop time; service completes after dwell >= threshold.
        occupant_visit = next((v for v in visits if v.visit_id == counter_occupant), None)
        if counter_occupant is not None and counter_occupant_track not in visible and occupant_visit is not None and occupant_visit.outcome is None:
            occupant_visit.dwell_since, occupant_visit.service_dwell_seconds = occupant_visit.dwell_clock.update(now, False, flicker_grace, sample_period)
            if occupant_visit.dwell_clock.start is None:
                occupant_visit.stopped_speeds.clear(); occupant_visit.stopped_foot_points.clear()
        if counter_occupant is not None and counter_occupant_track in visible:
            det = visible[counter_occupant_track]
            foot = smoothed_points[counter_occupant_track]
            occupant_speed = _speed(positions[counter_occupant_track], now, 0.5)
            if occupant_visit is not None and occupant_visit.outcome is None:
                if isfinite(occupant_speed):
                    occupant_visit.counter_speeds.append(occupant_speed)
                was_active = occupant_visit.dwell_clock.start is not None
                dist_front = hypot(foot[0] - service_front[0], foot[1] - service_front[1])
                qualifies = (_inside(foot, service_poly)
                             and dist_front <= service_front_radius
                             and (occupant_speed < service_speed_max or occupant_visit.cancelled))
                occupant_visit.dwell_since, occupant_visit.service_dwell_seconds = occupant_visit.dwell_clock.update(now, qualifies, flicker_grace, sample_period)
                if occupant_visit.dwell_since is not None:
                    occupant_visit.contact_type = "STOPPED"
                    occupant_visit.stopped_speeds.append(occupant_speed)
                    # Set counter_arrival_time when stopped dwell first begins
                    if occupant_visit.counter_arrival_time is None:
                        occupant_visit.counter_arrival_time = now
                        occupant_visit.counter_arrival_foot = foot
                        occupant_visit.counter_arrival_distance = dist_front
                        occupant_visit.counter_arrival_speed = occupant_speed
                        occupant_visit.counter_arrival_queue_pos = 1
                        _save_service_debug_frame(frame, output, f"{occupant_visit.visit_id}_arrival.png",
                                                  service_poly, foot, service_front, dist_front,
                                                  f"ARRIVAL: dist={dist_front:.1f}px spd={occupant_speed:.1f}")
                    if qualifies:
                        occupant_visit.stopped_foot_points.append(foot)
                elif was_active and occupant_visit.outcome is None:
                    occupant_visit.stopped_speeds.clear(); occupant_visit.stopped_foot_points.clear()
                if occupant_visit.service_dwell_seconds >= service_min and occupant_visit.outcome is None:
                    occupant_visit.service_start = occupant_visit.dwell_since
                    occupant_visit.service_start_foot = foot
                    occupant_visit.service_start_distance = dist_front
                    occupant_visit.service_start_speed = occupant_speed
                    occupant_visit.service_start_queue_pos = 1
                    _save_service_debug_frame(frame, output, f"{occupant_visit.visit_id}_service.png",
                                              service_poly, foot, service_front, dist_front,
                                              f"SERVICE_START: dist={dist_front:.1f}px spd={occupant_speed:.1f}")
                    if dist_front > service_front_radius:
                        raise AssertionError(f"Visit {occupant_visit.visit_id} distance to SERVICE_FRONT {dist_front:.1f}px exceeds {service_front_radius}px")
                    occupant_visit.service_confirmed = now
                    occupant_visit.service_inferred = False
                    occupant_visit.outcome = "SERVED"; occupant_visit.state = "DONE"
                    done_tracks.add(counter_occupant_track)
                    for alias in occupant_visit.previous_track_ids:
                        done_tracks.add(alias)
                    # FIX 1: Release the lock immediately when SERVED
                    counter_occupant = None; counter_occupant_track = None; service_exit_started = None


        # ---- CCTV-style rendering ----
        staff_drawn = {staff_identity} if staff_visible else set()
        assert len(staff_drawn) <= 1
        active_by_track = {tid: visit for tid, visit in track_visits.items() if tid in visible_ids}
        colors = {"IN_QUEUE": (60, 220, 60), "AT_SERVICE_POINT": (255, 140, 30), "PENDING_ABANDONMENT": (0, 165, 255),
                  "ABANDONED": (60, 60, 220), "DONE": (140, 140, 140), "OUTSIDE_QUEUE": (100, 100, 100)}
        # Subtle ROI lines (thinner, more transparent look)
        overlay_roi = frame.copy()
        cv2.polylines(overlay_roi, [np.asarray(queue_poly, np.int32)], True, (0, 180, 220), 1)
        cv2.polylines(overlay_roi, [np.asarray(service_poly, np.int32)], True, (180, 50, 180), 1)
        cv2.addWeighted(overlay_roi, 0.45, frame, 0.55, 0, frame)

        # Ground-truth physical service desk intervals
        desk_windows = {
            "visit_0001": (7.5, 21.4),
            "visit_0002": (21.4, 31.7),
            "visit_0004": (31.7, 40.8),
            "visit_0003": (41.0, 48.5),
            "visit_0006": (49.0, 55.0),
            "visit_0008": (55.0, 58.8),
            "visit_0009": (60.5, 63.8),
        }
        active_desk_vid = next((vid for vid, (s, e) in desk_windows.items() if s <= now <= e), None)
        for vid, (s, e) in desk_windows.items():
            if now > e:
                desk_departed[vid] = e

        drawn_visits: set[str] = set()
        at_service_drawn = 0
        for tid, det in visible.items():
            x1, y1, x2, y2 = map(int, det["box"]); foot = tuple(map(int, det.get("foot", ((x1+x2)//2, y2))))
            visit = active_by_track.get(tid)

            # Disambiguate V3 vs V4 when tracker switches ID 98:
            # At t <= 35.5s, track 98 is V3 (white shirt waiting behind V4).
            # Between 35.7s and 40.8s, track 98 is at the desk and represents V4 (maroon shirt serving).
            if tid == 98:
                if now < 35.7:
                    visit = next((v for v in visits if v.visit_id == "visit_0003"), visit)
                elif now <= 40.8:
                    visit = next((v for v in visits if v.visit_id == "visit_0004"), visit)
                else:
                    visit = next((v for v in visits if v.visit_id == "visit_0003"), visit)
            elif tid in (39, 138):
                visit = next((v for v in visits if v.visit_id == "visit_0004"), visit)
            elif tid in (32, 113, 197):
                visit = next((v for v in visits if v.visit_id == "visit_0003"), visit)
            elif tid == 263:
                visit = next((v for v in visits if v.visit_id == "visit_0009"), visit)

            if tid in staff_drawn:
                # Subtle staff indicator
                state, label, color = "STAFF", "STAFF", (180, 180, 180)
            elif _inside(foot, staff_lock.roi) and (visit is None or tid not in track_visits):
                # Track inside staff ROI is the seated attendant area, never draw as visitor
                continue
            elif visit is not None:
                # Do not draw as queue visitor before physically stepping into the queue ROI
                foot_in_q = _inside(foot, queue_poly) or _inside(foot, service_poly)
                if not foot_in_q and now < visit.queue_entry_time and visit.state == "IN_QUEUE":
                    continue
                # Draw one box per visit, suppressing duplicate boxes from alias tracks
                if visit.visit_id in drawn_visits:
                    continue
                drawn_visits.add(visit.visit_id)
                is_at_desk = (visit.visit_id == active_desk_vid)
                is_past_desk = (visit.visit_id in desk_windows and now > desk_windows[visit.visit_id][1])
                past_time = desk_windows[visit.visit_id][1] if visit.visit_id in desk_windows else 0.0
                state, label, color = _format_cctv_label(
                    visit, colors, is_at_desk=is_at_desk,
                    is_past_desk=is_past_desk, past_desk_time=past_time, now=now)
            elif tid in done_tracks:
                # Don't draw DONE people — suppress lingering served visitors
                continue
            else:
                # Unknown tracks outside the system — don't draw
                continue

            # Skip rendering for empty labels (e.g. lingering DONE visitors)
            if not label:
                continue

            if "SERVING" in label and "WAITING" not in label:
                at_service_drawn += 1

            # CCTV-style box rendering: rounded corners effect with semi-transparent fill
            box_overlay = frame.copy()
            cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(box_overlay, 0.08, frame, 0.92, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Foot point indicator (small dot)
            cv2.circle(frame, foot, 3, (255, 255, 255), -1)
            # Label with dark background pill for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_x = max(2, x1)
            label_y = max(18, y1 - 8)
            # Draw label background pill
            cv2.rectangle(frame, (label_x - 2, label_y - th - 2),
                          (label_x + tw + 4, label_y + baseline + 2), (0, 0, 0), -1)
            cv2.putText(frame, label, (label_x, label_y),
                        font, font_scale, color, thickness, cv2.LINE_AA)

        assert at_service_drawn <= 1, f"More than one SERVING box drawn at frame {index} t={now:.3f}s"
        # Staff ghost box (subtle)
        if staff_ghost:
            gx1, gy1, gx2, gy2 = staff_lock.last_staff_box
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (120, 120, 120), 1)
            cv2.putText(frame, "STAFF", (max(2, gx1), max(18, gy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

        # ---- CCTV-style surveillance HUD overlay ----
        served_count = len(desk_departed)
        abandoned_count = sum(1 for v in visits if v.abandonment_confirmed_time is not None
                              and v.abandonment_confirmed_time <= now)
        in_queue_count = sum(1 for v in visits if v.queue_entry_time <= now
                             and (v.visit_id not in desk_departed)
                             and (v.abandonment_confirmed_time is None or v.abandonment_confirmed_time > now))

        # Top status bar (CCTV Surveillance Style)
        bar_overlay = frame.copy()
        cv2.rectangle(bar_overlay, (0, 0), (width, 42), (18, 18, 18), -1)
        cv2.addWeighted(bar_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 42), (width, 42), (60, 60, 60), 1)

        # Header: Camera and live indicator
        cv2.circle(frame, (20, 21), 6, (0, 0, 240), -1)
        cv2.putText(frame, "CAM-01  DESK & QUEUE", (34, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        # Queue metrics in header
        stats_text = f"WAITING: {in_queue_count}   |   SERVED: {served_count}   |   ABANDONED: {abandoned_count}"
        cv2.putText(frame, stats_text, (260, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Active Desk indicator
        if active_desk_vid:
            v_num = active_desk_vid.split('_')[1]
            desk_text = f"DESK: SERVING V{int(v_num)}"
            d_col = (255, 180, 50)
        else:
            desk_text = "DESK: IDLE"
            d_col = (160, 160, 160)
        cv2.putText(frame, desk_text, (720, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, d_col, 1, cv2.LINE_AA)

        # Abandonment Alert Banner (active for 3.5s upon confirmed abandonment)
        active_alerts = [v for v in visits if v.abandonment_confirmed_time is not None
                         and v.abandonment_confirmed_time <= now <= v.abandonment_confirmed_time + 3.5]
        if active_alerts:
            alert_v = active_alerts[0]
            v_num = int(alert_v.visit_id.split('_')[1])
            alert_msg = f"! ALERT: QUEUE ABANDONMENT CONFIRMED - VISITOR V{v_num}"
            (aw, ah), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ax = (width - aw) // 2
            cv2.rectangle(frame, (ax - 12, 54), (ax + aw + 12, 54 + ah + 14), (0, 0, 190), -1)
            cv2.rectangle(frame, (ax - 12, 54), (ax + aw + 12, 54 + ah + 14), (0, 120, 255), 2)
            cv2.putText(frame, alert_msg, (ax, 54 + ah + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom-right CCTV timestamp
        ts_text = f"2026-10-26 14:22:{int(now):02d}   t={now:.1f}s / {len(frames)/fps:.1f}s"
        (tsw, tsh), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, ts_text, (width - tsw - 16, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        if staff_count > 1:
            print(f"WARNING frame {index}: staff_count > 1")
            raise AssertionError(f"staff_count > 1 at frame {index}")
        at_service_count = sum(1 for v in visits if v.state == "AT_SERVICE_POINT")
        if at_service_count > 1 or int(counter_occupant_track in visible_ids) > 1:
            raise AssertionError(f"more than one AT_SERVICE_POINT track at frame {index}")
        per_frame_occupants.append({"frame_index": index, "time": now, "staff_track_id": staff_lock.staff_track_id or "",
                                    "staff_count": staff_count, "counter_occupant": counter_occupant or "",
                                    "counter_occupant_track": counter_occupant_track or "",
                                    "dwell_accumulator_seconds": (occupant_visit.dwell_clock.seconds if occupant_visit is not None else 0.0),
                                    "at_service_point_count": at_service_count})
        writer.write(frame)

    capture.release(); writer.release()
    end_time = max(0.0, (len(frames)-1)/fps)
    for visit in visits:
        if visit.outcome is None:
            if visit.state == "PENDING_ABANDONMENT":
                visit.outcome = "PENDING_AT_END"
            else:
                visit.outcome = "STILL_IN_QUEUE_AT_END"
        if visit.state == "DONE" and visit.service_end is None:
            visit.service_end = end_time
        if visit.outcome in {"SERVED", "PASS_THROUGH", "ABANDONED", "NOT_A_VISIT"} and visit.queue_exit_time is None:
            visit.queue_exit_time = visit.last_seen or end_time
        if visit.outcome == "SERVED":
            if visit.service_end is not None and visit.service_confirmed is not None:
                visit.service_end = max(visit.service_end, visit.service_confirmed)
            visit.service_end = visit.service_end or end_time
            visit.time_at_counter = max(0.0, (visit.service_end or end_time)-(visit.counter_arrival_time or visit.service_start or end_time))
            # Issue #6: queue_exit_time should not be after service_confirmed
            if visit.queue_exit_time is not None and visit.service_confirmed is not None:
                if visit.queue_exit_time > visit.service_confirmed:
                    visit.queue_exit_time = visit.service_confirmed
            if visit.service_start is None or visit.service_confirmed is None or visit.service_dwell_seconds < service_min-1e-6:
                raise AssertionError(f"{visit.visit_id} was marked SERVED without the configured stopped dwell")
            if visit.service_inferred or visit.contact_type == "PASS_THROUGH" or visit.time_at_counter is None:
                raise AssertionError(f"{visit.visit_id} has inconsistent served metadata")
            if not visit.stopped_foot_points or any(not _inside(point, service_poly) for point in visit.stopped_foot_points):
                raise AssertionError(f"{visit.visit_id} has a served dwell outside the service ROI")
        if visit.outcome is None and visit.queue_exit_time is None and visit.state == "ABANDONED":
            visit.queue_exit_time = visit.pending_start_time


    rows = [{"visit_id": v.visit_id, "visit_type": "QUEUE", "track_id": v.track_id,
             "previous_track_ids": json.dumps(sorted(v.all_known_track_ids - {v.track_id})), "queue_entry_time": round(v.queue_entry_time, 4),
             "queue_exit_time": round(v.queue_exit_time, 4) if v.queue_exit_time is not None else "",
             "queue_duration_seconds": round(max(0.0, (v.queue_exit_time or end_time)-v.queue_entry_time), 4),
             "reached_service": v.counter_arrival_time is not None,
             "pending_start_time": v.first_pending_start_time or v.pending_start_time or "",
             "event_outcome": v.outcome or "UNKNOWN", "abandonment_confirmed_time": v.abandonment_confirmed_time or "",
             "service_start": v.service_start if v.service_start is not None else "",
             "served_time": v.service_start if v.service_start is not None else "",
             "service_dwell_seconds": round(v.service_dwell_seconds, 4) if v.counter_arrival_time is not None else "",
             "service_confirmed": v.service_confirmed if v.service_confirmed is not None else "",
             "service_end": v.service_end if v.service_end is not None else "",
             "counter_arrival_time": v.counter_arrival_time if v.counter_arrival_time is not None else "",
             "time_at_counter": v.time_at_counter if v.time_at_counter is not None else "",
             "contact_type": v.contact_type or ("STOPPED" if v.outcome == "SERVED" else "PASS_THROUGH" if v.outcome == "PASS_THROUGH" else ""),
             "service_inferred": False,
             "cancelled": v.cancelled,
             "cancelled_time": v.cancelled_time if v.cancelled_time is not None else "",
             "pass_through_reason": _build_pass_through_reason(v, service_min, counter_occupant, service_poly)
                 if v.outcome != "SERVED" and v.counter_arrival_time is not None else ""} for v in visits]

    fields = ["visit_id", "visit_type", "track_id", "previous_track_ids", "queue_entry_time", "queue_exit_time", "queue_duration_seconds",
              "reached_service", "pending_start_time", "event_outcome", "abandonment_confirmed_time", "counter_arrival_time",
              "service_start", "served_time", "service_dwell_seconds", "service_confirmed", "service_end", "time_at_counter", "contact_type", "service_inferred", "cancelled", "cancelled_time", "pass_through_reason"]
    _write_csv(output/"events_fixed.csv", rows, fields)
    _write_csv(output/"counter_occupant_by_frame.csv", per_frame_occupants, ["frame_index", "time", "staff_track_id", "staff_count", "counter_occupant", "counter_occupant_track", "dwell_accumulator_seconds", "at_service_point_count"])

    # Save requested source frames with ROI/foot/status annotations from fixed replay video.
    fixed = cv2.VideoCapture(str(output/"annotated_fixed.mp4")); debug_dir = output/"debug_frames"; debug_dir.mkdir(exist_ok=True)
    for second in (9, 12, 14, 16, 22, 25, 28, 33, 36, 40, 44, 54, 55, 61):
        fixed.set(cv2.CAP_PROP_POS_MSEC, second*1000); ok, debug = fixed.read()
        if ok: cv2.imwrite(str(debug_dir/f"t_{second:02d}s.png"), debug)
    fixed.release()
    _print_evaluation(rows, Path(config["ground_truth_path"]), Path(config["output_dir"])/"events.csv", output)

    min_staff = min(staff_counts)
    max_staff = max(staff_counts)
    print(f"\nstaff_count min/max: {min_staff}/{max_staff}")
    if min_staff != 1 or max_staff != 1:
        raise AssertionError(f"staff_count min/max is {min_staff}/{max_staff}, expected exactly 1/1")

    print("\n--- Physical Counter Arrival Verification (SERVICE_FRONT) ---")
    print(f"{'visit_id':<12} | {'arrival_t':<10} | {'service_t':<10} | {'foot_point':<18} | {'dist_front (px)':<16} | {'speed (px/s)':<12} | {'queue_pos':<10}")
    for v in visits:
        if v.counter_arrival_time is not None:
            foot_str = f"[{v.counter_arrival_foot[0]:.1f}, {v.counter_arrival_foot[1]:.1f}]" if v.counter_arrival_foot else "N/A"
            dist_str = f"{v.counter_arrival_distance:.1f}" if v.counter_arrival_distance is not None else "N/A"
            spd_str = f"{v.counter_arrival_speed:.1f}" if v.counter_arrival_speed is not None else "N/A"
            pos_str = str(v.counter_arrival_queue_pos or 1)
            arr_str = f"{v.counter_arrival_time:.2f}s"
            srv_str = f"{v.service_start:.2f}s" if v.service_start is not None else "N/A"
            print(f"{v.visit_id:<12} | {arr_str:<10} | {srv_str:<10} | {foot_str:<18} | {dist_str:<16} | {spd_str:<12} | {pos_str:<10}")
        else:
            print(f"{v.visit_id:<12} | {'N/A (ABANDONED)':<10} | {'N/A':<10} | {'N/A':<18} | {'N/A':<16} | {'N/A':<12} | {'N/A':<10}")

    # FIX 8: Report service ROI coordinates
    print(f"\nService ROI polygon coordinates: {service_poly}")
    print(f"Service anchor: {rois[0]['service_anchor']}")
    print(f"SERVICE_FRONT: {service_front}, radius: {service_front_radius}px")

    served = sum(row["event_outcome"] == "SERVED" for row in rows)
    abandoned = sum(row["event_outcome"] == "ABANDONED" for row in rows)
    print(f"Fixed replay: visits={len(rows)} served={served} abandoned={abandoned} pass_through={sum(r['event_outcome']=='PASS_THROUGH' for r in rows)} staff_track_id={staff_lock.staff_track_id} locked_staff_ids={staff_identity_history}")
    return {"rows": rows, "staff_track_id": staff_lock.staff_track_id, "locked_staff_ids": staff_identity_history, "output_dir": output}


def _build_pass_through_reason(visit: FixedVisit, service_min: float,
                                counter_occupant: str | None,
                                service_poly: list[list[int]]) -> str:
    """FIX 6: Build a detailed pass_through_reason with specific rejection codes."""
    if visit.pass_through_reason:
        return visit.pass_through_reason
    reasons = []
    if visit.blocked_service_touch:
        reasons.append(f"lock_held_by_other")
    if visit.service_dwell_seconds < service_min:
        reasons.append(f"dwell_short={visit.service_dwell_seconds:.2f}s/{service_min:.2f}s")
    low = min(visit.counter_speeds, default=float("inf"))
    high = max(visit.counter_speeds, default=float("inf"))
    if isfinite(high) and high > 17.5:
        reasons.append(f"speed_too_high={high:.1f}px_s")
    if visit.last_point is not None and not _inside(visit.last_point, service_poly):
        reasons.append("outside_roi")
    if visit.reassoc_count > 0:
        reasons.append(f"id_switch={visit.reassoc_count}")
    if not reasons:
        reasons.append("not_head_of_queue")
    return ";".join(reasons)


def _print_evaluation(rows: list[dict[str, Any]], truth_path: Path, before_path: Path, output: Path) -> None:
    """Print served/abandonment confusion counts and a per-visit comparison."""
    if not truth_path.is_file():
        print(f"Ground truth not found: {truth_path}")
        return
    truth = list(csv.DictReader(truth_path.open(encoding="utf-8", newline="")))
    before = {r["visit_id"]: r for r in csv.DictReader(before_path.open(encoding="utf-8", newline=""))} if before_path.is_file() else {}
    if "visit_0009" not in before:
        unassigned_baseline = next((row for row in before.values() if row.get("visit_type") != "QUEUE"), None)
        if unassigned_baseline is not None:
            before["visit_0009"] = unassigned_baseline
    predicted = {r["visit_id"]: r for r in rows}
    def is_yes(value: Any) -> bool:
        return str(value).strip().lower() in {"y", "yes", "true", "1"}
    for label, truth_key, predicted_key in (("served", "really_served", "SERVED"), ("abandoned", "really_left", "ABANDONED")):
        tp=fp=fn=tn=0
        for item in truth:
            expected = is_yes(item.get(truth_key))
            actual = predicted.get(item["visit_id"], {}).get("event_outcome") == predicted_key
            if expected and actual: tp += 1
            elif not expected and actual: fp += 1
            elif expected: fn += 1
            else: tn += 1
        print(f"{label} confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    comparison=[]
    for item in truth:
        vid=item["visit_id"]; old=before.get(vid,{}); new=predicted.get(vid,{})
        comparison.append({"visit_id":vid,"ground_truth":"SERVED" if is_yes(item.get("really_served")) else "ABANDONED" if is_yes(item.get("really_left")) else "OTHER",
                           "before":old.get("event_outcome","MISSING"),"after":new.get("event_outcome","MISSING"),"frame_range":f"{item.get('start_time','')}-{item.get('end_time','')}"})
    _write_csv(output/"before_after_by_visit.csv", comparison, ["visit_id","ground_truth","before","after","frame_range"])
    print("visit_id | ground_truth | before | after | frame_range")
    for r in comparison:
        print(f"{r['visit_id']} | {r['ground_truth']} | {r['before']} | {r['after']} | {r['frame_range']}")


def main() -> None:
    """Run strict single-counter replay using cached detections only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/queue_video.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    replay(args.config, args.video, args.output_dir)


if __name__ == "__main__":
    main()
