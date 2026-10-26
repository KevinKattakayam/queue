"""Recover short detector ID gaps using position and video timestamps."""
from dataclasses import dataclass
from collections import defaultdict, deque
from math import acos, degrees, hypot
from typing import Any
import numpy as np

@dataclass
class LostTrack:
    """Saved state for a temporarily missing track."""
    track_id: int | str
    position: tuple[float, float]
    timestamp: float
    visit: Any
    inside_queue: bool
    timeout_seconds: float | None = None
    appearance: Any = None
    velocity: tuple[float, float] = (0.0, 0.0)
    moving_away: bool = False

class IdentityRecovery:
    """Hold eligible tracks briefly and link nearby returning detections."""
    def __init__(self, timeout_seconds: float, distance_px: float,
                 appearance_min_similarity: float = 0.6, max_link_direction_angle: float = 110.0):
        """Set maximum missing duration and reassociation radius."""
        self.timeout = timeout_seconds
        self.distance = distance_px
        self.lost: list[LostTrack] = []
        self.links: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.appearance_min_similarity = float(appearance_min_similarity)
        self.max_direction_angle = float(max_link_direction_angle)
        self.track_samples: dict[int | str, tuple[tuple[float, float], float, tuple[float, float], Any]] = {}
        self.track_appearance_history: dict[int | str, deque] = defaultdict(lambda: deque(maxlen=15))
        self.visit_appearance: dict[str, Any] = {}
        self.visit_samples: dict[str, tuple[tuple[float, float], float, tuple[float, float]]] = {}

    @staticmethod
    def appearance_similarity(first: Any, second: Any) -> float | None:
        """Compare normalized HSV histograms; return 0..1, or None if missing."""
        if first is None or second is None:
            return None
        a = first.astype("float32").ravel(); b = second.astype("float32").ravel()
        denom = float((a @ a) ** 0.5 * (b @ b) ** 0.5)
        if denom <= 1e-12:
            return None
        return max(0.0, min(1.0, float(a @ b) / denom))

    def observe(self, track_id: int | str, position: tuple[float, float], timestamp: float,
                appearance: Any = None, visit: Any = None, overlap: bool = False) -> None:
        """Track recent velocity and learn appearance only from an unambiguous box."""
        previous = self.track_samples.get(track_id)
        velocity = (0.0, 0.0)
        if previous is not None and timestamp > previous[1]:
            velocity = ((position[0] - previous[0][0]) / (timestamp - previous[1]),
                        (position[1] - previous[0][1]) / (timestamp - previous[1]))
        prior_app = previous[3] if previous else None
        if appearance is not None and not overlap:
            self.track_appearance_history[track_id].append(appearance.astype("float32"))
        history = self.track_appearance_history[track_id]
        clean_app = np.median(np.stack(history), axis=0) if history else prior_app
        self.track_samples[track_id] = (position, timestamp, velocity, clean_app)
        if visit is not None and clean_app is not None and not overlap:
            previous_app = self.visit_appearance.get(visit.visit_id)
            self.visit_appearance[visit.visit_id] = clean_app if previous_app is None else 0.9 * previous_app + 0.1 * clean_app
        if visit is not None and not overlap:
            prior_visit = self.visit_samples.get(visit.visit_id)
            visit_velocity = (0.0, 0.0)
            if prior_visit is not None and timestamp > prior_visit[1]:
                visit_velocity = ((position[0] - prior_visit[0][0]) / (timestamp - prior_visit[1]),
                                  (position[1] - prior_visit[0][1]) / (timestamp - prior_visit[1]))
            self.visit_samples[visit.visit_id] = (position, timestamp, visit_velocity)

    def mark_lost(self, track_id: int | str, position: tuple[float, float], timestamp: float, visit: Any,
                  inside_queue: bool, timeout_seconds: float | None = None,
                  appearance: Any = None, velocity: tuple[float, float] = (0.0, 0.0),
                  moving_away: bool = False) -> bool:
        """Buffer an eligible queue visit instead of ending it immediately."""
        if visit is None or getattr(getattr(visit, "state", None), "value", None) not in {"IN_QUEUE", "PENDING_ABANDONMENT", "AT_SERVICE_POINT"}:
            return False
        self.lost = [item for item in self.lost if not (item.track_id == track_id and item.visit is visit)]
        saved_appearance = appearance if appearance is not None else self.visit_appearance.get(visit.visit_id)
        self.lost.append(LostTrack(track_id, position, timestamp, visit, inside_queue, timeout_seconds,
                                   saved_appearance, velocity, moving_away))
        return True

    def recover(self, new_track_id: int | str, position: tuple[float, float], timestamp: float,
                appearance: Any = None, live_visit_ids: set[str] | None = None,
                allow_closed_visit_ids: set[str] | None = None) -> LostTrack | None:
        """Match by predicted position, appearance, direction and single-live-owner rule."""
        live_visit_ids = live_visit_ids or set()
        allow_closed_visit_ids = allow_closed_visit_ids or set()
        candidates = [item for item in self.lost if (not getattr(item.visit, "queue_closed", False) or item.visit.visit_id in allow_closed_visit_ids)
                      and timestamp - item.timestamp <= (item.timeout_seconds or self.timeout) and timestamp >= item.timestamp]
        scored = []
        for item in candidates:
            dt = max(0.0, timestamp - item.timestamp)
            predicted = (item.position[0] + item.velocity[0] * dt, item.position[1] + item.velocity[1] * dt)
            distance = hypot(position[0] - predicted[0], position[1] - predicted[1])
            app = self.appearance_similarity(item.appearance, appearance)
            new_velocity = self.track_samples.get(new_track_id, (position, timestamp, (0, 0), appearance))[2]
            if hypot(*new_velocity) < 1e-3 and dt > 1e-6:
                new_velocity = ((position[0] - item.position[0]) / dt, (position[1] - item.position[1]) / dt)
            angle = self._direction_angle(item.velocity, new_velocity)
            reason = "accepted"
            if item.visit.visit_id in live_visit_ids:
                reason = "rejected: visit already has a live track"
            elif item.moving_away:
                reason = "rejected: lost while moving away from queue"
            elif distance > self.distance:
                reason = "rejected: predicted position too far"
            elif item.appearance is not None and (app is None or app < self.appearance_min_similarity):
                reason = "rejected: appearance mismatch"
            elif angle is not None and angle > self.max_direction_angle:
                reason = "rejected: direction mismatch"
            accepted = reason == "accepted"
            row = {"old_track_id": item.track_id, "new_track_id": new_track_id, "timestamp": float(timestamp),
                   "distance_px": float(distance), "appearance_similarity": app,
                   "direction_angle_degrees": angle, "accepted": accepted, "decision": reason,
                   "visit_id": item.visit.visit_id}
            self.decisions.append(row)
            if accepted:
                scored.append((distance, 1.0 - (app if app is not None else 0.0), item))
        if not scored:
            self.expire(timestamp)
            return None
        match = min(scored, key=lambda item: (item[0], item[1]))[2]
        self.lost.remove(match)
        if new_track_id != match.track_id:
            if match.track_id not in match.visit.previous_track_ids:
                match.visit.previous_track_ids.append(match.track_id)
            match.visit.track_id = new_track_id
            self.links.append({"old_track_id": match.track_id, "new_track_id": new_track_id,
                               "timestamp": float(timestamp), "visit_id": match.visit.visit_id})
        self.visit_appearance.setdefault(match.visit.visit_id, match.appearance)
        return match

    @staticmethod
    def _direction_angle(first: tuple[float, float], second: tuple[float, float]) -> float | None:
        a = hypot(*first); b = hypot(*second)
        if a < 1e-3 or b < 1e-3:
            return None
        cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / (a * b)))
        return degrees(acos(cosine))

    def reconcile_live(self, active: dict, signatures: dict, timestamp: float,
                       positions: dict | None = None) -> list[dict[str, Any]]:
        """Correct a two-track tracker-ID swap when crossed appearance evidence is clear."""
        owners = [tid for tid in active if tid in signatures and active[tid].visit_id in self.visit_appearance]
        positions = positions or {}
        swaps = []
        for index, first_id in enumerate(owners):
            for second_id in owners[index + 1:]:
                first_visit, second_visit = active[first_id], active[second_id]
                if first_visit is second_visit:
                    continue
                own = (self.appearance_similarity(self.visit_appearance[first_visit.visit_id], signatures[first_id]),
                       self.appearance_similarity(self.visit_appearance[second_visit.visit_id], signatures[second_id]))
                cross = (self.appearance_similarity(self.visit_appearance[first_visit.visit_id], signatures[second_id]),
                         self.appearance_similarity(self.visit_appearance[second_visit.visit_id], signatures[first_id]))
                own_distance = cross_distance = None
                first_motion = self.visit_samples.get(first_visit.visit_id); second_motion = self.visit_samples.get(second_visit.visit_id)
                if first_id in positions and second_id in positions and first_motion and second_motion:
                    dt_first = max(0.0, timestamp - first_motion[1]); dt_second = max(0.0, timestamp - second_motion[1])
                    predicted_first = (first_motion[0][0] + first_motion[2][0] * dt_first, first_motion[0][1] + first_motion[2][1] * dt_first)
                    predicted_second = (second_motion[0][0] + second_motion[2][0] * dt_second, second_motion[0][1] + second_motion[2][1] * dt_second)
                    own_distance = hypot(positions[first_id][0] - predicted_first[0], positions[first_id][1] - predicted_first[1]) + hypot(positions[second_id][0] - predicted_second[0], positions[second_id][1] - predicted_second[1])
                    cross_distance = hypot(positions[second_id][0] - predicted_first[0], positions[second_id][1] - predicted_first[1]) + hypot(positions[first_id][0] - predicted_second[0], positions[first_id][1] - predicted_second[1])
                position_supports_swap = cross_distance is None or cross_distance < own_distance
                possible = all(value is not None for value in cross) and min(cross) >= self.appearance_min_similarity and sum(cross) > sum(own) + 0.05 and position_supports_swap
                if not possible:
                    continue
                accepted = all(value is not None for value in own) and min(own) < self.appearance_min_similarity
                row = {"old_track_id": first_id, "new_track_id": second_id, "timestamp": float(timestamp),
                       "distance_px": cross_distance, "appearance_similarity": min(cross) if all(v is not None for v in cross) else None,
                       "direction_angle_degrees": None, "accepted": bool(accepted),
                       "decision": "accepted: live appearance swap" if accepted else "rejected: no clear appearance swap",
                       "visit_id": first_visit.visit_id}
                self.decisions.append(row)
                if accepted:
                    active[first_id], active[second_id] = second_visit, first_visit
                    swaps.append(row)
        return swaps

    def expire(self, timestamp: float) -> list[LostTrack]:
        """Drop buffers whose timeout elapsed and report likely exits/occlusions."""
        expired = [item for item in self.lost if timestamp - item.timestamp > (item.timeout_seconds or self.timeout)]
        self.lost = [item for item in self.lost if item not in expired]
        return expired

    @staticmethod
    def likely_departure(position: tuple[float, float], frame_size: tuple[int, int], inside_queue: bool, edge_margin: int) -> bool:
        """Treat edge disappearance as departure, interior queue loss as occlusion."""
        width, height = frame_size
        at_edge = position[0] <= edge_margin or position[1] <= edge_margin or position[0] >= width - edge_margin or position[1] >= height - edge_margin
        return at_edge or not inside_queue

@dataclass
class StaffCandidate:
    """Track early service-area dwell before confirming a staff identity."""
    started_at: float
    last_seen: float
    position: tuple[float, float]
    path_length: float = 0.0

class StaffRegistry:
    """Mark persistent counter-side tracks as staff and inherit brief ID changes."""
    def __init__(self, ignored_ids: list[int], dwell_seconds: float, max_speed: float, timeout_seconds: float, distance_px: float,
                 initial_dwell_seconds: float = 2.0):
        """Initialize configured staff IDs and staff-classification limits."""
        self.staff_ids = {int(track_id) for track_id in ignored_ids}
        self.staff_rules: dict[int, str] = {int(track_id): "ignore_track_ids" for track_id in ignored_ids}
        self.service_first_seen: set[int] = set()
        self.ever_queued: set[int] = set()
        self.staff_positions: dict[int, tuple[tuple[float, float], float]] = {}
        self.candidates: dict[int, StaffCandidate] = {}
        self.initial_candidates: dict[int, StaffCandidate] = {}
        self.dwell_seconds = float(dwell_seconds)
        self.max_speed = float(max_speed)
        self.timeout = float(timeout_seconds)
        self.distance = float(distance_px)
        self.initial_dwell = float(initial_dwell_seconds)

    def mark_queued(self, track_id: int) -> None:
        """Remember that this track has entered a queue visit."""
        self.ever_queued.add(int(track_id))

    def blocks_queue(self, track_id: int) -> bool:
        """Return whether this counter-first track must not join the queue."""
        track_id = int(track_id)
        return track_id in self.staff_ids or track_id in self.service_first_seen

    def observe(self, track_id: int, position: tuple[float, float], now: float, in_service_box: bool,
                has_queue_visit: bool, visible_ids: set[int], first_video_frame: bool = False) -> bool:
        """Update staff evidence and return true for confirmed or ignored staff."""
        track_id = int(track_id)
        if track_id in self.staff_ids:
            self.staff_positions[track_id] = (position, now)
            return True
        if has_queue_visit or track_id in self.ever_queued:
            self.mark_queued(track_id)
            return False
        inherited = self._inherit_staff(track_id, position, now, visible_ids)
        if inherited:
            self.staff_rules[track_id] = "rule3_inherited_lost_staff_track"
            return True
        if not in_service_box:
            self.candidates.pop(track_id, None)
            self.initial_candidates.pop(track_id, None)
            return False

        self.service_first_seen.add(track_id)
        if first_video_frame:
            self.initial_candidates[track_id] = StaffCandidate(now, now, position)
        initial = self.initial_candidates.get(track_id)
        if initial is not None:
            initial.path_length += hypot(position[0] - initial.position[0], position[1] - initial.position[1])
            initial.last_seen = now
            initial.position = position
            speed = initial.path_length / max(now - initial.started_at, 1e-6)
            if now - initial.started_at >= self.initial_dwell and speed <= self.max_speed:
                self.staff_ids.add(track_id)
                self.staff_rules[track_id] = "rule1_present_in_service_on_first_video_frame_and_stayed"
                self.staff_positions[track_id] = (position, now)
                self.initial_candidates.pop(track_id, None)
                self.candidates.pop(track_id, None)
                return True
        candidate = self.candidates.get(track_id)
        if candidate is None:
            candidate = self._inherit_candidate(track_id, position, now, visible_ids)
        if candidate is None or now - candidate.last_seen > self.timeout:
            candidate = StaffCandidate(now, now, position)
        else:
            candidate.path_length += hypot(position[0] - candidate.position[0], position[1] - candidate.position[1])
            candidate.last_seen = now
            candidate.position = position
        self.candidates[track_id] = candidate
        elapsed = now - candidate.started_at
        speed = candidate.path_length / max(elapsed, 1e-6)
        if self.dwell_seconds > 0 and elapsed >= self.dwell_seconds and speed <= self.max_speed:
            self.staff_ids.add(track_id)
            self.staff_rules[track_id] = "rule2_long_service_dwell"
            self.staff_positions[track_id] = (position, now)
            self.candidates.pop(track_id, None)
            return True
        return False

    def _inherit_staff(self, track_id: int, position: tuple[float, float], now: float, visible_ids: set[int]) -> bool:
        """Inherit STAFF only from a recently missing staff track nearby."""
        matches = []
        for old_id, (old_position, seen_at) in self.staff_positions.items():
            if old_id == track_id or old_id in visible_ids or now - seen_at > self.timeout or now < seen_at:
                continue
            distance = hypot(position[0] - old_position[0], position[1] - old_position[1])
            if distance <= self.distance:
                matches.append((distance, old_id, old_position))
        if not matches:
            return False
        _, old_id, _ = min(matches)
        self.staff_ids.add(track_id)
        self.staff_positions[track_id] = (position, now)
        return True

    def _inherit_candidate(self, track_id: int, position: tuple[float, float], now: float, visible_ids: set[int]) -> StaffCandidate | None:
        """Carry early staff dwell across a brief detector ID switch."""
        matches = []
        for old_id, candidate in self.candidates.items():
            if old_id in visible_ids or now - candidate.last_seen > self.timeout or now < candidate.last_seen:
                continue
            distance = hypot(position[0] - candidate.position[0], position[1] - candidate.position[1])
            if distance <= self.distance:
                matches.append((distance, old_id, candidate))
        if not matches:
            return None
        _, old_id, candidate = min(matches, key=lambda item: item[0])
        self.candidates.pop(old_id, None)
        return candidate
