"""Draw readable, low-clutter queue annotations and a compact legend."""
from typing import Any
import cv2
import numpy as np
from .state_machine import State

PERSON_COLORS = {
    "candidate": (150, 150, 150),
    State.IN_QUEUE: (35, 190, 35),
    State.AT_SERVICE_POINT: (210, 110, 20),
    State.PENDING_ABANDONMENT: (0, 145, 255),
    State.ABANDONED: (0, 0, 220),
    State.CANCELLED: (190, 70, 190),
    State.DONE: (120, 120, 120),
    "staff": (90, 90, 90),
    "counter": (220, 90, 25),
    "walkup": (90, 170, 210),
    "unconfirmed": (145, 145, 145),
    "ambiguous": (0, 190, 255),
}
ROI_COLORS = {"queue": (230, 210, 40), "service": (220, 70, 220), "staff_zone": (140, 140, 140)}

def _polygons(value: list[Any]) -> list[list[list[int]]]:
    """Normalize one polygon or multiple disjoint polygons for drawing."""
    if not value:
        return []
    if len(value) >= 3 and isinstance(value[0], (list, tuple)) and len(value[0]) == 2 and all(isinstance(v, (int, float)) for v in value[0]):
        return [value]
    return list(value)

def panel_text_fits(lines: list[str], panel_width: int, font_scale: float, thickness: int, padding: int) -> bool:
    """Check that every line fits inside a padded counter panel."""
    return all(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= panel_width - 2 * padding for line in lines)

def _intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    """Return true if two label rectangles overlap."""
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]

def displayed_person_states(detections: list[dict[str, Any]], visits: dict[Any, Any], staff_track_ids: set[int] | None = None,
                            walkup_track_ids: set[int] | None = None,
                            unconfirmed_track_ids: set[int] | None = None,
                            ambiguous_track_ids: set[int] | None = None) -> dict[int, Any]:
    """Return exactly the state key used to color each detected person's box."""
    staff_ids = staff_track_ids or set()
    walkup_ids = walkup_track_ids or set()
    unconfirmed_ids = unconfirmed_track_ids or set()
    ambiguous_ids = ambiguous_track_ids or set()
    states: dict[int, Any] = {}
    for det in detections:
        track_id = int(det["track_id"])
        visit = visits.get(track_id)
        if track_id in staff_ids:
            states[track_id] = "staff"
        elif track_id in ambiguous_ids:
            states[track_id] = "ambiguous"
        elif track_id in unconfirmed_ids:
            states[track_id] = "unconfirmed"
        elif visit is not None:
            states[track_id] = visit.state
        elif track_id in walkup_ids:
            states[track_id] = "walkup"
        else:
            states[track_id] = "candidate"
    return states

def displayed_state_counts(states: dict[int, Any]) -> dict[str, int]:
    """Count visible detections by the same state keys used by person boxes."""
    return {
        "in_queue": sum(state in {State.IN_QUEUE, "counter"} for state in states.values()),
        "served": sum(state == State.AT_SERVICE_POINT for state in states.values()),
        "pending": sum(state == State.PENDING_ABANDONMENT for state in states.values()),
        "abandoned": sum(state == State.ABANDONED for state in states.values()),
        "candidate": sum(state == "candidate" for state in states.values()),
        "cancelled": sum(state == State.CANCELLED for state in states.values()),
        "staff_tracks": sum(state == "staff" for state in states.values()),
        "walkup": sum(state == "walkup" for state in states.values()),
        "unconfirmed": sum(state in {"unconfirmed", "ambiguous"} for state in states.values()),
        "ambiguous": sum(state == "ambiguous" for state in states.values()),
    }

def counter_panel_lines(visible_counts: dict[str, int], counters: dict[str, int], pending_suffix: str = "") -> list[str]:
    """Build visible-state counts and persistent event totals for the panel."""
    total = counters.get('served_total', visible_counts['served'])
    lines = [f"In queue: {visible_counts['in_queue']}", f"Served (total): {total} (queue {counters.get('served_queue_total', total)}, direct {counters.get('served_direct_total', 0)})",
            f"Pending: {visible_counts['pending']}{pending_suffix}",
            f"Abandoned (total): {counters.get('abandoned_total', visible_counts['abandoned'])}",
            f"Cancelled (total): {counters.get('cancelled_total', visible_counts['cancelled'])}"]
    return lines

class Annotator:
    """Render polygons, tracks, foot points, counters and a color legend."""
    def __init__(self, config: dict[str, Any], rois: list[dict[str, Any]], fps: float, width: int = 0, height: int = 0):
        """Cache output style, ROI polygons and video rate."""
        self.config = config
        self.rois = rois
        self.fps = fps
        self.width = width; self.height = height

    def _counter_layout(self, lines: list[str], frame_width: int, scale: float) -> tuple[float, int, int, int, int, list[int]]:
        """Fit the counter box width and height to all displayed text."""
        padding = max(7, int(10 * scale))
        thickness = max(1, int(round(scale)))
        font = max(0.30, float(self.config.get("font_scale", 0.55)) * 0.87 * scale)
        max_allowed = max(1, frame_width - 2 * padding - 16)
        while max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0] for line in lines) > max_allowed and font > 0.22:
            font *= 0.90
        widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0] for line in lines]
        panel_width = max(widths, default=0) + 2 * padding
        if not panel_text_fits(lines, panel_width, font, thickness, padding):
            raise ValueError("Counter panel text exceeds its padded width.")
        heights = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][1] for line in lines]
        row_height = max(heights, default=1) + max(5, int(8 * scale))
        panel_height = len(lines) * row_height + padding
        return font, thickness, padding, panel_width, panel_height, heights

    def _draw_panel(self, image: Any, lines: list[str], scale: float) -> tuple[int, int, int, int]:
        """Draw a dynamically sized translucent counter panel."""
        height, width = image.shape[:2]
        font, thickness, padding, panel_width, panel_height, text_heights = self._counter_layout(lines, width, scale)
        x = max(4, int(10 * scale)); y = max(4, int(10 * scale))
        panel_width = min(panel_width, width - 2 * x)
        if not panel_text_fits(lines, panel_width, font, thickness, padding):
            raise ValueError("Counter panel text is wider than the panel.")
        panel_height = min(panel_height, height - 2 * y)
        patch = image[y:y + panel_height + 1, x:x + panel_width + 1]
        overlay = patch.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (25, 28, 32), -1)
        cv2.addWeighted(overlay, float(self.config["panel_alpha"]), patch, 1.0 - float(self.config["panel_alpha"]), 0, patch)
        baseline = y + padding + text_heights[0]
        row_gap = max(5, int(8 * scale))
        for line, text_height in zip(lines, text_heights):
            cv2.putText(image, line, (x + padding, baseline), cv2.FONT_HERSHEY_SIMPLEX, font, (250, 250, 250), thickness, cv2.LINE_AA)
            baseline += text_height + row_gap
        return x, y, x + panel_width, y + panel_height

    def _draw_legend(self, image: Any, scale: float, used_states: set[Any]) -> tuple[int, int, int, int]:
        """Draw every person and ROI color in a separate bottom-left box."""
        height, width = image.shape[:2]
        labels = {"candidate": "Candidate", "counter": "At counter", State.IN_QUEUE: "In queue", State.AT_SERVICE_POINT: "At service", State.DONE: "Done",
                  State.PENDING_ABANDONMENT: "Pending", State.ABANDONED: "Abandoned", State.CANCELLED: "Cancelled",
                  "staff": "Staff", "walkup": "Walk-up", "ambiguous": "Ambiguous", "unconfirmed": "Warming track"}
        order = ["candidate", "counter", State.IN_QUEUE, State.AT_SERVICE_POINT, State.PENDING_ABANDONMENT,
                 State.ABANDONED, State.CANCELLED, State.DONE, "staff", "walkup", "ambiguous", "unconfirmed"]
        entries = [(PERSON_COLORS[state], labels[state]) for state in order if state in used_states]
        entries.extend([(ROI_COLORS["queue"], "Queue ROI"), (ROI_COLORS["service"], "Service ROI")])
        if any(roi.get("staff_zone") for roi in self.rois):
            entries.append((ROI_COLORS["staff_zone"], "Staff area"))
        font = max(0.28, float(self.config.get("font_scale", 0.55)) * 0.73 * scale); thickness = max(1, int(round(scale)))
        pad = max(6, int(9 * scale)); swatch = max(8, int(12 * scale)); gap = max(4, int(7 * scale))
        label_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][0] for _, label in entries)
        row_height = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][1] for _, label in entries) + gap
        box_width = min(width - 16, pad * 3 + swatch + label_width)
        box_height = min(height - 16, 2 * pad + len(entries) * row_height)
        x = 8; y = max(8, height - box_height - 8)
        patch = image[y:y + box_height + 1, x:x + box_width + 1]
        overlay = patch.copy(); cv2.rectangle(overlay, (0, 0), (box_width, box_height), (25, 28, 32), -1)
        cv2.addWeighted(overlay, float(self.config["panel_alpha"]), patch, 1.0 - float(self.config["panel_alpha"]), 0, patch)
        baseline = y + pad + cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0][1]
        for color, label in entries:
            swatch_y = baseline - swatch + max(1, int(2 * scale))
            cv2.rectangle(image, (x + pad, swatch_y), (x + pad + swatch, baseline), color, -1)
            cv2.putText(image, label, (x + 2 * pad + swatch, baseline), cv2.FONT_HERSHEY_SIMPLEX, font, (250, 250, 250), thickness, cv2.LINE_AA)
            baseline += row_height
        return x, y, x + box_width, y + box_height

    def _label_position(self, box: list[int], text_size: tuple[int, int], frame_size: tuple[int, int], reserved: list[tuple[int, int, int, int]], occupied: list[tuple[int, int, int, int]], padding: int) -> tuple[int, int, tuple[int, int, int, int]]:
        """Place a label above its box, shifting upward around earlier labels."""
        frame_height, frame_width = frame_size
        x1, y1, x2, y2 = box
        text_width, text_height = text_size
        label_width = min(frame_width - 4, text_width + 2 * padding)
        label_height = text_height + 2 * padding
        x = min(max(2, x1), max(2, frame_width - label_width - 2))
        step = label_height + max(2, padding // 2)
        candidates = [y1 - label_height - 3 - index * step for index in range(8)]
        candidates.extend(y2 + 3 + index * step for index in range(8))
        for y in candidates:
            y = min(max(2, y), max(2, frame_height - label_height - 2))
            rect = (x, y, x + label_width, y + label_height)
            if not any(_intersects(rect, other) for other in [*reserved, *occupied]):
                return x, y, rect
        y = min(max(2, y1 - label_height - 3), max(2, frame_height - label_height - 2))
        rect = (x, y, x + label_width, y + label_height)
        return x, y, rect

    def draw(self, frame: Any, detections: list[dict[str, Any]], visits: dict[Any, Any], now: float, counters: dict[str, int], alerts: list[Any], staff_track_ids: set[int] | None = None, candidate_track_ids: set[int] | None = None, feet: dict[int, tuple[float, float]] | None = None, walkup_track_ids: set[int] | None = None, unconfirmed_track_ids: set[int] | None = None, ambiguous_track_ids: set[int] | None = None, frame_index: int = 0, visit_labels: dict[int, str] | None = None, direct_track_ids: set[int] | None = None, hidden_track_ids: set[int] | None = None, served_tags: dict[int, float] | None = None, served_tag_labels: dict[int, str] | None = None, counter_markers: dict[int, tuple[float, float]] | None = None) -> Any:
        """Draw thin ROIs, state boxes, foot points, fitted counters and legend."""
        image = frame.copy(); height, width = image.shape[:2]
        scale = max(0.45, min(2.0, min(width / 1280.0, height / 720.0)))
        staff_ids = staff_track_ids or set(); foot_points = feet or {}; labels_by_track = visit_labels or {}
        direct_ids = direct_track_ids or set(); hidden_ids = hidden_track_ids or set(); served_tag_ids = served_tags or {}
        tag_labels = served_tag_labels or {}
        render_detections = [det for det in detections if int(det["track_id"]) not in hidden_ids]
        state_by_track = displayed_person_states(render_detections, visits, staff_ids, walkup_track_ids, unconfirmed_track_ids, ambiguous_track_ids)
        for tid, state in list(state_by_track.items()):
            visit = visits.get(tid)
            if tid not in staff_ids and visit is not None and getattr(visit, "counter_arrival_time", None) is not None and getattr(visit, "counter_leave_time", None) is None:
                state_by_track[tid] = "counter"
        for tid in direct_ids:
            state_by_track[tid] = State.AT_SERVICE_POINT if tid in visits and visits[tid].state == State.AT_SERVICE_POINT else "candidate"
        visible_counts = displayed_state_counts(state_by_track)
        for tid, state in state_by_track.items():
            if state == "ambiguous" and tid in visits:
                if visits[tid].state == State.IN_QUEUE:
                    visible_counts["in_queue"] += 1
                elif visits[tid].state == State.PENDING_ABANDONMENT:
                    visible_counts["pending"] += 1

        # Light transparent fills make both ROIs visible without hiding people.
        fill = image.copy()
        for roi in self.rois:
            for name in ("queue", "service", "staff_zone"):
                if name == "staff_zone" and not roi.get(name): continue
                cv2.fillPoly(fill, [np.asarray(poly, dtype=np.int32) for poly in _polygons(roi[name])], ROI_COLORS[name])
        cv2.addWeighted(fill, 0.10, image, 0.90, 0, image)
        for roi in self.rois:
            for name in ("queue", "service", "staff_zone"):
                if name == "staff_zone" and not roi.get(name): continue
                roi_thickness = max(1, int(round(float(self.config.get("line_thickness", 2)) * 0.5 * scale)))
                cv2.polylines(image, [np.asarray(poly, dtype=np.int32) for poly in _polygons(roi[name])], True, ROI_COLORS[name], roi_thickness)
            anchor = roi.get("service_anchor")
            if anchor:
                cv2.drawMarker(image, tuple(map(int, anchor)), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            if roi.get("staff_zone"):
                first_staff_polygon = _polygons(roi["staff_zone"])[0]
                x, y = map(int, first_staff_polygon[0])
                cv2.putText(image, "STAFF AREA", (x + 2, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, ROI_COLORS["staff_zone"], 1, cv2.LINE_AA)

        pending_visit_ids = {visits[track_id].visit_id for track_id, state in state_by_track.items()
                             if state == State.PENDING_ABANDONMENT and visits[track_id].pending_start_time is not None}
        pending_visits = [visits[track_id] for track_id, state in state_by_track.items()
                          if state == State.PENDING_ABANDONMENT and visits[track_id].visit_id in pending_visit_ids]
        pending_visits = list({visit.visit_id: visit for visit in pending_visits}.values())
        pending_suffix = ""
        if pending_visits:
            next_expiry = min(float(self.config["grace_period_seconds"]) - (now - float(visit.pending_start_time)) for visit in pending_visits)
            pending_suffix = f" | {max(0.0, next_expiry):.1f}s"
        counter_lines = counter_panel_lines(visible_counts, counters, pending_suffix)
        counter_rect = self._draw_panel(image, counter_lines, scale)
        legend_rect = self._draw_legend(image, scale, set(state_by_track.values()))
        reserved = [counter_rect, legend_rect]
        occupied: list[tuple[int, int, int, int]] = []
        text_base = max(0.30, float(self.config.get("font_scale", 0.55)) * 0.87 * scale)
        padding = max(2, int(4 * scale))

        drawn_tracks: set[int] = set()
        for tid, expires in served_tag_ids.items():
            if now <= expires:
                det = next((item for item in detections if int(item["track_id"]) == tid), None)
                if det:
                    label = tag_labels.get(tid, "SERVED")
                    font = 0.40 * scale; thickness = max(1, int(scale)); pad = max(3, int(4 * scale))
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)
                    x = min(width - tw - 2 * pad - 2, max(2, int(det["box"][0])))
                    y = min(height - th - 2 * pad - 2, max(2, int(det["box"][1]) - th - 2 * pad))
                    box = [int(value) for value in det["box"]]
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (75, 75, 75), max(1, int(2 * scale)))
                    cv2.rectangle(image, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), (35, 38, 42), -1)
                    cv2.putText(image, label, (x + pad, y + th + pad), cv2.FONT_HERSHEY_SIMPLEX, font, (245, 245, 245), thickness, cv2.LINE_AA)
        for det in detections:
            tid = int(det["track_id"]); box = [int(value) for value in det["box"]]
            if tid in drawn_tracks:
                continue
            drawn_tracks.add(tid)
            if tid in hidden_ids: continue
            visit = visits.get(tid)
            state_key = state_by_track[tid]
            if state_key == "staff":
                label = "STAFF"
            elif state_key == "walkup":
                label = f"ID {tid}" if tid <= 99 else ""
            elif state_key == "ambiguous":
                label = ""
            elif state_key == "unconfirmed":
                label = f"ID {tid}" if tid <= 99 else ""
            elif visit is not None:
                if state_key == "counter":
                    counter_seconds = max(0.0, now - float(visit.counter_arrival_time))
                    label = f"AT COUNTER {counter_seconds:.1f}s"
                elif visit.state == State.PENDING_ABANDONMENT and visit.pending_start_time is not None:
                    remaining = max(0.0, float(self.config["grace_period_seconds"]) - (now - float(visit.pending_start_time)))
                    label = f"PENDING {remaining:.1f}s"
                elif visit.state == State.DONE:
                    label = "DONE"
                elif tid in direct_ids and visit.state == State.AT_SERVICE_POINT:
                    label = labels_by_track.get(tid, "DIRECT")
                else:
                    queue_entry_time = getattr(visit, "queue_entry_time", None)
                    if visit.state == State.AT_SERVICE_POINT:
                        service_start = getattr(visit, "service_touched_time", None)
                        counter_seconds = max(0.0, now - float(service_start if service_start is not None else now))
                        label = labels_by_track.get(tid, f"AT SERVICE {counter_seconds:.1f}s")
                    else:
                        wait = max(0.0, now - float(queue_entry_time or now))
                        label = f"{labels_by_track.get(tid, f'ID {tid}' if tid <= 99 else '')} {wait:.0f}s".strip()
            else:
                label = f"ID {tid}" if tid <= 99 else ""
            color = PERSON_COLORS[state_key]
            box_thickness = max(1, int(round(float(self.config.get("line_thickness", 2)) * scale)))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, box_thickness)
            if state_key == "ambiguous" or not label:
                continue
            foot = foot_points.get(tid, det.get("foot"))
            if foot is not None:
                center = (int(round(foot[0])), int(round(foot[1])))
                cv2.circle(image, center, max(2, int(round(3 * scale))), (255, 255, 255), -1)
                cv2.circle(image, center, max(2, int(round(3 * scale))), color, max(1, int(round(scale))))

            base_width = max(1, box[2] - box[0])
            font = text_base
            measured = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, max(1, int(round(scale))))[0][0]
            if measured > base_width:
                font *= max(0.52, base_width / measured)
            thickness = max(1, int(round(scale)))
            size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)
            label_padding = max(2, int(4 * scale))
            x, y, rect = self._label_position(box, size, (height, width), reserved, occupied, label_padding)
            occupied.append(rect)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
            text_color = (255, 255, 255) if state_key in {State.AT_SERVICE_POINT, State.PENDING_ABANDONMENT, State.ABANDONED, State.CANCELLED, "staff"} else (20, 20, 20)
            cv2.putText(image, label, (x + label_padding, y + size[1] + label_padding), cv2.FONT_HERSHEY_SIMPLEX, font, text_color, thickness, cv2.LINE_AA)

        for track_id, point in (counter_markers or {}).items():
            center = (int(round(point[0])), int(round(point[1])))
            cv2.drawMarker(image, center, PERSON_COLORS["counter"], cv2.MARKER_CROSS, max(16, int(22*scale)), max(2, int(2*scale)))
            cv2.putText(image, f"AT COUNTER ID {track_id}", (center[0]+8, center[1]-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42*scale, PERSON_COLORS["counter"], max(1, int(scale)), cv2.LINE_AA)

        stamp = f"{int(now // 60):02d}:{now % 60:04.1f}  F{frame_index:04d}"
        stamp_size = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.58 * scale, max(1, int(scale)))[0]
        sx = width - stamp_size[0] - 18; sy = 10
        cv2.rectangle(image, (sx - 8, sy), (width - 8, sy + stamp_size[1] + 16), (25, 28, 32), -1)
        cv2.putText(image, stamp, (sx, sy + stamp_size[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)

        if alerts:
            alert = alerts[-1]
            if isinstance(alert, dict):
                text_value = f"ABANDONED - {alert['visit_id']} - waited {alert['wait_seconds']:.0f} s"
                position = alert.get("position")
                if position is not None:
                    marker = (min(width - 18, max(18, int(round(position[0])))),
                              min(height - 18, max(18, int(round(position[1])))))
                    cv2.circle(image, marker, max(12, int(16 * scale)), (0, 0, 220), -1)
                    marker_font = max(0.42, 0.55 * scale); marker_thickness = max(1, int(round(scale)))
                    marker_text = cv2.getTextSize("ABANDONED", cv2.FONT_HERSHEY_SIMPLEX, marker_font, marker_thickness)[0]
                    mx = min(width - marker_text[0] - 3, max(3, marker[0] + 14))
                    my = min(height - 3, max(marker_text[1] + 3, marker[1] - 8))
                    cv2.putText(image, "ABANDONED", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, marker_font,
                                (0, 0, 220), marker_thickness, cv2.LINE_AA)
            else:
                text_value = f"ABANDONED - {alert.visit_id}"
            font = max(0.75, 0.95 * scale); thickness = max(2, int(round(2 * scale)))
            text_size = cv2.getTextSize(text_value, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)[0]
            pad = max(10, int(16 * scale)); x = max(2, (width - text_size[0] - 2 * pad) // 2); y = max(2, int(18 * scale))
            cv2.rectangle(image, (x, y), (x + text_size[0] + 2 * pad, y + text_size[1] + 2 * pad), (0, 0, 145), -1)
            cv2.putText(image, text_value, (x + pad, y + text_size[1] + pad), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thickness, cv2.LINE_AA)
        return image
